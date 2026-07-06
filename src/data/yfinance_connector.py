# Inteface with external data sources
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import time

from ..storage.base import StorageBackend
from .base_connector import DataConnector, OptionQuote
from ..storage.parquet_storage import ParquetStorage
from ..utils.time_utils import compute_time_to_expiry
from ..models.bsm import implied_volatility

class YFinanceConnector(DataConnector):
    """Data connector using yfinance"""
    def __init__(self, symbol: str = "SPY", storage: StorageBackend = None):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self._last_request_time = 0
        self._min_interval = 0.01  # 10ms delay to avoid rate limits. The delay may be set to a lower value
        
        # Storage defaults to Parquet (backwards compatible)
        if storage is None:
            storage = ParquetStorage()
        self.storage = storage
    
    def _rate_limit(self):
        """Avoid hitting Yahoo's rate limits"""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()
    
    def get_available_expiries(self) -> List[str]:
        """Returns list of expiration dates (e.g., ['2025-12-19', ...])."""
        self._rate_limit()
        try:
            expiries = self.ticker.options
            # yfinance returns strings like '2025-12-19'
            return expiries
        except Exception as e:
            print(f"Failed to fetch expiries: {e}")
            return []
    
    def get_chain_for_expiry(self, expiry: str, use_cache: bool = True) -> Dict[str, List[OptionQuote]]:
        """
        Fetches the full option chain for a specific expiry.
        Returns: {'calls': [OptionQuote, ...], 'puts': [OptionQuote, ...]}
        """
        if use_cache and self.storage.expiry_exists(self.symbol, expiry):
            return self.storage.load_expiry(self.symbol, expiry)

        self._rate_limit()
        try:
            # Parse data from yfinance to our own format
            chain = self.ticker.option_chain(expiry)
            calls = self._parse_df(chain.calls, "call", expiry)
            puts = self._parse_df(chain.puts, "put", expiry)

            # Save the chain
            self.storage.save_expiry(self.symbol, expiry, calls, puts)

            return {"calls": calls, "puts": puts}
        except Exception as e:
            print(f"Failed to fetch chain for {expiry}: {e}")
            return {"calls": [], "puts": []}
    
    def _parse_df(self, df: pd.DataFrame, option_type: str, expiry: str) -> List[OptionQuote]:
        """Convert yfinance DataFrame to your standardized format."""
        quotes = []
        
        # yfinance often has columns: 'strike', 'bid', 'ask', 'impliedVolatility', 'volume', 'openInterest'
        # Note: yfinance uses 'impliedVolatility' but it's sometimes stale.
        spot = self.get_spot_price()
        T = compute_time_to_expiry(expiry)
        for _, row in df.iterrows():

            strike = row.get('strike')
            bid = row.get('bid', 0.0)
            ask = row.get('ask', 0.0)
            
            # Filter out garbage: zero bid/ask or huge spreads
            if bid <= 0 or ask <= 0 or (ask - bid) / ((ask + bid) / 2) > 0.5:
                continue
            
            mid = (bid + ask) / 2
            iv = row.get('impliedVolatility', np.nan)

            # If yfinance didn't provide IV, we will compute it later in the pipeline.
            if pd.isna(iv) or iv <= 0:
                iv = implied_volatility(mid, spot, strike, T, 0.05, 0, option_type)

            # yfinance defines zero volume as NaN, so we must convert into zero
            volume = row.get("volume")
            if pd.isna(volume):
                volume = 0
            else:
                volume = int(volume)

            quotes.append(OptionQuote(
                strike=float(strike),
                bid=float(bid),
                ask=float(ask),
                mid=float(mid),
                implied_vol=iv,
                volume=volume,
                open_interest=int(row.get('openInterest')),
                option_type=option_type
            ))
        # Sort by strike for cleaner surface fitting
        return sorted(quotes, key=lambda x: x.strike)
    
    def get_spot_price(self) -> float:
        """
        Fetch the latest closing/current price for the underlying.
    
        Returns:
            float: Current spot price.
        
        Raises:
            ValueError: If yfinance fails to fetch the price or returns an empty DataFrame.
        """
        hist = self.ticker.history(period="1d")
        if hist.empty:
            raise ValueError(
            f"Failed to fetch spot price for {self.symbol} from yfinance. "
            "The history DataFrame is empty. Check your internet connection, "
            "the symbol ticker, or yfinance's availability."
        )
        return float(hist['Close'].iloc[-1])

    def get_surface_data(self, max_expiries: int = 5) -> Dict:
        """
        High-level method to fetches the N nearest expiries. 
        Returns cleaned data, which can be used for fitting the volatility surface.
        
        Output format:
        {
            'expiry_date': {
                'T': years_to_expiry,
                'calls': [OptionQuote],
                'puts': [OptionQuote],
                'spot': current_spot_price
            }
        }
        """
        # Fetch expiries
        expiries = self.get_available_expiries()
        if not expiries:
            return {}
        
        # Take the nearest N expiries
        expiries = expiries[:max_expiries]

        # Get current spot price (from underlying stock)
        spot = self.get_spot_price()
        
        result = {}
        for exp_str in expiries:
            T = compute_time_to_expiry(exp_str)

            chain_data = self.get_chain_for_expiry(exp_str)
            if chain_data["calls"] or chain_data["puts"]:
                result[exp_str] = {
                    "T": T,
                    "calls": chain_data["calls"],
                    "puts": chain_data["puts"],
                    "spot": spot
                }
        
        return result