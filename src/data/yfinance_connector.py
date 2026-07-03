# Inteface with external data sources
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import time

from .base_connector import DataConnector, OptionQuote

class YFinanceConnector(DataConnector):
    """Data connector using yfinance"""
    
    def __init__(self, symbol: str = "SPY"):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self._last_request_time = 0
        self._min_interval = 0.01  # 10ms delay to avoid rate limits
    
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
    
    def get_chain_for_expiry(self, expiry: str) -> Dict[str, List[OptionQuote]]:
        """
        Fetches the full option chain for a specific expiry.
        Returns: {'calls': [OptionQuote, ...], 'puts': [OptionQuote, ...]}
        """
        self._rate_limit()
        try:
            chain = self.ticker.option_chain(expiry)
            calls = self._parse_df(chain.calls, "call")
            puts = self._parse_df(chain.puts, "put")
            return {"calls": calls, "puts": puts}
        except Exception as e:
            print(f"Failed to fetch chain for {expiry}: {e}")
            return {"calls": [], "puts": []}
    
    def _parse_df(self, df: pd.DataFrame, option_type: str) -> List[OptionQuote]:
        """Convert yfinance DataFrame to your standardized format."""
        quotes = []
        
        # yfinance often has columns: 'strike', 'bid', 'ask', 'impliedVolatility', 'volume', 'openInterest'
        # Note: yfinance uses 'impliedVolatility' but it's sometimes stale.
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
            # For now, pass NaN and let your BSM inversion handle it.
            if pd.isna(iv) or iv <= 0:
                iv = np.nan

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
                implied_vol=float(iv) if not pd.isna(iv) else 0.0,
                volume=volume,
                open_interest=int(row.get('openInterest', 0)),
                option_type=option_type
            ))
        # Sort by strike for cleaner surface fitting
        return sorted(quotes, key=lambda x: x.strike)
    
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
        expiries = self.get_available_expiries()
        if not expiries:
            return {}
        
        # Take the nearest N expiries
        expiries = expiries[:max_expiries]
        
        # Get current spot price (from underlying stock)
        hist = self.ticker.history(period="1d")
        spot = hist['Close'].iloc[-1] if not hist.empty else 0.0
        
        result = {}
        for exp_str in expiries:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            days_to_exp = (exp_date - datetime.now()).days
            T = max(days_to_exp / 365.0, 1/365.0)  # Minimum 1 day
            
            chain_data = self.get_chain_for_expiry(exp_str)
            if chain_data["calls"] or chain_data["puts"]:
                result[exp_str] = {
                    "T": T,
                    "calls": chain_data["calls"],
                    "puts": chain_data["puts"],
                    "spot": spot
                }
        
        return result