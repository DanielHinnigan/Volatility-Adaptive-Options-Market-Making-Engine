"""
LOB Snapshot data container.

Encapsulates a single timestamp's worth of Limit Order Book data for options.
Provides clean, type-safe access to spot prices and option quotes.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd


@dataclass(frozen=True)
class LOBQuote:
    """A single option quote from the LOB."""
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    volume: int
    open_interest: int
    implied_vol: float
    last_price: float


@dataclass(frozen=True)
class LOBSnapshot:
    """
    A snapshot of the Limit Order Book at a specific timestamp.

    Contains the spot price and quotes for all options.
    """

    # The expected columns for the DataFrame passed to from_dataframe()
    EXPECTED_COLUMNS: List[str] = [
        'timestamp',
        'spot_price',
        'expiry',
        'strike',
        'type',
        'bid',
        'ask',
        'bid_size',
        'ask_size',
        'volume',
        'open_interest',
        'implied_vol',
        'last_price',
    ]

    timestamp: datetime
    spot: float
    _data: pd.DataFrame  # Filtered to this timestamp

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'LOBSnapshot':
        """
        Create a LOBSnapshot from a DataFrame.

        The DataFrame must contain the columns defined in EXPECTED_COLUMNS.
        """
        if df.empty:
            raise ValueError("Cannot create LOBSnapshot from empty DataFrame")

        # Enforce the data contract
        missing = set(cls.EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in LOB data: {missing}. "
                f"Expected: {cls.EXPECTED_COLUMNS}"
            )

        timestamp = df['timestamp'].iloc[0]
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)

        spot = float(df['spot_price'].iloc[0])

        # Filter to only the expected columns (ignores extra columns)
        return cls(timestamp=timestamp, spot=spot, _data=df[cls.EXPECTED_COLUMNS])

    def get_quote(self, expiry: str, strike: float, option_type: str) -> Optional[LOBQuote]:
        """
        Get the LOB quote for a specific option at this timestamp.

        Returns:
            LOBQuote if found, else None.
        """
        row = self._data[
            (self._data['expiry'] == expiry) &
            (self._data['strike'] == strike) &
            (self._data['type'] == option_type)
        ]

        if row.empty:
            return None

        r = row.iloc[0]
        return LOBQuote(
            bid=float(r['bid']),
            ask=float(r['ask']),
            bid_size=int(r.get('bid_size', 0)),
            ask_size=int(r.get('ask_size', 0)),
            volume=int(r.get('volume', 0)),
            open_interest=int(r.get('open_interest', 0)),
            implied_vol=float(r.get('implied_vol', 0.0)),
            last_price=float(r.get('last_price', 0.0)),
        )

    def get_all_quotes(self) -> Dict[str, LOBQuote]:
        """
        Get all quotes grouped by option_id (expiry_strike_type).
        """
        quotes = {}
        for _, row in self._data.iterrows():
            key = f"{row['expiry']}_{row['strike']}_{row['type']}"
            quotes[key] = LOBQuote(
                bid=float(row['bid']),
                ask=float(row['ask']),
                bid_size=int(row.get('bid_size', 0)),
                ask_size=int(row.get('ask_size', 0)),
                volume=int(row.get('volume', 0)),
                open_interest=int(row.get('open_interest', 0)),
                implied_vol=float(row.get('implied_vol', 0.0)),
                last_price=float(row.get('last_price', 0.0)),
            )
        return quotes

    def get_expiries(self) -> List[str]:
        """Return all expiries available in this snapshot."""
        return self._data['expiry'].unique().tolist()

    def get_strikes(self, expiry: str) -> List[float]:
        """Return all strikes for a given expiry in this snapshot."""
        return self._data[self._data['expiry'] == expiry]['strike'].unique().tolist()