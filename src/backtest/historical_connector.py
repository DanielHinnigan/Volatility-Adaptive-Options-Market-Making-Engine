"""
Historical Data Connector for the Backtester.

This connector wraps LOB data and serves spot prices and option chains
from the historical Parquet file, allowing the PricingEngine to operate
in a backtest environment without any modifications.

It implements the DataConnector interface, so the PricingEngine treats it
exactly like a live data feed.

Usage:
    connector = HistoricalConnector(lob_data)
    connector.set_timestamp(timestamp)
    spot = connector.get_spot_price()
    chain = connector.get_chain_for_expiry(expiry)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ..data.base_connector import DataConnector, OptionQuote
from ..data.lob_snapshot import LOBSnapshot, EXPECTED_COLUMNS

from ..config import settings

logger = logging.getLogger(__name__)


class HistoricalConnector(DataConnector):
    """
    A data connector that serves historical data from a Parquet file.

    This connector is used by the PricingEngine during backtesting.
    It returns the spot price and option chain for the current timestamp
    being replayed.

    The connector maintains an internal pointer to the current timestamp.
    Each call to `get_spot_price()` or `get_chain_for_expiry()` returns
    data for that timestamp. The timestamp is advanced by calling set_timestamp()
    """

    def __init__(self, data: pd.DataFrame, initial_timestamp: Optional[datetime] = None):
        """
        Initialize the historical connector.

        Args:
            data: The LOB DataFrame (must contain columns matching EXPECTED_COLUMNS).
            initial_timestamp: Optional starting timestamp. If None, uses the
                               first timestamp in the data.

        Raises:
            ValueError: If the data does not contain the required columns or
                        if no valid timestamps are found.
        """
        # Validate the data against the LOBSnapshot schema
        self._validate_data(data)

        # Build snapshots for each timestamp
        self._snapshots: Dict[datetime, LOBSnapshot] = {}
        for ts, group in data.groupby('timestamp'):
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
            try:
                self._snapshots[ts] = LOBSnapshot.from_dataframe(group)
            except ValueError as e:
                logger.warning(f"Skipping timestamp {ts}: {e}")
                continue

        if not self._snapshots:
            raise ValueError("No valid timestamps found in LOB data.")

        self._timestamps = sorted(self._snapshots.keys())
        self._current_timestamp = initial_timestamp or self._timestamps[0]

        # Ensure the initial timestamp is valid
        if self._current_timestamp not in self._snapshots:
            # Find the closest timestamp (within 1 minute tolerance)
            closest = min(self._timestamps, key=lambda t: abs((t - self._current_timestamp).total_seconds()))
            if abs((closest - self._current_timestamp).total_seconds()) < settings.LOB_INTERVAL_SECONDS:
                self._current_timestamp = closest
                logger.debug(f"Using closest timestamp: {closest}")
            else:
                raise ValueError(
                    f"Initial timestamp {initial_timestamp} not found in LOB data. "
                    f"Closest available: {closest}"
                )

        logger.info(f"HistoricalConnector initialized with {len(self._timestamps)} timestamps.")

    # ------------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------------

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Ensure the data has the columns expected by LOBSnapshot."""
        required_cols = EXPECTED_COLUMNS
        missing = set(required_cols) - set(data.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in LOB data: {missing}. "
                f"Expected: {required_cols}"
            )

    # ------------------------------------------------------------------------
    # Timestamp Management
    # ------------------------------------------------------------------------

    def set_timestamp(self, timestamp: datetime) -> None:
        """
        Advance the connector to a specific timestamp.

        Args:
            timestamp: The timestamp to advance to.

        Raises:
            ValueError: If the timestamp is not found (and no close match within 60s).
        """
        if timestamp in self._snapshots:
            self._current_timestamp = timestamp
            return

        # Find the closest timestamp within 1 minute tolerance
        closest = min(self._timestamps, key=lambda t: abs((t - timestamp).total_seconds()))
        if abs((closest - timestamp).total_seconds()) < settings.LOB_INTERVAL_SECONDS:
            self._current_timestamp = closest
            logger.debug(f"Using closest timestamp: {closest} (requested: {timestamp})")
        else:
            raise ValueError(
                f"Timestamp {timestamp} not found in LOB data. "
                f"Closest available: {closest}"
            )

    def get_current_timestamp(self) -> datetime:
        """Return the current timestamp the connector is pointing to."""
        return self._current_timestamp

    def get_current_snapshot(self) -> LOBSnapshot:
        """Return the LOBSnapshot for the current timestamp."""
        return self._snapshots[self._current_timestamp]

    def get_timestamps(self) -> List[datetime]:
        """
        Return the list of all available timestamps in chronological order.

        This is used by the backtester to iterate over the entire day.
        """
        return self._timestamps.copy()  # Return a copy to prevent external modification

    # ------------------------------------------------------------------------
    # DataConnector Interface Implementation
    # ------------------------------------------------------------------------

    def get_available_expiries(self) -> List[str]:
        """Return the expiries available at the current timestamp."""
        return self.get_current_snapshot().get_expiries()

    def get_chain_for_expiry(self, expiry: str, use_cache: bool = True) -> Dict[str, List[OptionQuote]]:
        """
        Return the option chain for a given expiry at the current timestamp.

        Args:
            expiry: The expiry date string (YYYY-MM-DD).
            use_cache: Unused in historical connector (kept for interface compatibility).

        Returns:
            Dict with 'calls' and 'puts' mapped to a lists of OptionQuote objects.
        """
        snapshot = self.get_current_snapshot()

        # Extract data for the specific expiry
        df = snapshot._data  # The filtered DataFrame inside the snapshot
        rows = df[df['expiry'] == expiry] # rows is the chain for expiry

        calls = []
        puts = []

        for _, row in rows.iterrows():
            # Handle NaN values: NaN occurs when fetching values of 0 in yfinance
            bid = 0.0 if pd.isna(row.get("bid", 0.0)) else float(row["bid"])
            ask = 0.0 if pd.isna(row.get("ask", 0)) else float(row["ask"])
            volume = 0 if pd.isna(row.get("volume", 0)) else int(row["volume"])
            open_interest = 0 if pd.isna(row.get("open_interest", 0)) else int(row["open_interest"])
            implied_vol = 0.0 if pd.isna(row.get("implied_vol", 0)) else float(row["implied_vol"])

            # Create quote
            quote = OptionQuote(
                strike=float(row['strike']),
                bid=bid,
                ask=ask,
                mid=(bid + ask) / 2,
                implied_vol=implied_vol,
                volume=volume,
                open_interest=open_interest,
                option_type=str(row['type'])
            )
            if row['type'] == 'call':
                calls.append(quote)
            else:
                puts.append(quote)

        return {"calls": calls, "puts": puts}

    def get_spot_price(self) -> float:
        """Return the spot price at the current timestamp."""
        return self.get_current_snapshot().spot

    def get_surface_data(self, max_expiries: int = 5) -> Dict:
        """
        Not used in backtest. Required by the DataConnector interface.

        Returns an empty dict.
        """
        return {}