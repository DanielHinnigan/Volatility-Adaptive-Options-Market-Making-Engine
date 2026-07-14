"""
Event-driven backtester for options market-making.

Replays a single day of historical LOB data (bid/ask snapshots) and simulates the bot's
quoting, fill, inventory, and PnL dynamics.

The backtester uses the exact same components as the live bot:
    - PricingEngine (with SABR re-pricing)
    - LucicTseQuotingEngine
    - RiskManager

Fills are simulated using the actual market bid/ask and queue depth from the LOB data.

This backtester is designed for a single trading day.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from ..pricing_engine import PricingEngine
from ..quoting.lucic_tse import LucicTseQuotingEngine, Quote
from ..risk.risk_manager import RiskManager, RiskStatus
from ..utils.time_utils import compute_time_to_expiry
from ..data.option_spec import OptionSpec
from ..data.lob_snapshot import LOBSnapshot, LOBQuote
from ..data.historical_connector import HistoricalConnector

logger = logging.getLogger(__name__)


@dataclass
class Fill:
    """A simulated fill."""
    timestamp: datetime
    option_id: str
    strike: float
    expiry: str
    option_type: str
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: int
    pnl: float


@dataclass
class InventorySnapshot:
    """Snapshot of inventory and PnL at a point in time."""
    timestamp: datetime
    positions: Dict[str, int]
    cash: float
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    delta: float
    gamma: float
    vega: float
    theta: float


class Backtester:
    """
    Event-driven backtester for options market-making.

    Replays a single day of historical LOB data and simulates the full quoting
    and risk management loop.
    """

    def __init__(
        self,
        pricing_engine: PricingEngine,
        quoting_engine: LucicTseQuotingEngine,
        risk_manager: RiskManager,
        option_specs: List[OptionSpec],
        data_path: Path,
        initial_capital: float = 100000.0,
        transaction_cost_per_contract: float = 0.50,
    ):
        """
        Initialize the backtester.

        Args:
            pricing_engine: The PricingEngine instance (must be calibrated).
            quoting_engine: The LucicTseQuotingEngine instance (must be initialized).
            risk_manager: The RiskManager instance.
            option_specs: List of OptionSpec objects to quote.
            data_path: Path to the Parquet file containing LOB data for a SINGLE day.
            initial_capital: Starting capital for PnL tracking.
            transaction_cost_per_contract: Commission per contract per trade.
        """
        self._validate_data_path(data_path)

        self.pricing_engine = pricing_engine
        self.quoting_engine = quoting_engine
        self.risk_manager = risk_manager
        self.option_specs = option_specs
        self.initial_capital = initial_capital
        self.transaction_cost_per_contract = transaction_cost_per_contract

        # Load data
        self.data = self._load_data(data_path)
        if self.data.empty:
            raise ValueError("No LOB data loaded from file.")

        self._validate_option_specs()

        # Internal state
        self._positions: Dict[str, int] = {}
        self._cash = 0.0
        self._fill_history: List[Fill] = []
        self._inventory_history: List[InventorySnapshot] = []

        # PnL Tracking
        self._daily_pnl = 0.0
        self._previous_daily_pnl = 0.0
        self._update_pnl = 0.0

        # Risk tracking
        self._risk_breaches = 0
        self._drawdown_breaches = 0

        # Mapping from option ID to option specifications
        self._spec_map = {spec.id: spec for spec in option_specs}

        logger.info(f"Backtester initialized with {len(self.data['timestamp'].unique())} timestamps.")

    # ------------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------------

    def _validate_data_path(self, data_path: Path) -> None:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if data_path.is_dir():
            raise ValueError(f"data_path must be a single Parquet file, not a directory.")
        if data_path.suffix != '.parquet':
            raise ValueError(f"data_path must be a .parquet file. Found: {data_path.suffix}")

    def _load_data(self, data_path: Path) -> pd.DataFrame:
        df = pd.read_parquet(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        required_cols = LOBSnapshot.EXPECTED_COLUMNS
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in LOB data: {missing}")
        return df.sort_values('timestamp').reset_index(drop=True)

    def _validate_option_specs(self) -> None:
        """Check that the option specs exist in the LOB data (at least in the first snapshot)."""
        # Quick check using the first timestamp
        first_ts = self.data['timestamp'].iloc[0]
        first_slice = self.data[self.data['timestamp'] == first_ts]
        available = set(zip(first_slice['expiry'], first_slice['strike'], first_slice['type']))

        for spec in self.option_specs:
            key = (spec.expiry, spec.strike, spec.option_type)
            if key not in available:
                logger.warning(
                    f"Option {spec.id} not found in the first timestamp of the LOB data. It is not guareented to be included in the backtest."
                )

    # ------------------------------------------------------------------------
    # Fill Simulation
    # ------------------------------------------------------------------------

    def _simulate_fill(self, quote: Quote, lob_quote: LOBQuote) -> Tuple[bool, bool]:
        """
        Simulate whether a bid and/or ask order gets filled based on market LOB.

        Returns:
            (bid_filled, ask_filled)
        """
        bid_filled = False
        ask_filled = False

        market_bid = lob_quote.bid
        market_ask = lob_quote.ask
        market_bid_size = lob_quote.bid_size
        market_ask_size = lob_quote.ask_size

        # --------------------------------------------------------------------
        # 1. Bid Fill Simulation (We are buying)
        # --------------------------------------------------------------------
        if quote.bid > 0:
            # Case A: No other bids in the market – we are the sole liquidity provider.
            if market_bid == 0:
                # We are the only bid. If any sell order arrives, we fill.
                # Use a high probability (front of queue).
                bid_filled = np.random.rand() < 0.9

            # Case B: Other bids exist.
            else:
                if quote.bid > market_bid:
                    # We improved the best bid → front of queue
                    bid_filled = np.random.rand() < 0.9
                elif quote.bid == market_bid:
                    # We match the best bid → join queue behind existing volume
                    if market_bid_size > 0:
                        prob = quote.bid_size / (market_bid_size + quote.bid_size)
                        bid_filled = np.random.rand() < prob
                    else:
                        # Market has a price but zero depth → treat as front of queue
                        bid_filled = np.random.rand() < 0.9
                else:
                    # Our bid is worse than market → no fill
                    bid_filled = False

        # --------------------------------------------------------------------
        # 2. Ask Fill Simulation (We are selling)
        # --------------------------------------------------------------------
        if quote.ask > 0:
            # Case A: No other asks in the market – we are the sole liquidity provider.
            if market_ask == 0:
                # We are the only ask. If any buy order arrives, we fill.
                ask_filled = np.random.rand() < 0.9

            # Case B: Other asks exist.
            else:
                if quote.ask < market_ask:
                    # We improved the best ask → front of queue
                    ask_filled = np.random.rand() < 0.9
                elif quote.ask == market_ask:
                    # We match the best ask → join queue
                    if market_ask_size > 0:
                        prob = quote.ask_size / (market_ask_size + quote.ask_size)
                        ask_filled = np.random.rand() < prob
                    else:
                        ask_filled = np.random.rand() < 0.9
                else:
                    # Our ask is worse than market → no fill
                    ask_filled = False

        return bid_filled, ask_filled

    # ------------------------------------------------------------------------
    # PnL & Inventory Updates
    # ------------------------------------------------------------------------

    def _update_inventory_and_pnl(self, fill: Fill) -> None:
        """
        Update positions, cash, and realized PnL for a single fill.
        
        BUY:  Cash decreases, inventory increases (+)
        SELL: Cash increases, inventory decreases (-)
        """
        # 1. Update Inventory (Position)
        if fill.side == 'BUY':
            self._positions[fill.option_id] = self._positions.get(fill.option_id, 0) + fill.quantity
        else:  # SELL
            self._positions[fill.option_id] = self._positions.get(fill.option_id, 0) - fill.quantity

        # 2. Update Cash
        if fill.side == 'BUY':
            cash_delta = -fill.price * fill.quantity
        else:  # SELL
            cash_delta = fill.price * fill.quantity

        # Transaction costs are always negative
        cash_delta -= self.transaction_cost_per_contract * abs(fill.quantity)
        self._cash += cash_delta

        # 3. Update Realized PnL
        self._daily_pnl += cash_delta

    def _mark_to_market(self, fair_values: Dict[str, float]) -> float:
        """Compute total PnL (realized + unrealized) at current timestamps."""
        unrealized = 0.0
        for opt_id, quantity in self._positions.items():
            if quantity == 0:
                continue
            fair = fair_values.get(opt_id, 0.0)
            unrealized += quantity * fair
        return self._cash + unrealized - self.initial_capital

    # ------------------------------------------------------------------------
    # Risk Unwind Logic
    # ------------------------------------------------------------------------

    def _unwind_excess_risk(self, risk_status: RiskStatus, timestamp: datetime, snapshot: LOBSnapshot) -> None:
        """
        Unwind excess risk when hard limits are breached.

        Delta: Unwound by buying/selling the underlying ETF (SPY).
        Gamma/Vega/Drawdown: Triggers a full fire sale of all option positions.
        """
        self._risk_breaches += 1

        # 1. Handle Delta Excess (Simple: Buy/Sell Underlying)
        if risk_status.excess_delta != 0:
            spot = self.pricing_engine.get_spot()
            shares_to_trade = -risk_status.excess_delta  # Negative = sell, Positive = buy

            cash_delta = shares_to_trade * spot
            cash_delta -= self.transaction_cost_per_contract * abs(shares_to_trade)  # Underlying commission

            self._cash += cash_delta

            logger.warning(
                f"Unwound Delta excess: {risk_status.excess_delta:.2f} shares equivalent. "
                f"Traded {shares_to_trade:.0f} shares of SPY at ${spot:.2f}. "
                f"Cash delta: ${cash_delta:.2f}"
            )

        # 2. Handle Gamma, Vega, and Drawdown (Flatten Everything)
        if (risk_status.excess_gamma != 0 or
            risk_status.excess_vega or
            risk_status.drawdown_excess > 0):

            if risk_status.drawdown_excess > 0:
                self._drawdown_breaches += 1

            reason_parts = []
            if risk_status.excess_gamma != 0:
                reason_parts.append(f"Gamma (excess {risk_status.excess_gamma:.2f})")
            if risk_status.excess_vega:
                reason_parts.append(f"Vega ({list(risk_status.excess_vega.keys())})")
            if risk_status.drawdown_excess > 0:
                reason_parts.append(f"Drawdown (excess {risk_status.drawdown_excess:.2f})")

            logger.warning(
                f"Severe risk breach ({', '.join(reason_parts)}). Flattening ALL option positions."
            )
            self._flatten_all_positions(timestamp, snapshot)

    def _flatten_all_positions(self, timestamp: datetime, snapshot: LOBSnapshot) -> None:
        """
        Emergency flatten: Close all option positions at realistic liquidation prices (penalized due to fire sale).

        Use:
            - LOB Bid/Ask (if available)
            - Heavy haircut/markup if zero recorded bid/ask
            - Market impact based on position size / Open Interest
        """
        if not self._positions:
            return

        logger.warning(f"FIRE SALE: Flattening all positions at {timestamp}")
        total_exit_value = 0.0
        closed_positions = []

        for opt_id, quantity in list(self._positions.items()):
            if quantity == 0:
                continue

            spec = self._spec_map.get(opt_id)
            if not spec:
                continue

            # 1. Fair Value
            fair = self.pricing_engine.get_price(
                spec.strike, spec.expiry, spec.option_type, use_sabr=True
            )

            # 2. LOB Quote
            lob_quote = snapshot.get_quote(spec.expiry, spec.strike, spec.option_type)

            # 3. Base Liquidation Price
            if quantity > 0:
                # LONG → Sell
                if lob_quote and lob_quote.bid > 0:
                    base_price = lob_quote.bid
                else:
                    base_price = fair * 0.70  # Heavy haircut
                    logger.debug(f"No bid for {opt_id}. Using 70% of fair ({base_price:.2f})")
            else:
                # SHORT → Buy to cover
                if lob_quote and lob_quote.ask > 0:
                    base_price = lob_quote.ask
                else:
                    base_price = fair * 1.30  # Heavy markup
                    logger.debug(f"No ask for {opt_id}. Using 130% of fair ({base_price:.2f})")

            # 4. Market Impact (Slippage) based on Open Interest
            if lob_quote and lob_quote.open_interest > 0:
                oi = lob_quote.open_interest
                size_ratio = min(abs(quantity) / max(oi, 1), 1.0) # The more we have to unload compared to open interest, the higher the slippage factor.
                slippage_factor = min(0.20, 0.02 * (size_ratio ** 0.5))

                if quantity > 0:
                    execution_price = base_price * (1 - slippage_factor)
                else:
                    execution_price = base_price * (1 + slippage_factor)
            else:
                # No OI info → flat 30% slippage
                if quantity > 0:
                    # LONG -> Sell
                    execution_price = base_price * 0.7
                else:
                    # SHORT -> Buy back
                    execution_price = base_price * 1.3

            # 5. Execute
            cash_delta = quantity * execution_price
            cash_delta -= self.transaction_cost_per_contract * abs(quantity)

            self._cash += cash_delta
            total_exit_value += cash_delta

            closed_positions.append(f"{opt_id} ({quantity} @ {execution_price:.2f})")
            self._positions[opt_id] = 0

        logger.info(
            f"Fire Sale Complete: Closed {len(closed_positions)} positions. "
            f"Net cash change: ${total_exit_value:.2f}."
        )

    # ------------------------------------------------------------------------
    # Main Run Loop
    # ------------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Run the backtest over the entire LOB data.

        Returns:
            Dict containing performance metrics.
        """
        logger.info("Starting backtest...")

        # Reset state
        self._positions = {}
        self._cash = 0.0
        self._daily_pnl = 0.0
        self._previous_daily_pnl = 0.0
        self._update_pnl = 0.0
        self._fill_history = []
        self._inventory_history = []
        self._risk_breaches = 0
        self._drawdown_breaches = 0

        # 1. Create historical connector and inject into PricingEngine
        historical_connector = HistoricalConnector(self.data)
        self.pricing_engine._connector = historical_connector

        # 2. Iterate over timestamps
        for timestamp in historical_connector.get_timestamps():
            # 2.1 Advance the connector to this timestamp
            historical_connector.set_timestamp(timestamp)
            snapshot = historical_connector.get_current_snapshot()

            spot = snapshot.spot

            # 2.2 Build valid option specs for this timestamp (recompute T)
            valid_specs = []
            for spec in self.option_specs:
                lob_quote = snapshot.get_quote(spec.expiry, spec.strike, spec.option_type)
                if lob_quote is not None:
                    new_T = compute_time_to_expiry(spec.expiry, current_time=timestamp)
                    valid_specs.append(OptionSpec(
                        strike=spec.strike,
                        expiry=spec.expiry,
                        T=new_T,
                        option_type=spec.option_type,
                        id=spec.id,
                    ))

            if not valid_specs:
                continue

            # 2.3 Get current positions: How long/short is the bot w.r.t each of the specified options
            positions = {opt_id: self._positions.get(opt_id, 0) for opt_id in self._spec_map.keys()}

            # 2.4 Generate quotes via RiskManager
            risk_status = self.risk_manager.get_quotes(
                option_specs=valid_specs,
                positions=positions,
                update_pnl=self._update_pnl,
            )

            if risk_status.halted:
                self._unwind_excess_risk(risk_status, timestamp, snapshot)
                continue

            # 2.5 Process each quote and simulate fills
            for opt_id, quote in risk_status.quotes.items():
                spec = self._spec_map.get(opt_id)
                if not spec:
                    continue

                # Quote from historical data 
                lob_quote = snapshot.get_quote(spec.expiry, spec.strike, spec.option_type)
                if lob_quote is None:
                    continue

                bid_filled, ask_filled = self._simulate_fill(quote, lob_quote)

                if bid_filled:
                    fill = Fill(
                        timestamp=timestamp,
                        option_id=opt_id,
                        strike=spec.strike,
                        expiry=spec.expiry,
                        option_type=spec.option_type,
                        side='BUY',
                        price=quote.bid,
                        quantity=quote.bid_size,
                        pnl=0.0,
                    )
                    self._update_inventory_and_pnl(fill)
                    self._fill_history.append(fill)

                if ask_filled:
                    fill = Fill(
                        timestamp=timestamp,
                        option_id=opt_id,
                        strike=spec.strike,
                        expiry=spec.expiry,
                        option_type=spec.option_type,
                        side='SELL',
                        price=quote.ask,
                        quantity=quote.ask_size,
                        pnl=0.0,
                    )
                    self._update_inventory_and_pnl(fill)
                    self._fill_history.append(fill)

            # 2.6 Mark-to-Market: Values the inventory at its fair value calculated based on our option models.
            # Escpecially important for risk management.
            fair_values = {}
            for opt_id, spec in self._spec_map.items():
                if self._positions.get(opt_id, 0) != 0:
                    try:
                        # The PricingEngine now uses the historical connector internally
                        fair_values[opt_id] = self.pricing_engine.get_price(
                            spec.strike, spec.expiry, spec.option_type, use_sabr=True
                        )
                    except Exception as e:
                        logger.debug(f"Could not price {opt_id}: {e}")

            total_pnl = self._mark_to_market(fair_values)
            realized_pnl = self._daily_pnl
            unrealized_pnl = total_pnl - realized_pnl

            # Update PnL of risk manager
            self._update_pnl = self._daily_pnl - self.previous_daily_pnl
            self._previous_daily_pnl = self._daily_pnl

            # 2.7 Get Greeks for the current inventory
            inventory_greeks = self.quoting_engine.aggregate_inventory(positions, valid_specs)

            # 2.8 Record snapshot
            self._inventory_history.append(InventorySnapshot(
                timestamp=timestamp,
                positions=dict(self._positions),
                cash=self._cash,
                total_pnl=total_pnl,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                delta=inventory_greeks.delta,
                gamma=inventory_greeks.gamma,
                vega=inventory_greeks.vega,
                theta=inventory_greeks.theta,
            ))

        # 3. Compute metrics (after iterating over timestampts)
        results = self._compute_metrics()
        logger.info("Backtest complete.")
        return results

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------

    def _compute_metrics(self) -> Dict[str, Any]:
        if not self._inventory_history:
            return {
                'total_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'risk_breaches': 0,
                'drawdown_breaches': 0,
                'equity_curve': pd.DataFrame(),
            }

        equity_df = pd.DataFrame([{
            'timestamp': s.timestamp,
            'total_pnl': s.total_pnl,
            'realized_pnl': s.realized_pnl,
            'unrealized_pnl': s.unrealized_pnl,
        } for s in self._inventory_history])

        total_pnl = equity_df['total_pnl'].iloc[-1]

        returns = equity_df['total_pnl'].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 390) if len(returns) > 1 else 0.0

        peak = equity_df['total_pnl'].cummax()
        max_drawdown = ((equity_df['total_pnl'] - peak) / (self.initial_capital + peak)).min()

        total_trades = len(self._fill_history)
        win_rate = 0.0
        if total_trades > 0:
            daily_pnl = equity_df.groupby(equity_df['timestamp'].dt.date)['total_pnl'].last().diff().fillna(0)
            win_days = (daily_pnl > 0).sum()
            win_rate = win_days / len(daily_pnl) if len(daily_pnl) > 0 else 0.0

        return {
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'risk_breaches': self._risk_breaches,
            'drawdown_breaches': self._drawdown_breaches,
            'equity_curve': equity_df,
        }