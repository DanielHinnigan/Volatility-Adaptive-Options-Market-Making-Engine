"""
Risk Manager for Options Market Making.

This module provides a wrapper around the LucicTseQuotingEngine that enforces
hard caps on portfolio Greeks and PnL drawdown. It acts as a circuit breaker,
cancelling or reducing quotes when risk limits are breached.

The risk manager is designed to be fast (< 1 microsecond for checks) and
completely decoupled from the quoting engine.

Usage:
    risk_manager = RiskManager(quoting_engine)
    quotes = risk_manager.get_quotes(option_specs, positions)
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from ..quoting.lucic_tse import LucicTseQuotingEngine, InventoryGreeks, Quote, TENOR_BUCKETS
from ..data.option_spec import OptionSpec

logger = logging.getLogger(__name__)

# ============================================================================
# Data Containers
# ============================================================================

@dataclass
class RiskStatus:
    halted: bool
    reason: Optional[str] = None
    excess_delta: float = 0.0
    excess_gamma: float = 0.0
    excess_theta: float = 0.0
    excess_vega: Dict[str, float] = field(default_factory=dict)
    drawdown_excess: float = 0.0
    quotes: Dict[str, Quote] = field(default_factory=dict)

@dataclass
class _BreachReport:
    """Internal report of all risk limit breaches. Not part of public API."""
    breached: bool = False
    delta_excess: float = 0.0
    gamma_excess: float = 0.0
    theta_excess: float = 0.0
    vega_excess: Dict[str, float] = field(default_factory=dict)
    drawdown_excess: float = 0.0

# ============================================================================
# Main Class
# ============================================================================

class RiskManager:
    """
    Hard-cap risk manager for options market making.

    Enforces:
        - Net Delta limit (directional exposure)
        - Net Gamma limit (convexity risk, only negative side)
        - Net Theta limit (time decay cost, only negative side)
        - Vega limits by tenor (volatility exposure per bucket)
        - Daily PnL drawdown limit (stops trading if loss exceeds threshold)

    If any limit is breached, the risk manager returns empty quotes (halts trading)
    or reduces quote sizes if risk is elevated.
    """

    def __init__(
        self,
        quoting_engine: LucicTseQuotingEngine,
        delta_limit: float = 50000.0,
        gamma_limit: float = -100.0,
        theta_limit: float = -100.0,
        vega_limits: Optional[Dict[str, float]] = None,
        drawdown_limit: float = -0.02,
        reduce_size_threshold: float = 0.8,
        initial_capital: float = 100000.0,
    ):
        """
        Initialize the risk manager.

        Args:
            quoting_engine: The LucicTseQuotingEngine instance.
            delta_limit: Maximum absolute net Delta (default 50,000 shares equivalent).
                        Units are shares per Dollar: How many shares equivalent are the portfolio long or short due to the change in the value of the portfolio
            gamma_limit: Minimum net Gamma (must be > this value, default -100).
                        Units are delta change per change in underlying price.
            theta_limit: Minimum net Theta (must be > this value, default -100).
                         This implicitly caps positive Gamma.
                        Caps how much money can be lost per day due to time decay.
                        For example, a theta of -100 corresponds to losing 100 USD per day.
            vega_limits: Dict mapping tenor to max Vega.
                         Defaults to standard tenors.
                         Units are Dollars per 1% IV move.
            drawdown_limit: Maximum daily drawdown as a fraction of capital (default -0.02).
            reduce_size_threshold: Fraction of limit at which to halve quote sizes (default 0.8).
            initial_capital: Starting capital for PnL tracking (default 100,000).
        """
        # Public Variables
        self.quoting_engine = quoting_engine
        self.initial_capital = initial_capital
        self.reduce_size_threshold  = reduce_size_threshold
        self.drawdown_limit = drawdown_limit

        # Hard limits
        self.delta_limit = delta_limit
        self.gamma_limit = gamma_limit
        self.theta_limit = theta_limit

        # Validate and store vega limits
        if vega_limits is None:
            self.vega_limits = {
                "0-7D": 500.0,
                "8-30D": 1500.0,
                "30-60D": 2000.0,
                "60-90D": 2000.0,
                "90D+": 2000.0,
            }
        else:
            invalid_keys = set(vega_limits.keys()) - set(TENOR_BUCKETS.keys())
            if invalid_keys:
                raise ValueError(
                    f"Invalid tenor keys in vega_limits: {invalid_keys}. "
                    f"Valid tenors are: {list(TENOR_BUCKETS.keys())}"
                )
            self.vega_limits = vega_limits

        # More hard limits
        self._current_capital = initial_capital
        self._day_start_capital = initial_capital
        self._daily_pnl = 0.0
        self._halted = False

        # Halt flag
        self._halted = False

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def get_quotes(
        self,
        option_specs: List[OptionSpec],
        positions: Dict[str, int],
        update_pnl: Optional[float] = None,
    ) -> RiskStatus:
        # 1. Update PnL
        if update_pnl is not None:
            self._update_pnl(update_pnl)

        # 2. Aggregated Greeks
        inventory = self.quoting_engine.aggregate_inventory(positions, option_specs)

        # 3. Check all limits (Greek + Drawdown)
        breach_report = self._check_breach(inventory)

        if breach_report.breached:
            reasons = []
            if breach_report.delta_excess != 0:
                reasons.append(f"Delta (excess {breach_report.delta_excess:.2f})")
            if breach_report.gamma_excess != 0:
                reasons.append(f"Gamma (excess {breach_report.gamma_excess:.2f})")
            if breach_report.theta_excess != 0:
                reasons.append(f"Theta (excess {breach_report.theta_excess:.2f})")
            if breach_report.vega_excess:
                reasons.append(f"Vega {breach_report.vega_excess}")
            if breach_report.drawdown_excess > 0:
                reasons.append(f"Drawdown (excess {breach_report.drawdown_excess:.2f})")

            return RiskStatus(
                halted=True,
                reason="; ".join(reasons),
                excess_delta=breach_report.delta_excess,
                excess_gamma=breach_report.gamma_excess,
                excess_theta=breach_report.theta_excess,
                excess_vega=breach_report.vega_excess,
                drawdown_excess=breach_report.drawdown_excess,
            )

        # 4. Get quotes from the quoting engine if no limits are breached
        quotes = self.quoting_engine.generate_quotes(option_specs, positions)

        # 5. Reduce sizes if risk is elevated
        if self._is_risk_elevated(inventory):
            quotes = self._reduce_quote_sizes(quotes)

        return RiskStatus(halted=False, quotes=quotes)

    def update_pnl(self, pnl_delta: float) -> None:
        """
        Update daily PnL (e.g., after a trade or at the end of a tick).

        Args:
            pnl_delta: The change in PnL (positive for profit, negative for loss).
        """
        self._update_pnl(pnl_delta)

    def reset_daily_pnl(self) -> None:
        """Reset daily PnL tracking (call at the start of each trading day)."""
        # 1. Carry forward the running capital
        self._day_start_capital = self._current_capital
        
        # 2. Reset daily counters
        self._daily_pnl = 0.0
        self._halted = False
        logger.info(f"Daily PnL reset. Day-start capital: {self._day_start_capital:.2f}")

    def is_halted(self) -> bool:
        """Returns True if the risk manager has halted trading."""
        return self._halted

    # ------------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------------

    def _update_pnl(self, pnl_delta: float) -> None:
        # 1. Update running total
        self._current_capital += pnl_delta

        # 2. Update daily PnL (relative to start of day)
        self._daily_pnl += pnl_delta
            
    def _is_risk_elevated(self, inventory: InventoryGreeks) -> bool:
        """
        Returns True if risk is elevated (near limits) and quote sizes should be reduced.
        """
        # Delta
        if abs(inventory.delta) > self.reduce_size_threshold * self.delta_limit:
            return True
        # Gamma (negative side)
        if inventory.gamma < self.reduce_size_threshold * self.gamma_limit:
            return True
        # Theta (negative side)
        if inventory.theta < self.reduce_size_threshold * self.theta_limit:
            return True
        # Vega (by tenor)
        for tenor, vega in inventory.vega_by_tenor.items():
            limit = self.vega_limits.get(tenor, 2000.0)
            if abs(vega) > self.reduce_size_threshold * limit:
                return True
        return False

    def _reduce_quote_sizes(self, quotes: Dict[str, Quote]) -> Dict[str, Quote]:
        """
        Halve bid and ask sizes for all quotes.
        Rounds down the size if division (size / 2) is not an integer.
        """
        reduced = {}
        for opt_id, quote in quotes.items():
            reduced[opt_id] = Quote(
                option_id=quote.option_id,
                strike=quote.strike,
                expiry=quote.expiry,
                T=quote.T,
                option_type=quote.option_type,
                bid=quote.bid,
                ask=quote.ask,
                bid_size=max(1, quote.bid_size // 2),
                ask_size=max(1, quote.ask_size // 2),
                fair_value=quote.fair_value,
                spread=quote.spread,
            )
        return reduced

    def _check_breach(self, inventory: InventoryGreeks) -> _BreachReport:
        """
        Checks all risk limits and returns a report of all breaches.
        This includes Delta, Gamma, Theta, Vega, AND Drawdown.
        """
        report = _BreachReport(breached=False)

        # 1. Delta
        if abs(inventory.delta) > self.delta_limit:
            report.breached = True
            excess = abs(inventory.delta) - self.delta_limit
            report.delta_excess = excess * (1 if inventory.delta > 0 else -1)

        # 2. Gamma (Min)
        if inventory.gamma < self.gamma_limit:
            report.breached = True
            report.gamma_excess = self.gamma_limit - inventory.gamma

        # 3. Theta (Min)
        if inventory.theta < self.theta_limit:
            report.breached = True
            report.theta_excess = self.theta_limit - inventory.theta

        # 4. Vega (by tenor)
        for tenor, vega in inventory.vega_by_tenor.items():
            limit = self.vega_limits.get(tenor, 2000.0)
            if abs(vega) > limit:
                report.breached = True
                report.vega_excess[tenor] = abs(vega) - limit

        # 5. Drawdown
        drawdown_limit_abs = self.drawdown_limit * self._day_start_capital
        if self._daily_pnl < drawdown_limit_abs:
            report.breached = True
            report.drawdown_excess = drawdown_limit_abs - self._daily_pnl
            self._halted = True  # Set the internal halt flag

        return report

    # ------------------------------------------------------------------------
    # Limit Management (Runtime adjustments)
    # ------------------------------------------------------------------------

    def set_delta_limit(self, new_limit: float) -> None:
        self.delta_limit = new_limit

    def set_gamma_limit(self, new_limit: float) -> None:
        self.gamma_limit = new_limit

    def set_theta_limit(self, new_limit: float) -> None:
        self.theta_limit = new_limit

    def set_vega_limit(self, tenor: str, new_limit: float) -> None:
        if tenor not in TENOR_BUCKETS:
            raise ValueError(
                f"Invalid tenor: {tenor}. Valid tenors are: {list(TENOR_BUCKETS.keys())}"
            )
        self.vega_limits[tenor] = new_limit

    def set_drawdown_limit(self, new_limit: float) -> None:
        self.drawdown_limit = new_limit