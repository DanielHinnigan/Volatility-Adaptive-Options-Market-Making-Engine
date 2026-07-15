"""
Lucic-Tse (2024) Portfolio Market-Making Model for Options.

This module implements the portfolio-level market-making model from:
    Lucic, V. & Tse, A. (2024). "Optimal Option Market Making and Volatility Arbitrage."

The model provides closed-form optimal bid/ask spreads for a portfolio of options,
incorporating:
    - Portfolio-level risk penalties via risk factors (vega buckets, delta, gamma).
    - Volatility arbitrage edge (realised vs implied vol).
    - Order flow dynamics (elasticity and baseline arrival rates).
    - Cross-option inventory correlations.

The implementation uses the closed-form solution (Equations 27-30) for the case
B = 0 (no running penalty), with the "frozen gamma" approximation (constant C0)
and constant subjective volatility over the short horizon T.

Threading:
    The engine runs a background daemon thread that automatically updates the
    internal state (Theta_2, theta_1) at a configurable interval.

Usage:
    engine = LucicTseQuotingEngine(
        pricing_engine=pricing_engine,
        risk_factors=risk_factors,
        order_flow_params=order_flow_params,
        auto_update=True,
        update_interval_ms=200,
        horizon_hours=0.5,
        initial_realized_vol=0.18,
        option_specs=my_option_specs,
    )

    # On every tick:
    quotes = engine.generate_quotes(option_specs, positions, r, q)

    # Update vol forecast when new estimate arrives:
    engine.set_vol_forecast(new_vol)

    # Shutdown:
    engine.stop_background_update()
"""

import logging
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from scipy.linalg import inv
from scipy.special import erf

from ..pricing_engine import PricingEngine
from ..data.option_spec import OptionSpec

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

TENOR_BUCKETS = {
    "0-7D": 7,
    "8-30D": 30,
    "30-60D": 60,
    "60-90D": 90,
    "90D+": float('inf'),  # Catch-all for anything > 90 days
}

# ============================================================================
# Data Containers
# ============================================================================

@dataclass
class InventoryGreeks:
    """
    Aggregated Greeks of the current options inventory.

    Attributes:
        delta: Net Delta exposure (dP/dS).
        gamma: Net Gamma exposure (d²P/dS²).
        vega: Net Vega exposure (dP/dσ) for a 1% IV move.
        theta: Net Theta exposure (dP/dt) - daily time decay.
        vega_by_tenor: Vega bucketed by tenor (0-7D, 8-30D, etc.).
        gamma_by_expiry: Gamma bucketed by expiry (for monitoring).
        delta_by_expiry: Delta bucketed by expiry (for monitoring).
    """
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    vega_by_tenor: Dict[str, float] = field(default_factory=dict)
    delta_by_expiry: Dict[str, float] = field(default_factory=dict)
    gamma_by_expiry: Dict[str, float] = field(default_factory=dict)


@dataclass
class Quote:
    """A single bid/ask quote for an option."""
    option_id: str
    strike: float
    expiry: str
    T: float
    option_type: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    fair_value: float   # Calculated by option models
    spread: float       # Total spread (ask - bid)


# ============================================================================
# Main Quoting Engine
# ============================================================================

class LucicTseQuotingEngine:
    """
    Portfolio-level market-making engine based on Lucic & Tse (2024).

    This engine computes optimal bid/ask spreads for a portfolio of options
    on the same underlying. It uses the closed-form solution (B=0) with
    frozen gamma and constant subjective volatility over the horizon.

    The implementation follows the paper's assumption of constant subjective
    volatility σ over the short horizon T (e.g., 15-30 minutes). The background
    thread updates the state periodically with the latest realized vol estimate,
    allowing the engine to adapt to a changing volatility view between updates.

    Key components:
        1. Risk penalty matrix A (from risk factors).
        2. Order flow diagonal matrix D.
        3. Volatility edge vector C0 (frozen at current sigma).
        4. Precomputed matrices Theta_2 and theta_1 via Equations 27-28.
        5. Live spread evaluation via Equations 29-30.

    Threading:
        The engine runs a background daemon thread that automatically updates
        the internal state. The user can also call update_state() manually for
        synchronous use (e.g., in unit tests).

    Usage:
        engine = LucicTseQuotingEngine(
            pricing_engine=pricing_engine,
            risk_factors=risk_factors,
            order_flow_params=order_flow_params,
            auto_update=True,
            update_interval_ms=200,
            initial_realized_vol=0.18,
            option_specs=my_option_specs,
        )

        # On every tick:
        quotes = engine.generate_quotes(option_specs, positions, r, q)

        # Update vol forecast:
        engine.set_vol_forecast(new_vol)

        # Shutdown:
        engine.stop_background_update()
    """

    def __init__(
        self,
        pricing_engine: PricingEngine,
        risk_factors: List[Dict],
        order_flow_params: Dict,
        horizon_hours: float = 0.5,
        risk_aversion: float = 0.1,
        default_bid_size: int = 1,
        default_ask_size: int = 1,
        auto_update: bool = True,
        update_interval_ms: int = 200,
        initial_realized_vol: Optional[float] = None,
        option_specs: Optional[List[OptionSpec]] = None,
    ):
        """
        Initialize the quoting engine.

        Args:
            pricing_engine: The PricingEngine facade for fair values and Greeks.
            risk_factors: List of risk factor definitions, each with:
                - 'name': str
                - 'alpha': float (penalty strength)
                - 'membership': function(option_spec) -> bool
                - 'weight_key': str (e.g., 'vega', 'delta', 'gamma')
            order_flow_params: Dict with:
                - 'lambda0_a': float or array (baseline ask order rate)
                - 'lambda0_b': float or array (baseline bid order rate)
                - 'kappa_a': float or array (ask elasticity)
                - 'kappa_b': float or array (bid elasticity)
            horizon_hours: Planning horizon in hours (T in the paper).
            risk_aversion: Overall risk aversion scaling (multiplies A).
            default_bid_size, default_ask_size: Default quote sizes.
            auto_update: If True, starts a background thread for state updates.
            update_interval_ms: Interval between background state updates.
            initial_realized_vol: Initial realised volatility forecast (default 0.20).
            option_specs: Initial list of option specs. If not provided, will be
                          set on the first call to generate_quotes().
        """
        self.pricing_engine = pricing_engine
        self.risk_factors = risk_factors
        self.order_flow_params = order_flow_params
        self.horizon_hours = horizon_hours
        self.risk_aversion = risk_aversion
        self.default_bid_size = default_bid_size
        self.default_ask_size = default_ask_size
        self.update_interval_ms = update_interval_ms

        # Internal state (protected by a lock)
        self._lock = threading.RLock()
        self._is_initialized = False
        self._N = 0
        self._option_specs = []
        self._A = None
        self._D = None
        self._Theta_2 = None
        self._theta_1 = None
        self._C0 = None
        self._kappa_a = None
        self._kappa_b = None
        self._lambda0_a = None
        self._lambda0_b = None
        self._diag_theta2 = None

        # Latest parameters used in the last state update
        self._last_realized_vol = None
        self._option_specs = []

        # Background thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Initialise the volatility forecast
        if initial_realized_vol is not None:
            self._last_realized_vol = initial_realized_vol
            logger.info(f"Initial realized vol set to {initial_realized_vol:.4f}")
        else:
            self._last_realized_vol = 0.20
            logger.info(
                f"No initial realized vol provided. Using default {self._last_realized_vol}. "
                "You can update it later with set_vol_forecast()."
            )

        # Initialise the option specs (if provided)
        if option_specs is not None:
            if not option_specs:
                logger.warning("option_specs is an empty list. No options will be quoted.")
            self._option_specs = option_specs
            self._N = len(option_specs)
            logger.info(f"Initialised with {self._N} option specs.")
        else:
            self._option_specs = []
            self._N = 0
            logger.info("No option specs provided. They will be set on the first call to generate_quotes().")

        # Start background thread if requested
        if auto_update:
            self.start_background_update(update_interval_ms)

    # ------------------------------------------------------------------------
    # 1. Background Thread
    # ------------------------------------------------------------------------

    def start_background_update(self, interval_ms: int = None) -> None:
        """
        Starts a daemon thread that updates the state periodically.

        Args:
            interval_ms: Update interval in milliseconds. If not provided,
                         uses the value from __init__.
        """
        # 1. If the flag says running, check if the thread is actually alive
        if self._running:
            if self._thread is not None and self._thread.is_alive():
                logger.debug("Background update already running.")
                return
            else:
                # Thread is dead but flag is stale. Reset.
                logger.debug("Stale running flag detected. Resetting.")
                self._running = False
                self._thread = None

        # 2. If the thread is still alive (orphaned), do NOT start a new one
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Background thread is still alive. Cannot start a new one.")
            return

        if interval_ms is not None:
            self.update_interval_ms = interval_ms

        logger.info(f"Performing initial synchronous state update...")
        # 1. Fetch initial inputs
        spot = self.pricing_engine.get_spot()
        if spot is None or spot <= 0:
            logger.warning("Cannot perform initial state update: invalid spot.")
        else:
            # 2. Perform a blocking initial update
            # We need option_specs. If not set, we can't do it yet.
            if self._option_specs:
                try:
                    with self._lock:
                        self._update_state(
                            spot=spot,
                            realized_vol_estimate=self._last_realized_vol,
                            option_specs=self._option_specs,
                        )
                    logger.info("Initial state update completed.")
                except Exception as e:
                    logger.error(f"Initial state update failed: {e}", exc_info=True)
            else:
                logger.info("No option specs provided. Initial state update skipped (will be done on first generate_quotes).")

        # 3. Start daemon thread for periodic updates
        logger.info(f"Starting background update thread (interval={self.update_interval_ms}ms).")
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def stop_background_update(self) -> None:
        """Stops the background update thread safely."""
        if not self._running:
            logger.info("No background calibration thread running.")
            return

        logger.info("Stopping background update thread...")
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                logger.warning(
                    "Background thread did not stop within 1s. "
                    "It will stop on its next cycle or when the program exits. "
                    "The thread reference is kept to prevent a new thread from starting."
                )
                # DO NOT set self._thread = None here
            else:
                self._thread = None
        else:
            self._thread = None

        logger.info("Background update thread stopped (or marked for stopping).")

    def is_background_running(self) -> bool:
        """Returns True if the background update thread is running."""
        return self._running and self._thread is not None and self._thread.is_alive()

    def _update_loop(self) -> None:
        """Daemon loop: sleeps, then updates the state."""
        while self._running:
            time.sleep(self.update_interval_ms / 1000.0)

            # Fetch fresh spot from the pricing engine
            spot = self.pricing_engine.get_spot()
            if spot is None or spot <= 0:
                logger.debug("Skipping update: invalid spot from PricingEngine.")
                continue

            # Check required inputs
            if not self._option_specs:
                logger.debug("Skipping update: no option specs set.")
                continue

            try:
                with self._lock:
                    self._update_state(
                        spot=spot,
                        realized_vol_estimate=self._last_realized_vol,
                        option_specs=self._option_specs,
                    )
            except Exception as e:
                logger.error(f"Background state update failed: {e}", exc_info=True)

    # ------------------------------------------------------------------------
    # 2. Volatility Forecast Update
    # ------------------------------------------------------------------------

    def set_vol_forecast(self, realized_vol: float) -> None:
        """
        Update the subjective realized volatility forecast.

        This is called by the main loop whenever your statistical model
        (e.g., HAR-RV, GARCH) outputs a new volatility estimate.

        The new forecast will be used in the next background state update
        (or immediately if you call update_state() manually).
        """
        if realized_vol <= 0:
            logger.warning(f"Invalid realized vol: {realized_vol}. Ignoring.")
            return

        self._last_realized_vol = realized_vol
        logger.debug(f"Volatility forecast updated to {realized_vol:.4f}")

    # ------------------------------------------------------------------------
    # 3. State Update (Public and Internal)
    # ------------------------------------------------------------------------

    def update_state(
        self,
        spot: Optional[float] = None,
        realized_vol_estimate: Optional[float] = None,
        option_specs: Optional[List[OptionSpec]] = None,
    ) -> None:
        """
        Manually trigger a state update (for testing, or if auto_update=False).

        This method is synchronous and blocks until the update is complete.

        Args:
            spot: If provided, uses this spot. Otherwise fetches from PricingEngine.
            realized_vol_estimate: If provided, updates the forecast.
            option_specs: If provided, updates the portfolio.
        """
        # Store for background thread reuse
        if spot is None:
            spot = self.pricing_engine.get_spot()
        if realized_vol_estimate is not None:
            self._last_realized_vol = realized_vol_estimate
        if option_specs is not None:
            self._option_specs = option_specs
            self._N = len(option_specs)

        with self._lock:
            self._update_state(spot, self._last_realized_vol, self._option_specs)

    def _update_state(
        self,
        spot: float,
        realized_vol_estimate: float,
        option_specs: List[OptionSpec],
    ) -> None:
        """
        Internal state update (called by background thread or manually).

        This recomputes:
            1. Per-option Greeks and C0 (skipping options that cannot be priced).
            2. Risk matrix A.
            3. Order flow matrix D.
            4. Theta_2 (Equation 27).
            5. theta_1 (Equation 28 via numerical integration).
        """
        # 1. Filter out options that cannot be priced by the PricingEngine
        valid_specs = []
        valid_greeks = []
        C0_list = []

        r = self.pricing_engine.r
        q = self.pricing_engine.q

        for spec in option_specs:
            try:
                iv = self.pricing_engine.get_iv(spec.strike, spec.expiry, use_sabr=True)
            except (RuntimeError, ValueError) as e:
                logger.error(
                    f"PricingEngine error for {spec.id} (Strike={spec.strike}, Expiry={spec.expiry}): {e}. "
                    "Skipping this option from Lucic-Tse state update."
                )
                continue

            # Compute Greeks for this option
            greeks = self._compute_single_option_greeks(
                spot=spot,
                strike=spec.strike,
                T=spec.T,
                option_type=spec.option_type,
                iv=iv,
            )

            # Compute C0
            sigma_realized = realized_vol_estimate
            sigma_imp = iv
            dollar_gamma = spot * spot * greeks['gamma']
            C0 = 0.5 * (sigma_realized**2 - sigma_imp**2) * dollar_gamma

            valid_specs.append(spec)
            valid_greeks.append(greeks)
            C0_list.append(C0)

        # 2. If no valid options, abort the state update
        if not valid_specs:
            logger.error("No valid options with IVs. State update aborted.")
            with self._lock:
                self._is_initialized = False
                self._N = 0
            return

        # 3. Update internal state using ONLY the valid options
        self._option_specs = valid_specs
        self._N = len(valid_specs)
        self._C0 = np.array(C0_list)

        # Update per-option parameters (only for valid ones)
        self._kappa_a = self._get_per_option('kappa_a', valid_specs)
        self._kappa_b = self._get_per_option('kappa_b', valid_specs)
        self._lambda0_a = self._get_per_option('lambda0_a', valid_specs)
        self._lambda0_b = self._get_per_option('lambda0_b', valid_specs)

        # 4. Build risk matrix A
        self._A = self._build_risk_matrix_A(valid_specs, valid_greeks)

        # 5. Build order flow diagonal matrix D
        self._D = np.diag(self._lambda0_b * self._kappa_b + self._lambda0_a * self._kappa_a)

        # 6. Compute Theta_2 (Equation 27)
        T_horizon = self.horizon_hours / (24 * 365)  # convert hours to years
        # At t=0, time_remaining = T_horizon
        time_remaining = T_horizon

        factor = 2 *np.exp(-1.0) * time_remaining
        I = np.eye(self._N)
        mat = I + factor * self._D @ self._A # Matrix to invert
        try:
            inv_mat = inv(mat)
        except np.linalg.LinAlgError:
            logger.warning("Matrix inversion failed. Using pseudo-inverse.")
            inv_mat = np.linalg.pinv(mat)

        self._Theta_2 = -self._A @ inv_mat
        self._diag_theta2 = np.diag(self._Theta_2)

        # 7. Compute theta_1 (Equation 28) via numerical integration
        self._theta_1 = self._compute_theta_1_integral(T_horizon)

        self._is_initialized = True
        logger.debug(f"State updated with {self._N} valid options.")

    def _get_per_option(self, key: str, option_specs: List[OptionSpec]) -> np.ndarray:
        """Extract per-option parameters (lambda0, kappa) from order_flow_params."""
        base = self.order_flow_params.get(key)
        N = len(option_specs)

        if isinstance(base, (int, float)):
            return np.full(N, base)
        elif isinstance(base, (list, np.ndarray)):
            if len(base) != N:
                raise ValueError(f"Length of {key} must match number of options.")
            return np.array(base)
        else:
            # Defaults: if not provided, use paper's typical values
            if 'lambda' in key:
                return np.full(N, 50 * 252)  # 50 fills per day at zero spread
            else:  # kappa
                return np.full(N, 0.75)

    def _build_risk_matrix_A(
        self,
        option_specs: List[OptionSpec],
        greeks_list: List[Dict]
    ) -> np.ndarray:
        """
        Build the risk penalty matrix A from risk factors.
        """
        N = len(option_specs)
        E = len(self.risk_factors)
        A = np.zeros((N, N))

        for k, factor in enumerate(self.risk_factors):
            alpha_k = factor.get('alpha', 1.0)
            membership_func = factor.get('membership')
            weight_key = factor.get('weight_key', 'vega')

            # Build v_k: N-dimensional vector for this factor
            v_k = np.zeros(N)
            for i, spec in enumerate(option_specs):
                if membership_func(spec):
                    # Option belongs to this factor
                    weight = greeks_list[i].get(weight_key, 1.0)
                    v_k[i] = weight

            # Add this factor's contribution to A
            A += alpha_k * np.outer(v_k, v_k)

        # Scale by overall risk aversion
        A *= self.risk_aversion

        return A

    def _compute_theta_1_integral(
        self,
        T_horizon: float,
    ) -> np.ndarray:
        """
        Compute theta_1(t) via numerical integration (Equation 28).

        theta_1(t) = (I + 2*e^{-1}*(T-t)*A*D)^-1 *
                     ∫_t^T (I + 2*e^{-1}*(T-u)*A*D) * f(u) du

        where f(u) = C0 + 2*e^{-1} * Theta_2(u)*(lambda_b - lambda_a)
                     + 2*e^{-1} * Theta_2(u)*diag(Theta_2(u))*(lambda_b*kappa_b - lambda_a*kappa_a)

        We use Gaussian quadrature (4-point) for the integral.
        """
        N = self._N
        e = np.exp(1.0)
        t = 0.0  # we're at the start of the horizon

        # Precompute constants
        lambda_b = self._lambda0_b
        lambda_a = self._lambda0_a
        kappa_b = self._kappa_b
        kappa_a = self._kappa_a
        D_mat = self._D
        A_mat = self._A

        # Define f(u) as a function of time u
        def f(u):
            # Theta_2(u) = -A * (I + 2*e^{-1}*(T-u)*D*A)^-1
            factor_u = 2 * e**(-1) * (T_horizon - u)
            I = np.eye(N)
            mat_u = I + factor_u * D_mat @ A_mat
            try:
                inv_u = inv(mat_u)
            except np.linalg.LinAlgError:
                inv_u = np.linalg.pinv(mat_u)
            Theta_2_u = -A_mat @ inv_u

            # Terms
            term1 = self._C0
            term2 = 2 * e**(-1) * Theta_2_u @ (lambda_b - lambda_a)
            term3 = 2 * e**(-1) * Theta_2_u @ np.diag(Theta_2_u) @ (lambda_b * kappa_b - lambda_a * kappa_a)
            return term1 + term2 + term3

        # Gauss-Legendre quadrature points (4 points, normalized to [-1,1])
        x, w = np.polynomial.legendre.leggauss(4)
        a = t
        b = T_horizon
        t_mid = (a + b) / 2
        t_half = (b - a) / 2
        integral = np.zeros(N)

        for xi, wi in zip(x, w):
            u = t_mid + t_half * xi
            fu = f(u)
            # Compute (I + 2*e^{-1}*(T-u)*A*D)
            factor_u = 2 * e**(-1) * (T_horizon - u)
            I = np.eye(N)
            mat_u = I + factor_u * D_mat @ A_mat
            integrand = mat_u @ fu
            integral += wi * integrand

        integral *= t_half

        # Pre-factor
        factor_t = 2 * e**(-1) * (T_horizon - t)
        I = np.eye(N)
        mat_t = I + factor_t * D_mat @ A_mat
        try:
            inv_t = inv(mat_t)
        except np.linalg.LinAlgError:
            inv_t = np.linalg.pinv(mat_t)

        theta_1 = inv_t @ integral
        return theta_1

    # ------------------------------------------------------------------------
    # 4. Inventory Aggregation
    # ------------------------------------------------------------------------

    def aggregate_inventory(
        self,
        positions: Dict[str, int],
        option_specs: List[OptionSpec],
    ) -> InventoryGreeks:
        """
        Aggregate Greeks from current positions.

        Args:
            positions: Dict of {option_id: quantity}.
            option_specs: List of option specs.

        Returns:
            InventoryGreeks: Aggregated Greeks.
        """
        # Set spot
        spot = self.pricing_engine.get_spot()

        # Parameters to return
        spec_map = {spec.id: spec for spec in option_specs}
        delta_total = 0.0
        gamma_total = 0.0
        vega_total = 0.0
        theta_total = 0.0
        vega_by_tenor = {}
        delta_by_expiry = {}
        gamma_by_expiry = {}

        for opt_id, quantity in positions.items():
            if quantity == 0:
                continue
            spec = spec_map.get(opt_id)
            if spec is None:
                continue

            # Compute Greeks for this option
            try:
                iv = self.pricing_engine.get_iv(spec.strike, spec.expiry, use_sabr=True)
            except (RuntimeError, ValueError) as e:
                logger.error(f"Skipping {opt_id} in inventory aggregation: {e}")
                continue

            greeks = self._compute_single_option_greeks(
                spot=spot,
                strike=spec.strike,
                T=spec.T,
                option_type=spec.option_type,
                iv=iv,
            )

            # Aggregate
            delta_total += greeks['delta'] * quantity
            gamma_total += greeks['gamma'] * quantity
            vega_total += greeks['vega'] * quantity
            theta_total += greeks['theta'] * quantity

            tenor = self._get_tenor_bucket(spec.T)
            vega_by_tenor[tenor] = vega_by_tenor.get(tenor, 0.0) + greeks['vega'] * quantity
            delta_by_expiry[spec.expiry] = delta_by_expiry.get(spec.expiry, 0.0) + greeks['delta'] * quantity
            gamma_by_expiry[spec.expiry] = gamma_by_expiry.get(spec.expiry, 0.0) + greeks['gamma'] * quantity

        return InventoryGreeks(
            delta=delta_total,
            gamma=gamma_total,
            vega=vega_total,
            theta=theta_total,
            vega_by_tenor=vega_by_tenor,
            delta_by_expiry=delta_by_expiry,
            gamma_by_expiry=gamma_by_expiry,
        )

    def _compute_single_option_greeks(
        self,
        spot: float,
        strike: float,
        T: float,
        option_type: str,
        iv: float,
    ) -> Dict[str, float]:
        """
        Compute Greeks for a single option using analytic formulas.

        Args:
            spot: Current spot price.
            strike: Strike price.
            T: Time to expiry in years.
            option_type: 'call' or 'put'.
            iv: Implied volatility (guaranteed by caller to be valid).

        Returns:
            Dict with keys: delta, gamma, vega, theta (daily).
        """
        # Take from pricing engine (one source of truth)
        r = self.pricing_engine.r
        q = self.pricing_engine.q

        # Constants in BSM
        d1 = (np.log(spot / strike) + (r - q + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)

        # Delta
        if option_type == 'call':
            delta = np.exp(-q * T) * norm_cdf(d1)
        else:
            delta = -np.exp(-q * T) * norm_cdf(-d1)

        # Gamma (same for call/put)
        gamma = norm_pdf(d1) * np.exp(-q * T) / (spot * iv * np.sqrt(T))

        # Vega (same for call/put)
        vega = spot * np.exp(-q * T) * np.sqrt(T) * norm_pdf(d1)

        # Theta: daily time decay (annual theta divided by 252 trading days)
        if option_type == 'call':
            theta_annual = -spot * norm_pdf(d1) * iv * np.exp(-q * T) / (2 * np.sqrt(T)) \
                           - r * strike * np.exp(-r * T) * norm_cdf(d2) \
                           + q * spot * np.exp(-q * T) * norm_cdf(d1)
        else:
            theta_annual = -spot * norm_pdf(d1) * iv * np.exp(-q * T) / (2 * np.sqrt(T)) \
                           + r * strike * np.exp(-r * T) * norm_cdf(-d2) \
                           - q * spot * np.exp(-q * T) * norm_cdf(-d1)

        theta = theta_annual / 252.0  # convert to daily

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
        }

    # Fine for now - can be improved by making adjustable by user.
    def _get_tenor_bucket(self, T: float) -> str:
        """Returns the tenor bucket for a given time-to-expiry in years."""
        days = T * 365
        for bucket, max_days in TENOR_BUCKETS.items():
            if days <= max_days:
                return bucket
        # Fallback (should never be reached because 90D+ catches everything)
        return "90D+"

    # ------------------------------------------------------------------------
    # Quote Generation (Live Tick)
    # ------------------------------------------------------------------------

    def generate_quotes(
        self,
        option_specs: List[OptionSpec],
        positions: Dict[str, int],
    ) -> Dict[str, Quote]:
        """
        Generate bid/ask quotes for a list of options.

        This is called on every tick (microseconds latency).

        It also updates the internal option_specs for the background thread.

        Args:
            option_specs: List of type OptionSpec.
            positions: Dict of {option_id: quantity} (current inventory).

        Returns:
            Dict[str, Quote]: Mapping of option_id -> Quote.
        """
        with self._lock:
            # 1. Validate and filter option_specs to only those that can be priced
            valid_specs = []
            for spec in option_specs:
                try:
                    iv = self.pricing_engine.get_iv(spec.strike, spec.expiry, use_sabr=True)
                except Exception as e:
                    logger.error(
                        f"Ignoring {spec.id} in portfolio: PricingEngine error: {e}. "
                        "This option will not be quoted until the PricingEngine can price it."
                    )
                    continue
                # If we get here, the option is priciable
                valid_specs.append(spec)

            if not valid_specs:
                logger.error("No valid options in the portfolio. Returning empty quotes.")
                return {}

            # 2. Update stored option specs (only the valid ones)
            if self._option_specs != valid_specs:
                logger.info(f"Option specs updated. Valid count: {len(valid_specs)}.")
                self._option_specs = valid_specs
                self._N = len(valid_specs)

            # 3. Build inventory vector q (only for valid options)
            idx_map = {spec.id: i for i, spec in enumerate(self._option_specs)}
            q_vec = np.zeros(self._N)
            for opt_id, quantity in positions.items():
                if opt_id in idx_map:
                    q_vec[idx_map[opt_id]] = quantity
                else:
                    logger.debug(f"Inventory position {opt_id} ignored: not in valid portfolio.")

            # 4. If engine is not initialized, use fallback spreads
            if not self._is_initialized:
                logger.warning("Engine not initialized. Using fallback spreads (fair value ± 2%).")
                fallback_spread_pct = 0.02
                quotes = {}
                for spec in valid_specs:
                    fair_value = self.pricing_engine.get_price(
                        spec.strike, spec.expiry, spec.option_type, use_sabr=True
                    )
                    if np.isnan(fair_value) or fair_value <= 0:
                        continue
                    spread = fallback_spread_pct * fair_value
                    quotes[spec.id] = Quote(
                        option_id=spec.id,
                        strike=spec.strike,
                        expiry=spec.expiry,
                        T=spec.T,
                        option_type=spec.option_type,
                        bid=max(fair_value - spread, 0.01),
                        ask=fair_value + spread,
                        bid_size=self.default_bid_size,
                        ask_size=self.default_ask_size,
                        fair_value=fair_value,
                        spread=2 * spread,
                    )
                return quotes

            # 5. Compute spreads (Equations 29-30)
            e = np.exp(1.0)
            ask_spreads = 1.0 / self._kappa_a + self._theta_1 - self._diag_theta2 + 2 * self._Theta_2 @ q_vec
            bid_spreads = 1.0 / self._kappa_b - self._theta_1 - self._diag_theta2 - 2 * self._Theta_2 @ q_vec

            # 6. Generate quotes
            quotes = {}
            for i, spec in enumerate(valid_specs):
                # Fair value
                fair_value = self.pricing_engine.get_price(
                    spec.strike, spec.expiry, spec.option_type, use_sabr=True
                )
                if np.isnan(fair_value) or fair_value <= 0:
                    continue

                ask_spread = ask_spreads[i] if i < len(ask_spreads) else 0.0
                bid_spread = bid_spreads[i] if i < len(bid_spreads) else 0.0

                # Ensure positive spread
                ask_spread = max(ask_spread, 0.01)
                bid_spread = max(bid_spread, 0.01)

                bid = max(fair_value - bid_spread, 0.01)
                ask = fair_value + ask_spread

                quotes[spec.id] = Quote(
                    option_id=spec.id,
                    strike=spec.strike,
                    expiry=spec.expiry,
                    T=spec.T,
                    option_type=spec.option_type,
                    bid=bid,
                    ask=ask,
                    bid_size=self.default_bid_size,
                    ask_size=self.default_ask_size,
                    fair_value=fair_value,
                    spread=ask_spread + bid_spread,
                )

            return quotes

    # ------------------------------------------------------------------------
    # 6. Helpers
    # ------------------------------------------------------------------------

    def is_initialized(self) -> bool:
        """Returns True if the engine has been initialized (state updated at least once)."""
        with self._lock:
            return self._is_initialized

    def close(self) -> None:
        """Clean up resources (stop background thread)."""
        self.stop_background_update()


# ============================================================================
# 3. Helper Functions (NumPy-based, fast)
# ============================================================================

def norm_cdf(x):
    """Standard normal cumulative distribution function (fast, NumPy)."""
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def norm_pdf(x):
    """Standard normal probability density function (fast, NumPy)."""
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)