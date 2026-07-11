# Interface for pricing options (calculating the theoretical value of options)
import threading
import time
import copy
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np

from .data.connector_factory import ConnectorFactory
from .data.base_connector import DataConnector
from .models.bsm import implied_volatility, black_scholes_call, black_scholes_put
from .models.svi import fit_svi_smile, svi_total_variance
from .models.ssvi import calibrate_ssvi, ssvi_iv
from .models.sabr import calibrate_sabr, sabr_iv
from .utils.time_utils import compute_time_to_expiry
from .utils.data_preprocessing import filter_otm_for_calibration
from .config import settings as global_settings

# Set up logger for this module
logger = logging.getLogger(__name__)

class PricingEngine:
    """
    High-level pricing facade for a single underlying symbol.

    The engine manages its own data fetching, calibration, and spot price.
    It runs a background thread for periodic surface recalibration.

    Usage Example:
        engine = PricingEngine(symbol="SPY")
        engine.start_background_calibration(interval_ms=200)

        # On every tick:
        price = engine.get_price(strike=750, expiry="2026-07-13")
        iv = engine.get_iv(strike=750, expiry="2026-07-13")

        # Gracefully shut down:
        engine.stop_background_calibration()
        engine.close()
    """

    def __init__(
        self,
        symbol: str,
        r: Optional[float] = None,
        q: Optional[float] = None,
        max_expiries: int = 5,
        data_provider: str = "yfinance",
        cache: bool = True,
        connector: Optional[DataConnector] = None
    ):
        """
        Initialize the pricing engine for a single symbol.

        Args:
            symbol: The underlying symbol (e.g., "SPY").
            r: Risk-free rate (constant for now).
            q: Dividend yield.
            max_expiries: Number of expiries to use for surface calibration.
            cache: Should the retrieved option chains be taken from cache or should a fresh retrieve be used
            data_provider: "yfinance" or future providers.
            connector: Used for providing synthetic data when performing unit testing. Should not be used as standalone
        """
        self.symbol = symbol
        self.r = r
        self.q = q
        self.max_expiries = max_expiries
        self.cache = cache

        # Use injected connector, or create a default one
        if connector is not None:
            self._connector = connector
        else:
            self._connector = ConnectorFactory.get_connector(
                provider=data_provider,
            )

        # Use provided values or fall back to global settings
        self.r = r if r is not None else global_settings.R
        self.q = q if q is not None else global_settings.Q

        # Internal state (protected by a lock)
        self._lock = threading.RLock()
        self._svi_slices: Dict[str, Dict] = {}       # expiry -> SVI params + metadata
        self._ssvi_params: Dict[str, float] = {}     # {'rho': ..., 'eta': ..., 'gamma': ...}
        self._sabr_params: Dict[str, Dict] = {}      # expiry -> {'alpha': ..., 'rho': ..., 'nu': ..., 'beta': 0.5}
        self._surface_ready = False

        # Spot price management
        self._spot: Optional[float] = None
        self._last_spot_update: Optional[datetime] = None

        # Background thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._interval_ms = 200

    # -------------------------------------------------------------------------
    # Background Recalibration Thread
    # -------------------------------------------------------------------------

    def start_background_calibration(self, interval_ms: int = 200) -> None:
        """
        Performs an immediate calibration, then starts a daemon thread
        that recalibrates every `interval_ms`.
        """
        if self._running:
            logger.warning("Background calibration already running. Ignoring start request.")
            return

        self._interval_ms = interval_ms

        try:
            # Initial calibration (synchronous, blocks until done)
            logger.info("Starting initial calibration...")
            self._calibrate()
            logger.info("Initial calibration completed successfully.")
        except Exception as e:
            logger.error(f"Initial calibration failed: {e}", exc_info=True)
            raise RuntimeError("PricingEngine failed to initialize.") from e

        # Start daemon thread
        self._running = True
        self._thread = threading.Thread(target=self._recalibration_loop, daemon=True)
        self._thread.start()
        logger.info(f"Background calibration thread started (interval={interval_ms}ms).")


    def stop_background_calibration(self) -> None:
        """Stops the background recalibration thread safely."""
        if not self._running:
            return
        logger.info("Stopping background calibration thread...")
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("Background calibration thread stopped.")

    def is_background_running(self) -> bool:
        """Returns true if background (calibration) thread is running, otherwise returns false."""
        return self._running and self._thread is not None and self._thread.is_alive()

    def _recalibration_loop(self) -> None:
        """Daemon loop: sleeps, then recalibrates."""
        while self._running:
            time.sleep(self._interval_ms / 1000.0)
            try:
                with self._lock:
                    self._calibrate()
                logger.debug("Background recalibration completed successfully.")
            except Exception as e:
                logger.error(f"Background recalibration failed: {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # Core Calibration (Internal on background thread)
    # -------------------------------------------------------------------------

    def _calibrate(self) -> None:
        """
        Full calibration:
            1. Fetch data from the data connector.
            2. Compute IVs (via BSM).
            3. Fit SVI per expiry.
            4. Fit SSVI globally.
            5. Fit SABR per expiry (calibrated to SSVI surface).
        """
        logger.debug(f"Calibrating surface for {self.symbol}...")

        # 1. Fetch data
        expiries = self._connector.get_available_expiries()
        if not expiries:
            raise RuntimeError(f"No expiries available for {self.symbol}")

        expiries = expiries[:self.max_expiries]
        raw_chains = {}
        for exp in expiries:
            raw_chains[exp] = self._connector.get_chain_for_expiry(exp, use_cache=self.cache)

        # 2. Compute IVs and prepare SVI slices
        spot = self._connector.get_spot_price()
        self._spot = spot
        self._last_spot_update = datetime.now()
        logger.debug(f"Spot price: {spot:.2f}")

        svi_slices = []
        for exp, raw in raw_chains.items():
            slice_data = self._prepare_svi_slice(exp, raw, spot)
            if slice_data is not None:
                svi_slices.append(slice_data)

        if len(svi_slices) < 2:
            raise RuntimeError(f"Only {len(svi_slices)} slices available – need at least 2 for SSVI.")

        # 3. Build SSVI input
        ssvi_input = {}
        for s in svi_slices:
            ssvi_input[s["expiry"]] = {
                "k": s["k"],
                "iv": s["ivs"],
                "theta": s["theta"],
                "T": s["T"],
            }

        # 4. Calibrate SSVI
        gamma = 0.5  # fixed for now
        ssvi_result = calibrate_ssvi(ssvi_input, gamma=gamma)
        if not ssvi_result.success:
            raise RuntimeError(f"SSVI calibration failed: {ssvi_result.message}")
        rho_ssvi, eta_ssvi = ssvi_result.params

        # 5. Calibrate SABR per expiry to the SSVI surface
        sabr_params = {}
        for s in svi_slices:
            k = s["k"]
            T = s["T"]
            theta = s["theta"]

            # Target IVs from SSVI
            iv_ssvi_target = ssvi_iv(k, theta, rho_ssvi, eta_ssvi, T, gamma)
            
            # Calibrate SABR
            sabr_res = calibrate_sabr(
                strikes=s["strikes"],
                ivs=iv_ssvi_target,
                T=T,
                forward=s["forward"],
                beta=0.5,
                max_retries=3
            )
            if sabr_res.success:
                alpha, rho, nu = sabr_res.params
                sabr_params[s["expiry"]] = {
                    "alpha": alpha,
                    "rho": rho,
                    "nu": nu,
                    "beta": 0.5,
                }
            else:
                logger.warning(f"SABR calibration failed for {s['expiry']}. SSVI will be used as fallback.")
                               
        # 6. Store results
        with self._lock:
            self._svi_slices = {s["expiry"]: s for s in svi_slices}
            self._ssvi_params = {"rho": rho_ssvi, "eta": eta_ssvi, "gamma": gamma}
            self._sabr_params = sabr_params
            self._surface_ready = True

    def _prepare_svi_slice(self, expiry: str, raw_data: Dict, spot: float) -> Optional[Dict]:
        """
        Prepare a single expiry slice for SVI calibration.
        Returns a dict with keys: expiry, T, forward, strikes, ivs, k, theta, svi_params.
        """
        T = compute_time_to_expiry(expiry)
        forward = spot * np.exp((self.r - self.q) * T)

        # Compute IVs for all options first
        all_ops = raw_data["calls"] + raw_data["puts"]
        for opt in all_ops:
            if opt.mid > 0 and opt.bid > 0 and opt.ask > 0:
                iv = implied_volatility(
                    price=opt.mid,
                    S=spot,
                    K=opt.strike,
                    T=T,
                    r=self.r,
                    q=self.q,
                    option_type=opt.option_type
                )
                if not np.isnan(iv):
                    opt.implied_vol = iv

        # Filter OTM
        strikes, ivs = filter_otm_for_calibration(
            all_ops,
            spot=spot,
            T=T,
            r=self.r,
            q=self.q,
            max_spread_pct=0.5
        )

        if len(strikes) < 5:
            logger.warning(f"{expiry}: Only {len(strikes)} OTM options found. Skipping slice.")
            return None

        # Calibrate Raw SVI
        result = fit_svi_smile(strikes, ivs, T, forward, r=self.r, q=self.q)
        if not result.success:
            logger.warning(f"{expiry}: SVI calibration failed. Skipping slice.")
            return None

        a, b, rho, m, sigma = result.params
        theta = svi_total_variance(0.0, a, b, rho, m, sigma)
        if theta <= 0:
            logger.warning(f"{expiry}: theta = {theta:.6f} <= 0. Skipping slice.")
            return None

        k = np.log(strikes / forward)

        return {
            "expiry": expiry,
            "T": T,
            "forward": forward,
            "strikes": strikes,
            "ivs": ivs,
            "k": k,
            "theta": theta,
            "svi_params": result.params,
        }

    # -------------------------------------------------------------------------
    # Spot Price Management
    # -------------------------------------------------------------------------

    def _ensure_fresh_spot(self) -> None:
        """Fetches fresh spot if the current spot is stale (> 1s)."""
        now = datetime.now()
        if self._spot is None or self._last_spot_update is None:
            self._spot = self._connector.get_spot_price()
            self._last_spot_update = now
            return

        if (now - self._last_spot_update).total_seconds() > 1.0:
            self._spot = self._connector.get_spot_price()
            self._last_spot_update = now

    def get_spot(self) -> float:
        """Returns the current spot price."""
        self._ensure_fresh_spot()
        return self._spot

    # -------------------------------------------------------------------------
    # Core Public API (Pricing)
    # -------------------------------------------------------------------------

    def get_price(
        self,
        strike: float,
        expiry: str,
        option_type: str,
        use_sabr: bool = True
    ) -> float:
        """
        Returns the theoretical price for a single option.

        Args:
            strike: Strike price.
            expiry: Expiry string (e.g., "2026-07-13").
            option_type: "call" or "put".
            use_sabr: If True, uses SABR for dynamic re-pricing (fast).
                      If False, uses the SSVI surface directly (safe, slower).

        Returns:
            float: The theoretical price.
        """
        self._ensure_fresh_spot()
        T = compute_time_to_expiry(expiry)

        iv = self._get_iv_internal(strike, T, expiry, use_sabr)

        if option_type == 'call':
            return black_scholes_call(self._spot, strike, T, self.r, self.q, iv)
        else:
            return black_scholes_put(self._spot, strike, T, self.r, self.q, iv)

    def get_iv(
        self,
        strike: float,
        expiry: str,
        use_sabr: bool = True
    ) -> float:
        """
        Returns the implied volatility for a single option.

        Args:
            strike: Strike price.
            expiry: Expiry string.
            use_sabr: If True, uses SABR for dynamic re-pricing (fast).
                      If False, uses the SSVI surface directly (safe, slower).

        Returns:
            float: The implied volatility.
        """
        self._ensure_fresh_spot()
        T = compute_time_to_expiry(expiry)
        return self._get_iv_internal(strike, T, expiry, use_sabr)

    def _get_iv_internal(self, strike: float, T: float, expiry: str, use_sabr: bool) -> float:
        """
        Internal IV retrieval (called with spot already ensured fresh).
        """
        with self._lock:
            if not self._surface_ready:
                raise RuntimeError("PricingEngine is not calibrated yet. Call start_background_calibration() first.")

            if use_sabr and expiry in self._sabr_params:
                # SABR pricing
                sabr = self._sabr_params[expiry]
                forward = self._spot * np.exp((self.r - self.q) * T)
                iv = sabr_iv(
                    f=forward,
                    K=strike,
                    T=T,
                    alpha=sabr["alpha"],
                    beta=sabr["beta"],
                    rho=sabr["rho"],
                    nu=sabr["nu"]
                )
                if not np.isnan(iv):
                    return iv
                
                # Fallback to SSVI if SABR fails
                logger.warning(f"SABR returned NaN for {strike}, falling back to SSVI.")

            # SSVI fallback
            if expiry not in self._svi_slices:
                raise ValueError(f"Expiry {expiry} not available in the surface.")
            slice_data = self._svi_slices[expiry]
            theta = slice_data["theta"]
            rho_ssvi = self._ssvi_params["rho"]
            eta_ssvi = self._ssvi_params["eta"]
            gamma_ssvi = self._ssvi_params["gamma"]
            forward = self._spot * np.exp((self.r - self.q) * T)
            k = np.log(strike / forward)
            iv = ssvi_iv(k, theta, rho_ssvi, eta_ssvi, T, gamma_ssvi)
            if np.isnan(iv):
                raise RuntimeError(f"SSVI returned NaN for strike {strike}, expiry {expiry}.")
            return iv

    def get_atm_iv(self, expiry: str) -> float:
        """
        Returns the ATM implied volatility for a given expiry.
        """
        with self._lock:
            if not self._surface_ready:
                raise RuntimeError("PricingEngine is not calibrated yet.")

            if expiry not in self._svi_slices:
                raise ValueError(f"Expiry {expiry} not available.")

            slice_data = self._svi_slices[expiry]
            theta = slice_data["theta"]
            T = slice_data["T"]
            rho_ssvi = self._ssvi_params["rho"]
            eta_ssvi = self._ssvi_params["eta"]
            gamma_ssvi = self._ssvi_params["gamma"]

            # ATM is calculated when log-moneyness = 0, i.e. strike = forward price
            iv = ssvi_iv(0.0, theta, rho_ssvi, eta_ssvi, T, gamma_ssvi)
            return iv

    def get_forward(self, expiry: str) -> float:
        """Returns the forward price F = S * exp((r - q) * T)."""
        self._ensure_fresh_spot()
        T = compute_time_to_expiry(expiry)
        return self._spot * np.exp((self.r - self.q) * T)

    # -------------------------------------------------------------------------
    # Surface Inspection (Read-Only)
    # -------------------------------------------------------------------------

    def get_svi_params(self) -> Dict:
        with self._lock:
            return copy.deepcopy(self._svi_slices)

    def get_ssvi_params(self) -> Dict:
        with self._lock:
            return copy.deepcopy(self._ssvi_params)

    def get_sabr_params(self) -> Dict:
        with self._lock:
            return copy.deepcopy(self._sabr_params)

    def is_calibrated(self) -> bool:
        with self._lock:
            return self._surface_ready

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Stops the background thread and releases resources."""
        self.stop_background_calibration()