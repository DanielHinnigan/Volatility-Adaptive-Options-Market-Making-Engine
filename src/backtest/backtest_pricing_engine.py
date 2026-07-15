"""
A custom PricingEngine used with the backtester.
This pricing engine inherits from the main pricing engine (in pricing_engine.py), but handles past expiries.
This creates a modular approach without altering the main pricing engine to accomodate the backtester.
"""
import numpy as np
import logging
from typing import Dict, Optional
from datetime import datetime

from ..pricing_engine import PricingEngine
from ..models.bsm import implied_volatility
from ..models.svi import fit_svi_smile, svi_total_variance
from ..utils.time_utils import compute_time_to_expiry
from ..utils.data_preprocessing import filter_otm_for_calibration

# Set up logger for this module
logger = logging.getLogger(__name__)

class HistoricalPricingEngine(PricingEngine):
    def __init__(self, initial_timestamp: datetime, *args, **kwargs):
        # Added member variable
        self.initial_timestamp = initial_timestamp

        # Initialize parent class
        super().__init__(*args, **kwargs)

    def _prepare_svi_slice(self, expiry: str, raw_data: Dict, spot: float) -> Optional[Dict]:
        """
        Prepare a single expiry slice for SVI calibration.
        Returns a dict with keys: expiry, T, forward, strikes, ivs, k, theta, svi_params.
        """
        T = compute_time_to_expiry(expiry, self.initial_timestamp)
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
# Programmatic Docstring Modification
# -------------------------------------------------------------------------
if PricingEngine.__init__.__doc__:
    parent_doc = PricingEngine.__init__.__doc__
    
    # Define the explanation for your new variable
    x_doc = "            x: An integer representing your custom parameter (e.g., lookback window or threshold).\n"
    
    # Locate the 'Args:' section and inject the new variable documentation
    if "Args:" in parent_doc:
        parts = parent_doc.split("Args:\n", 1)
        modified_doc = f"{parts[0]}Args:\n{x_doc}{parts[1]}"
    else:
        # Fallback if 'Args:' isn't found for some reason
        modified_doc = parent_doc + f"\n\nCustom Args:\n{x_doc}"
        
    # Assign the dynamically built docstring to the child's __init__
    HistoricalPricingEngine.__init__.__doc__ = modified_doc