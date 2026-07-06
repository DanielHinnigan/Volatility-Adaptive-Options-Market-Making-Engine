import numpy as np
from typing import List, Tuple
from ..data.base_connector import OptionQuote

def filter_otm_for_calibration(
    options: List[OptionQuote],
    spot: float,
    T: float,
    r: float,
    q: float,
    max_spread_pct: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter options to keep only Out-of-The-Money (OTM) options suitable for SVI/SSVI calibration.
    
    This function:
    1. Calculates the forward price.
    2. Keeps only OTM options (calls with strike > forward, puts with strike < forward).
    3. Filters out options with wide spreads (> max_spread_pct).
    4. Returns arrays of (strike, implied_volatility) for calibration.
    
    Returns:
        strikes: np.ndarray of filtered strikes
        ivs: np.ndarray of corresponding implied volatilities
    """
    forward = spot * np.exp((r - q) * T)
    
    filtered_strikes = []
    filtered_ivs = []
    
    for opt in options:
        # Basic sanity checks
        if opt.mid <= 0 or opt.bid <= 0 or opt.ask <= 0:
            continue
        
        # Check spread width (avoid illiquid options)
        spread_pct = (opt.ask - opt.bid) / opt.mid
        if spread_pct > max_spread_pct:
            continue
        
        # OTM filter
        if opt.option_type == 'call' and opt.strike <= forward:
            continue  # ITM call -> discard
        if opt.option_type == 'put' and opt.strike >= forward:
            continue  # ITM put -> discard
        
        filtered_strikes.append(opt.strike)
        filtered_ivs.append(opt.implied_vol)
    
    return np.array(filtered_strikes), np.array(filtered_ivs)