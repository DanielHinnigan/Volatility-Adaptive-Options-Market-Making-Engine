# Raw SVI per expiry
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np
from scipy.optimize import minimize

def svi_total_variance(k, a, b, rho, m, sigma):
    """Raw SVI total variance."""
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def svi_iv(k, a, b, rho, m, sigma, T):
    """Raw SVI Implied Volatility."""
    return np.sqrt(svi_total_variance(k, a, b, rho, m, sigma) / T)

def calibrate_raw_svi(k_values, iv_values, T):
    """
    Calibrate Raw SVI for one expiry.
    k_values: log-moneyness array
    iv_values: market implied volatilities
    T: time-to-expiry in y-ears
    """
    # Empty arrays return none
    if len(k_values) == 0 or len(iv_values) == 0:
        return None

    # Initial guess (heuristic)
    atm_idx = np.argmin(np.abs(k_values))
    atm_iv = iv_values[atm_idx]
    a_init = (atm_iv ** 2) * T
    b_init = 0.1
    rho_init = -0.3
    m_init = 0.0
    sigma_init = 0.1


    
    def objective(params):
        a, b, rho, m, sigma = params

        # Enforce bounds
        if b <= 0 or abs(rho) >= 1 or sigma <= 0:
            return 1e10
        
        # Enforce minimum total variance > 0
        min_variance = a + b * sigma * np.sqrt(1 - rho**2)
        if min_variance < 0:
            return 1e10
        
        # Fit using SVI
        fitted_iv = svi_iv(k_values, a, b, rho, m, sigma, T)

        # Return squared error
        return np.sum((iv_values - fitted_iv)**2)
    
    result = minimize(
        objective,
        x0=[a_init, b_init, rho_init, m_init, sigma_init],
        method='SLSQP',
        bounds=[
            (None, None),      # a unconstrained
            (1e-6, None),   # b > 0
            (-0.99, 0.99),  # |rho| < 1
            (None, None),   # m unconstrained
            (1e-6, None)    # sigma > 0
        ],
        options={'maxiter': 1000, 'ftol': 1e-4}
    )

    # If SLSQP fails, use the more robust Nelder-Mead minimizer
    if result.success:
        return result.x
    
    result_nm = minimize(
        objective,
        x0=[a_init, b_init, rho_init, m_init, sigma_init],
        method='Nelder-Mead',
        options={'maxiter': 1000, 'ftol': 1e-4}
    )
    
    return result_nm.x if result_nm.success else None

@dataclass
class SVICalibrationResult:
    """
    Structured result of an SVI calibration.
    
    Attributes:
        success: Whether the calibration converged.
        params: Tuple of (a, b, rho, m, sigma) if success is True, else None.
        message: Additional diagnostic message (e.g., optimizer status).
        fitted_ivs: Fitted IVs evaluated at the input strikes (for validation).
        strikes: Input strikes used for calibration.
        ivs: Input IVs used for calibration.
    """
    success: bool
    params: Optional[tuple]
    message: str
    fitted_ivs: Optional[np.ndarray]
    strikes: np.ndarray
    ivs: np.ndarray

def fit_svi_smile(
    strikes: np.ndarray,
    ivs: np.ndarray,
    T: float,
    forward: float,
    r: float,
    q: float
) -> SVICalibrationResult:
    """
    Public interface for calibrating a single SVI smile.
    
    This function:
        1. Computes log-moneyness k = ln(strike / forward).
        2. Calls the low-level optimizer `calibrate_raw_svi`.
        3. Returns a structured `SVICalibrationResult` object.
    
    Args:
        strikes: Array of strike prices (should be already filtered to OTM).
        ivs: Array of market implied volatilities (matching strikes).
        T: Time to expiry in years.
        forward: Forward price of the underlying.
        r: Risk-free rate (used only for forward computation consistency, 
            but passed here for metadata).
        q: Dividend yield (same as above).
    
    Returns:
        SVICalibrationResult: Contains success flag, parameters, and metadata.
    
    Example:
        >>> strikes = np.array([730, 735, 740, 745, 750, 755, 760])
        >>> ivs = np.array([0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.055])
        >>> T = 30 / 365
        >>> forward = 745.0
        >>> r = 0.05
        >>> q = 0.0
        >>> result = fit_svi_smile(strikes, ivs, T, forward, r, q)
        >>> if result.success:
        ...     a, b, rho, m, sigma = result.params
        ...     print(f"SVI fit successful: a={a:.4f}, b={b:.4f}, rho={rho:.4f}")
        ... else:
        ...     print(f"Calibration failed: {result.message}")
    """
    # Input validation
    if len(strikes) == 0 or len(ivs) == 0:
        return SVICalibrationResult(
            success=False,
            params=None,
            message="Empty arrays provided.",
            fitted_ivs=None,
            strikes=strikes,
            ivs=ivs
        )
    
    if len(strikes) != len(ivs):
        return SVICalibrationResult(
            success=False,
            params=None,
            message="Strikes and IVs arrays must have the same length.",
            fitted_ivs=None,
            strikes=strikes,
            ivs=ivs
        )
    
    if T <= 0:
        return SVICalibrationResult(
            success=False,
            params=None,
            message=f"Invalid T: {T} (must be > 0).",
            fitted_ivs=None,
            strikes=strikes,
            ivs=ivs
        )
    
    if forward <= 0:
        return SVICalibrationResult(
            success=False,
            params=None,
            message=f"Invalid forward: {forward} (must be > 0).",
            fitted_ivs=None,
            strikes=strikes,
            ivs=ivs
        )
    
    # Compute log-moneyness
    k = np.log(strikes / forward)
    
    # Call the low-level optimizer
    params = calibrate_raw_svi(k, ivs, T)
    
    if params is None:
        return SVICalibrationResult(
            success=False,
            params=None,
            message="Optimizer failed to converge. Check data quality.",
            fitted_ivs=None,
            strikes=strikes,
            ivs=ivs
        )
    
    # Compute fitted IVs for validation (useful for debugging)
    a, b, rho, m, sigma = params
    fitted_ivs = svi_iv(k, a, b, rho, m, sigma, T)
    
    return SVICalibrationResult(
        success=True,
        params=tuple(params),
        message="Calibration successful.",
        fitted_ivs=fitted_ivs,
        strikes=strikes,
        ivs=ivs
    )
