# Global SSVI surface
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

def ssvi_total_variance(k: np.ndarray, theta: float, rho: float, eta: float, gamma: float = 0.5) -> np.ndarray:
    """
    SSVI total variance: w(k, theta) = (theta/2) * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2))
    where phi = eta / theta^gamma.
    """
    # Check if input is valid
    if theta <= 0 or eta <= 0:
        return np.full_like(k, np.nan)
    
    # Calculate total variance
    phi = eta / (theta ** gamma)
    sqrt_term = np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2)
    return 0.5 * theta * (1 + rho * phi * k + sqrt_term)


def ssvi_iv(k: np.ndarray, theta: float, rho: float, eta: float, T: float, gamma: float = 0.5) -> np.ndarray:
    """
    SSVI Implied Volatility: sqrt(w / T).
    """

    # Check if input is valid
    if T <= 0:
        return np.full_like(k, np.nan)
    
    # Return IV
    w = ssvi_total_variance(k, theta, rho, eta, gamma)
    return np.sqrt(w / T)

@dataclass
class SSVICalibrationResult:
    """Structured result of an SSVI global calibration."""
    success: bool
    params: Optional[Tuple[float, float]]  # (rho, eta)
    gamma: float
    message: str
    fitted_ivs: Optional[Dict[str, np.ndarray]]  # Expiry -> fitted IVs
    slices: Dict[str, dict]  # Input data used for calibration (for reference)

# Public Interface for constructing (calibrating) the SSVI volatility surface
def calibrate_ssvi(
    slices: Dict[str, dict],
    gamma: float = 0.5,
    initial_rho: float = -0.3,
    initial_eta: float = 0.5,
) -> SSVICalibrationResult:
    """
    Calibrate the SSVI model globally across multiple expiries.

    Args:
        slices: Dictionary mapping expiry (string) to a dict with:
            - 'k': np.ndarray of log-moneyness
            - 'iv': np.ndarray of market implied volatilities
            - 'theta': float (ATM total variance)
            - 'T': float (time to expiry in years)
        gamma: Power-law exponent for phi(theta) = eta / theta^gamma.
        initial_rho: Initial guess for rho.
        initial_eta: Initial guess for eta.

    Returns:
        SSVICalibrationResult: Contains success flag, fitted params, and metadata.
    """
    
    # Validate input
    if not slices:
        return SSVICalibrationResult(
            success=False,
            params=None,
            gamma=gamma,
            message="No expiry slices provided.",
            fitted_ivs=None,
            slices=slices,
        )

    for expiry, data in slices.items():
        if 'k' not in data or 'iv' not in data or 'theta' not in data or 'T' not in data:
            return SSVICalibrationResult(
                success=False,
                params=None,
                gamma=gamma,
                message=f"Missing required fields in slice '{expiry}'. Need 'k', 'iv', 'theta', 'T'.",
                fitted_ivs=None,
                slices=slices,
            )
        if len(data['k']) == 0 or len(data['iv']) == 0:
            return SSVICalibrationResult(
                success=False,
                params=None,
                gamma=gamma,
                message=f"Empty arrays in slice '{expiry}'.",
                fitted_ivs=None,
                slices=slices,
            )
        if data['theta'] <= 0 or data['T'] <= 0:
            return SSVICalibrationResult(
                success=False,
                params=None,
                gamma=gamma,
                message=f"Invalid theta or T in slice '{expiry}'.",
                fitted_ivs=None,
                slices=slices,
            )

    # Define the objective function
    def objective(params):
        rho, eta = params

        # Basic parameter sanity (should follow bounds)
        if abs(rho) >= 0.99 or eta <= 0:
            return 1e10

        total_error = 0.0

        # Iterate over all expiries
        for expiry, data in slices.items():
            k = data['k']
            iv_market = data['iv']
            theta = data['theta']
            T = data['T']

            # Compute phi = eta / theta^gamma
            phi = eta / (theta ** gamma)

            # ENFORCE NO-BUTTERFLY ARBITRAGE (per expiry)
            # Condition: theta * phi <= 4 / (1 + |rho|)
            if theta * phi > 4.0 / (1.0 + abs(rho)):
                # Heavy penalty to force the optimizer away from arbitrage
                # The further the constraint is broken, the higher the penality
                return 1e10 + (theta * phi - 4.0 / (1.0 + abs(rho))) * 1e6

            # Compute fitted IVs using SSVI
            iv_fitted = ssvi_iv(k, theta, rho, eta, T, gamma)

            # Weighted least squares: May be updated in the future to be weighted (by liquitity, vega, etc.)
            error = np.sum((iv_market - iv_fitted) ** 2)
            total_error += error

        # ENFORCE NO-CALENDAR SPREAD ARBITRAGE (global)
        # For square-root SSVI (gamma=0.5), a sufficient condition is:
        # eta <= 4 / (1 + |rho|)
        if eta > 4.0 / (1.0 + abs(rho)):
            total_error += (eta - 4.0 / (1.0 + abs(rho))) * 1e6

        return total_error

    # Bounds and Optimization
    bounds = [
        (-0.99, 0.99),  # rho
        (1e-6, 10.0),   # eta (capped to avoid numerical blow-up)
    ]

    initial_guess = [initial_rho, initial_eta]

    # Run the optimization
    result = minimize(
        objective,
        x0=initial_guess,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )

    if not result.success:
        return SSVICalibrationResult(
            success=False,
            params=None,
            gamma=gamma,
            message=f"Optimization failed: {result.message}",
            fitted_ivs=None,
            slices=slices,
        )

    rho_opt, eta_opt = result.x

    # Compute Fitted IVs for all slices (for validation/debugging)
    fitted_ivs_dict = {}
    for expiry, data in slices.items():
        k = data['k']
        theta = data['theta']
        T = data['T']
        fitted_ivs_dict[expiry] = ssvi_iv(k, theta, rho_opt, eta_opt, T, gamma)

    return SSVICalibrationResult(
        success=True,
        params=(rho_opt, eta_opt),
        gamma=gamma,
        message=f"Calibration successful. Iterations: {result.nit}",
        fitted_ivs=fitted_ivs_dict,
        slices=slices,
    )