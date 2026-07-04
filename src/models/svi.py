# Raw SVI per expiry
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
    T: time-to-expiry in years
    """
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
        options={'maxiter': 500, 'ftol': 1e-8}
    )
    
    return result.x if result.success else None