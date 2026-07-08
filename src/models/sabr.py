# SABR Hagan equation + calibration
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional, Tuple

# ============================================================================
# Hagan Formula (Closed-Form approximation of IV)
# ============================================================================
def sabr_iv(
    f: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    eps: float = 1e-8,
) -> float:
    """
    Hagan's 2002 closed-form approximation for SABR Implied Volatility.
    
    Args:
        f: Current forward price (spot * exp((r-q)*T)).
        K: Strike price.
        T: Time to expiry in years.
        alpha: Stochastic volatility.
        beta: CEV elasticity (fixed to 0.5 for equities).
        rho: Correlation between asset and volatility.
        nu: Vol-of-vol.
        eps: Small epsilon to prevent division by zero.
    
    Returns:
        float: Implied volatility.
    """
    # Handle Edge Cases
    if K <= 0 or f <= 0 or T <= 0 or alpha <= 0 or nu <= 0:
        return np.nan
    
    # ATM case: if K == f, use the ATM formula to avoid z = 0
    if abs(K - f) < eps:
        # ATM IV simplifies to:
        # sigma_ATM = alpha / f^(1-beta) * (1 + ((1-beta)^2/24 * alpha^2/(f^(2-2*beta)) + (rho*beta*nu*alpha)/(4*f^(1-beta)) + (2-3*rho^2)/24 * nu^2) * T)
        # Return the ATM formula directly:
        term1 = (1 - beta)**2 / 24 * alpha**2 / (f**(2 - 2*beta))
        term2 = rho * beta * nu * alpha / (4 * f**(1 - beta))
        term3 = (2 - 3 * rho**2) / 24 * nu**2
        sigma_atm = alpha / f**(1 - beta) * (1 + (term1 + term2 + term3) * T)
        return max(sigma_atm, 1e-6)
    
    # General Case
    log_fk = np.log(f / K)
    f_mean = np.sqrt(f * K)
    z = (nu / alpha) * (f_mean ** (1 - beta)) * log_fk

    sqrt_term = np.sqrt(1 - 2 * rho * z + z**2)
    numerator = sqrt_term + z - rho
    denominator = 1 - rho
    
    if abs(denominator) < eps or numerator <= 0:
        # If rho is 1, or the numerator is non-positive, revert to a safe approximation.
        # This is extremely rare in practice.
        return np.nan
    
    chi = np.log(numerator / denominator)
    
    # Handle z close to 0 (ATM case again, but with a small epsilon)
    if abs(z) < eps:
        # Limit of z / chi(z) is 1
        z_chi = 1.0
    else:
        z_chi = z / chi
    
    # Denominator of the main term:
    fK_power = (f * K) ** ((1 - beta) / 2)
    log_sq = log_fk**2
    log_quad = log_fk**4
    bracket = 1 + ((1 - beta)**2 / 24) * log_sq + ((1 - beta)**4 / 1920) * log_quad
    denom = fK_power * bracket
    
    # Estimated IV using Hagan
    sigma = (alpha / denom) * z_chi
    
    # Multiplicative correction for time (the "shift" term)
    # This is the expansion around T=0.
    term1 = (1 - beta)**2 / 24 * alpha**2 / (fK_power**2)
    term2 = rho * beta * nu * alpha / (4 * (fK_power))
    term3 = (2 - 3 * rho**2) / 24 * nu**2
    shift = 1 + (term1 + term2 + term3) * T
    
    sigma = sigma * shift
    
    # Safety clamp
    return max(sigma, 1e-6)


# ============================================================================
# Vectorized Version (can handle array of strikes)
# ============================================================================

def sabr_iv_vectorized(
    f: float,
    K: np.ndarray,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Vectorized SABR IV for multiple strikes."""
    ivs = []
    for k in K:
        iv = sabr_iv(f, k, T, alpha, beta, rho, nu)
        ivs.append(iv)
    return np.array(ivs)


# ============================================================================
# Dataclass for Calibration Result
# ============================================================================

@dataclass
class SABRCalibrationResult:
    success: bool
    params: Optional[Tuple[float, float, float]]  # (alpha, rho, nu)
    beta: float
    message: str
    fitted_ivs: Optional[np.ndarray]
    strikes: np.ndarray
    ivs: np.ndarray


# ============================================================================
# Robust Calibrator
# ============================================================================

def calibrate_sabr(
    strikes: np.ndarray,
    ivs: np.ndarray,
    T: float,
    forward: float,
    beta: float = 0.5,
    initial_guess: Optional[Tuple[float, float, float]] = None,
    max_retries: int = 3,
) -> SABRCalibrationResult:
    """
    Calibrate SABR parameters (alpha, rho, nu) to a smile slice.
    
    Args:
        strikes: Array of strike prices. At least four prices needs to be provided
        ivs: Array of market implied volatilities.
        T: Time to expiry in years.
        forward: The current forward price
        beta: CEV elasticity (fixed, default 0.5).
        initial_guess: Optional (alpha, rho, nu) initial guess.
        max_retries: Number of retries with perturbed initial guesses.
    
    Returns:
        SABRCalibrationResult: Contains success flag, fitted params, and metadata.
    """
    # Input validation
    if len(strikes) < 4:
        return SABRCalibrationResult(
            success=False,
            params=None,
            beta=beta,
            message="At least 4 strikes required for calibration.",
            fitted_ivs=None,
            strikes=strikes,
            ivs=ivs,
        )
    
    if T <= 0:
        return SABRCalibrationResult(
            success=False,
            params=None,
            beta=beta,
            message=f"Invalid T: {T} (must be > 0).",
            fitted_ivs=None,
            strikes=strikes,
            ivs=ivs,
        )

    # Set forward to the provided
    f = forward
    
    # Initial guess if none is provided
    if initial_guess is None:
        atm_idx = np.argmin(ivs)
        alpha_guess = ivs[atm_idx] * (f ** (1 - beta))  # ~ ATM vol * f^(1-beta)
        rho_guess = -0.3
        nu_guess = 0.5
        initial_guess = (alpha_guess, rho_guess, nu_guess)
    
    # Objective function
    def objective(params):
        alpha, rho, nu = params
        
        # Bounds check
        if alpha <= 0 or abs(rho) >= 0.99 or nu <= 0:
            return 1e10
        
        # Compute fitted IVs
        fitted_ivs = []
        for K in strikes:
            iv = sabr_iv(f, K, T, alpha, beta, rho, nu)
            if np.isnan(iv):
                return 1e10
            fitted_ivs.append(iv)
        
        fitted_ivs = np.array(fitted_ivs)

        # MSE
        error = np.sum((ivs - fitted_ivs) ** 2)
        return error
    
    # Optimizer with retry: Bounded within reasonable limits
    bounds = [
        (1e-6, None),   # alpha
        (-0.99, 0.99),  # rho
        (1e-6, None),   # nu
    ]
    
    best_result = None
    best_error = np.inf
    
    for attempt in range(max_retries):
        # Perturb initial guess for retries (deterministic seed)
        if attempt > 0:
            np.random.seed(42 + attempt)
            perturb = np.array([
                np.random.uniform(-0.1, 0.1),  # alpha
                np.random.uniform(-0.05, 0.05),  # rho
                np.random.uniform(-0.1, 0.1),  # nu
            ])
            x0 = np.array(initial_guess) + perturb * np.array(initial_guess)
            x0 = np.clip(x0, [1e-6, -0.99, 1e-6], [10.0, 0.99, 10.0])
        else:
            x0 = initial_guess
        
        # Try SLSQP
        try:
            result = minimize(
                objective,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-4}
            )
            if result.success and result.fun < best_error:
                best_error = result.fun
                best_result = result
        except Exception:
            pass
        
        # If SLSQP failed, try Nelder-Mead (robust but slower)
        if result is None or not result.success:
            try:
                result_nm = minimize(
                    objective,
                    x0=x0,
                    method='Nelder-Mead',
                    options={'maxiter': 1000}
                )
                if result_nm.success and result_nm.fun < best_error:
                    best_error = result_nm.fun
                    best_result = result_nm
            except Exception:
                pass
    
    # Check result
    if best_result is None or not best_result.success:
        return SABRCalibrationResult(
            success=False,
            params=None,
            beta=beta,
            message="All calibration attempts failed.",
            fitted_ivs=None,
            strikes=strikes,
            ivs=ivs,
        )
    
    alpha_opt, rho_opt, nu_opt = best_result.x
    
    # Compute fitted IVs for validation
    fitted_ivs = sabr_iv_vectorized(f, strikes, T, alpha_opt, beta, rho_opt, nu_opt)
    
    return SABRCalibrationResult(
        success=True,
        params=(alpha_opt, rho_opt, nu_opt),
        beta=beta,
        message=f"Calibration successful. Iterations: {best_result.nit}",
        fitted_ivs=fitted_ivs,
        strikes=strikes,
        ivs=ivs,
    )