# Black-Scholes-Meron IV Inversion from market prices

import numpy as np
from scipy.stats import norm
from numba import jit

@jit(nopython=True, cache=True)
def norm_cdf(x) -> float:
    """Approximate normal CDF (Abramowitz & Stegun)."""
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    p = 0.2316419 

    if x>0:
        k= 1 / (1 + p * x)
        Z = 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)

        y = 1-Z*(a1*k+a2*k**2+a3*k**3+a4*k**4+a5*k**5)
    else:
        x *= -1
        k= 1 / (1 + p * x)
        Z = 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)

        y = 1-(1-Z*(a1*k+a2*k**2+a3*k**3+a4*k**4+a5*k**5))

    return  y

@jit(nopython=True, cache=True)
def norm_pdf(x) -> float:
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)

@jit(nopython=True, cache=True)
def black_scholes_call(S, K, T, r, q, sigma) -> float:
    """BSM Call Price."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)

@jit(nopython=True, cache=True)
def black_scholes_put(S, K, T, r, q, sigma) -> float:
    """BSM Put Price."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm_cdf(-d2) - S * np.exp(-q * T) * norm_cdf(-d1)

@jit(nopython=True, cache=True)
def implied_volatility(price, S, K, T, r, q, guess=0.2, max_iter=100, tol=1e-6):
    """
    Newton-Raphson inversion of BSM Call Price to Implied Volatility.
    Returns IV or NaN if inversion fails.
    Inversion fails if it becomes unstable (f''(x) is to small) or number of iterations is exceeded.
    The IV is bounded by 1e-6 and 5.
    """
    sigma = abs(guess)
    for i in range(max_iter):
        price_est = black_scholes_call(S, K, T, r, q, sigma)
        vega = S * np.exp(-q * T) * np.sqrt(T) * norm_pdf(
            (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        )
        diff = price_est - price
        
        if abs(diff) < tol:
            return sigma
        
        if vega < 1e-12:
            return np.nan  # Vega too small, inversion unstable
        
        sigma = sigma - diff / vega
        
        if sigma < 1e-6:
            sigma = 1e-6
        if sigma > 5.0:
            sigma = 5.0  # Cap at 500% IV
    
    return np.nan  # Max iterations reached