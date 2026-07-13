import pytest
import numpy as np
from src.models.bsm import black_scholes_call, black_scholes_put, implied_volatility

class TestBSM:
    def test_call_price(self):
        # Known case: S=100, K=100, T=1, r=0.05, q=0.02, σ=0.20 → C ≈ 9.23
        price = black_scholes_call(100, 100, 1.0, 0.05, 0.02, 0.20)
        assert price == pytest.approx(9.23, abs=0.01)
    
    def test_put_price(self):
        # Put-call parity check
        S, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.02, 0.20
        call = black_scholes_call(S, K, T, r, q, sigma)
        put = black_scholes_put(S, K, T, r, q, sigma)

        assert call - put == pytest.approx(S*np.exp(-q*T) - K * np.exp(-r * T), abs=0.01)
    
    def test_iv_inversion(self):
        S, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.02, 0.20
        price = black_scholes_call(S, K, T, r, q, sigma)
        iv = implied_volatility(price, S, K, T, r, q, "call")
        assert iv == pytest.approx(sigma, abs=1e-6)