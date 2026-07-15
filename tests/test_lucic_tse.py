"""
Unit tests for LucicTseQuotingEngine.

Uses a mock PricingEngine to avoid network calls and ensure deterministic behaviour.
"""

import pytest
import numpy as np
from unittest.mock import patch

from src.quoting.lucic_tse import (
    LucicTseQuotingEngine,
    InventoryGreeks,
)
from src.models.bsm import black_scholes_call, black_scholes_put
from src.data.option_spec import OptionSpec


# ============================================================================
# Mock PricingEngine for deterministic testing
# ============================================================================

class MockPricingEngine:
    """A mock PricingEngine that returns fixed IVs and prices for testing."""

    def __init__(self, spot=100.0, r=0.045, q=0.012):
        self.spot = spot
        self.r = r
        self.q = q
        # Define a simple IV smile: IV = 0.2 + 0.1 * (strike/forward - 1)^2
        # We'll use a fixed forward = spot for simplicity.
        self._iv_cache = {}

    def get_spot(self):
        return self.spot

    def get_iv(self, strike, expiry, use_sabr=True):
        # Compute a simple smile
        forward = self.spot  # simplified
        k = np.log(strike / forward)
        iv = 0.2 + 0.1 * k**2
        # Clamp to avoid negative
        iv = max(iv, 0.05)
        return iv

    def get_price(self, strike, expiry, option_type, use_sabr=True):
        iv = self.get_iv(strike, expiry, use_sabr)
        # We'll just use a fixed T=0.1 for mock price.
        T = 0.1
        if option_type == 'call':
            return black_scholes_call(self.spot, strike, T, self.r, self.q, iv)
        else:
            return black_scholes_put(self.spot, strike, T, self.r, self.q, iv)

    def get_atm_iv(self, expiry):
        return self.get_iv(self.spot, expiry)

    def get_forward(self, expiry):
        T = 0.1  # dummy
        return self.spot * np.exp((self.r - self.q) * T)

    def is_calibrated(self):
        return True


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_pricing_engine():
    return MockPricingEngine(spot=100.0)

@pytest.fixture
def risk_factors():
    return [
        {
            'name': 'Vega 0-7D',
            'alpha': 10.0,
            'membership': lambda spec: spec['T'] <= 7/365,
            'weight_key': 'vega',
        },
        {
            'name': 'Total Delta',
            'alpha': 1.0,
            'membership': lambda spec: True,
            'weight_key': 'delta',
        },
    ]

@pytest.fixture
def order_flow_params():
    return {
        'lambda0_a': 50 * 252,
        'lambda0_b': 50 * 252,
        'kappa_a': 0.75,
        'kappa_b': 0.75,
    }

@pytest.fixture
def option_specs():
    return [
        OptionSpec(**{'id': 'C_100', 'strike': 100, 'expiry': '2026-01-01', 'T': 30/365, 'option_type': 'call'}),
        OptionSpec(**{'id': 'C_105', 'strike': 105, 'expiry': '2026-01-01', 'T': 30/365, 'option_type': 'call'}),
        OptionSpec(**{'id': 'C_95', 'strike': 95, 'expiry': '2026-01-01', 'T': 30/365, 'option_type': 'call'}),
    ]

@pytest.fixture
def quoting_engine(mock_pricing_engine, risk_factors, order_flow_params, option_specs):
    engine = LucicTseQuotingEngine(
        pricing_engine=mock_pricing_engine,
        risk_factors=risk_factors,
        order_flow_params=order_flow_params,
        auto_update=False,  # we will manually control state
        horizon_hours=0.5,
        option_specs=option_specs,
        initial_realized_vol=0.18,
    )
    return engine


# ============================================================================
# Test Cases
# ============================================================================

class TestLucicTseInitialization:
    def test_initial_state(self, quoting_engine):
        assert quoting_engine.is_initialized() is False
        assert quoting_engine._N == len(quoting_engine._option_specs)

    def test_fallback_spreads(self, quoting_engine, option_specs):
        # When not initialized, generate_quotes should return fallback spreads (±2%)
        positions = {} # Required for .generate_quotes()
        quotes = quoting_engine.generate_quotes(option_specs, positions)
        assert len(quotes) == len(option_specs)
        for q in quotes.values():
            assert q.spread == pytest.approx(0.04 * q.fair_value, rel=0.01)


class TestLucicTseStateUpdate:
    def test_update_state_sets_initialized(self, quoting_engine, option_specs, mock_pricing_engine):
        spot = mock_pricing_engine.get_spot()
        quoting_engine.update_state(spot=spot, realized_vol_estimate=0.18, option_specs=option_specs)
        assert quoting_engine.is_initialized() is True
        assert quoting_engine._N == len(option_specs)

    def test_update_state_filters_invalid_options(self, quoting_engine, mock_pricing_engine):
        # Store the original method
        original_get_iv = mock_pricing_engine.get_iv

        def mock_get_iv_raises(strike, expiry, use_sabr=True):
            if strike == 105:
                raise RuntimeError("Pricing error")
            # For valid strikes, call the ORIGINAL method (not the patched one - causes infinite recursion)
            return original_get_iv(strike, expiry, use_sabr)

        with patch.object(mock_pricing_engine, 'get_iv', side_effect=mock_get_iv_raises):
            option_specs = [
                OptionSpec(**{'id': 'C_100', 'strike': 100, 'expiry': '2026-01-01', 'T': 30/365, 'option_type': 'call'}),
                OptionSpec(**{'id': 'C_105', 'strike': 105, 'expiry': '2026-01-01', 'T': 30/365, 'option_type': 'call'}),
            ]
            quoting_engine.update_state(
                spot=100,
                realized_vol_estimate=0.18,
                option_specs=option_specs
            )
            # Should have skipped strike 105, kept strike 100
            assert quoting_engine._N == 1
            assert quoting_engine._option_specs[0]['strike'] == 100
            assert quoting_engine.is_initialized() is True

class TestLucicTseQuotes:
    def test_generate_quotes_after_update(self, quoting_engine, option_specs, mock_pricing_engine):
        spot = mock_pricing_engine.get_spot()
        quoting_engine.update_state(spot=spot, realized_vol_estimate=0.18, option_specs=option_specs)
        positions = {} # Positions will only shift the bid and ask spreads, but will not influence whether fallback values are used or not
        quotes = quoting_engine.generate_quotes(option_specs, positions)
        assert len(quotes) == len(option_specs)
        # Check that spreads are not the fallback (should be different from 2%*fair)
        for q in quotes.values():
            assert q.spread != pytest.approx(0.04 * q.fair_value, rel=0.01)
            assert q.bid > 0
            assert q.ask > q.bid # Must sell (ask) for more than buy (bid)

    def test_quote_asymmetry_with_inventory(self, quoting_engine, option_specs, mock_pricing_engine):
        spot = mock_pricing_engine.get_spot()
        quoting_engine.update_state(spot=spot, realized_vol_estimate=0.18, option_specs=option_specs)

        # Baseline: zero inventory
        quotes_zero = quoting_engine.generate_quotes(option_specs, {})
        q0 = quotes_zero['C_100']
        mid0 = (q0.bid + q0.ask) / 2
        spread0 = q0.ask - q0.bid

        # Long 5 calls (positive inventory)
        quotes_long = quoting_engine.generate_quotes(option_specs, {'C_100': 5})
        q1 = quotes_long['C_100']
        mid1 = (q1.bid + q1.ask) / 2
        spread1 = q1.ask - q1.bid

        # Short 5 calls (negative inventory)
        quotes_short = quoting_engine.generate_quotes(option_specs, {'C_100': -5})
        q2 = quotes_short['C_100']
        mid2 = (q2.bid + q2.ask) / 2
        spread2 = q2.ask - q2.bid

        # ---- Assertions ----

        # 1. Midpoint shifts: long inventory lowers the midpoint, short raises it.
        assert mid1 < mid0, "Long inventory should lower the midpoint to encourage selling"
        assert mid2 > mid0, "Short inventory should raise the midpoint to encourage buying"

        # 2. The spread width should remain unchanged (inventory cancels out in Lucic-Tse model).
        # Allow a tiny numerical tolerance (1e-12).
        assert abs(spread1 - spread0) < 1e-12, "Spread should not change with inventory"
        assert abs(spread2 - spread0) < 1e-12, "Spread should not change with inventory"

        # 3. The shift should be roughly symmetric.
        # The midpoint shift for +5 and -5 should be opposite and similar magnitude.
        # We can check that the distance from zero is similar.
        shift_long = mid1 - mid0
        shift_short = mid2 - mid0
        assert abs(shift_long + shift_short) < 1e-12, "Midpoint shifts should be symmetric"

class TestLucicTseGreeks:
    def test_compute_single_option_greeks(self, quoting_engine, mock_pricing_engine):
        spot = 100.0
        strike = 100.0
        T = 0.5
        iv = 0.2
        greeks = quoting_engine._compute_single_option_greeks(
            spot=spot, strike=strike, T=T, option_type='call', iv=iv
        )
        # Approximate expected values for ATM call:
        # Delta ~ 0.5 (with small dividend/rate adjustment)
        # Gamma > 0
        # Vega > 0
        # Theta < 0
        assert greeks['delta'] == pytest.approx(0.5, abs=0.1)
        assert greeks['gamma'] > 0
        assert greeks['vega'] > 0
        assert greeks['theta'] < 0

    def test_aggregate_inventory(self, quoting_engine, option_specs, mock_pricing_engine):
        spot = mock_pricing_engine.get_spot()
        positions = {'C_100': 2, 'C_105': -1}
        
        greeks = quoting_engine.aggregate_inventory(positions, option_specs)
        assert isinstance(greeks, InventoryGreeks)

        # Delta should be roughly 2*delta_100 - delta_105
        # We can compute individual deltas separately and compare.
        delta_100 = quoting_engine._compute_single_option_greeks(
            spot=spot, strike=100, T=30/365, option_type='call', iv=mock_pricing_engine.get_iv(100, 'dummy')
        )['delta']

        delta_105 = quoting_engine._compute_single_option_greeks(
            spot=spot, strike=105, T=30/365, option_type='call', iv=mock_pricing_engine.get_iv(105, 'dummy')
        )['delta']
        
        expected_delta = 2 * delta_100 - delta_105
        assert greeks.delta == pytest.approx(expected_delta, abs=1e-6)
        
        # Check vega_by_tenor exists
        assert '8-30D' in greeks.vega_by_tenor


class TestLucicTseRiskMatrix:
    def test_build_risk_matrix_A_values(self, quoting_engine):
        """
        Deterministic test for _build_risk_matrix_A.
        We control the inputs and compute the expected A by hand.
        """
        # 1. Define a small, controlled set of options
        option_specs = [
            OptionSpec(**{'id': 'A', 'strike': 100, 'expiry': '2026-01-01', 'T': 0.5, 'option_type': 'call'}),
            OptionSpec(**{'id': 'B', 'strike': 105, 'expiry': '2026-01-01', 'T': 0.5, 'option_type': 'call'}),
        ]

        # 2. Define the corresponding greeks (we control these values - only vega should be used for calculations)
        greeks_list = [
            {'delta': 0.5, 'gamma': 0.02, 'vega': 2.0},  # Option A
            {'delta': 0.4, 'gamma': 0.03, 'vega': 3.0},  # Option B
        ]

        # 3. Define a single, simple risk factor
        #    - alpha = 2.0
        #    - membership: all options belong (lambda spec: True)
        #    - weight_key: 'vega'
        risk_factors = [
            {
                'name': 'Test Factor',
                'alpha': 2.0,
                'membership': lambda spec: True,  # All options belong
                'weight_key': 'vega',
            },
        ]

        # Temporarily override the engine's risk factors for this test
        original_risk_factors = quoting_engine.risk_factors
        quoting_engine.risk_factors = risk_factors

        try:
            # 4. Build the matrix
            A = quoting_engine._build_risk_matrix_A(option_specs, greeks_list)

            # 5. Compute the expected A by hand
            #    v = [vega_A, vega_B] = [2.0, 3.0]
            #    alpha = 2.0
            #    A = alpha * outer(v, v)*risk_aversion
            #    A[0,0] = 2.0 * 2.0 * 2.0 = 8.0
            #    A[0,1] = 2.0 * 2.0 * 3.0 = 12.0
            #    A[1,0] = 2.0 * 3.0 * 2.0 = 12.0
            #    A[1,1] = 2.0 * 3.0 * 3.0 = 18.0
            risk_aversion = quoting_engine.risk_aversion
            v = np.array([2.0, 3.0])
            expected_A = risk_aversion * risk_factors[0]["alpha"] * np.outer(v, v)
            # 6. Assert equality
            np.testing.assert_allclose(A, expected_A, rtol=1e-12)

        finally:
            # Restore the original risk factors
            quoting_engine.risk_factors = original_risk_factors


class TestLucicTseThetaIntegration:
    def test_compute_theta_1_integral_returns_nonzero(self, quoting_engine, option_specs, mock_pricing_engine):
        spot = mock_pricing_engine.get_spot()
        quoting_engine.update_state(spot=spot, realized_vol_estimate=0.18, option_specs=option_specs)

        T_horizon = quoting_engine.horizon_hours / (24 * 365) # In years as Lucic-Tse uses units of years
        theta_1 = quoting_engine._compute_theta_1_integral(T_horizon)

        # Theta_1 should not be all zeros (since C0 is non-zero due to realized vol != IV)
        assert np.any(np.abs(theta_1) > 1e-6)