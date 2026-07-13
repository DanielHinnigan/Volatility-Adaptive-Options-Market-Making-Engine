"""
Unit tests for RiskManager.

Uses a mock quoting engine to test risk limits in isolation.
"""

import pytest
from typing import Dict, List

from src.risk.risk_manager import RiskManager
from src.quoting.lucic_tse import InventoryGreeks, Quote


# ============================================================================
# Mock Quoting Engine
# ============================================================================

class MockQuotingEngine:
    """
    Mock for LucicTseQuotingEngine that returns predefined Greeks and quotes.

    This mock is used to test the Risk Manager in isolation.
    """

    def __init__(self, mock_greeks: InventoryGreeks, mock_quotes: Dict[str, Quote]):
        self.mock_greeks = mock_greeks
        self.mock_quotes = mock_quotes

        # Counters to verify that the Risk Manager calls these methods correctly.
        self.aggregate_inventory_calls = 0
        self.generate_quotes_calls = 0

    def aggregate_inventory(self, positions: Dict[str, int], option_specs: List[Dict]) -> InventoryGreeks:
        # 1. Increment the call counter (so tests can assert it was called).
        self.aggregate_inventory_calls += 1

        # 2. Return the fixed Greeks we set up in the test.
        return self.mock_greeks

    def generate_quotes(self, option_specs: List[Dict], positions: Dict[str, int]) -> Dict[str, Quote]:
        # 1. Increment the call counter.
        self.generate_quotes_calls += 1

        # 2. Return the fixed quotes we set up in the test.
        return self.mock_quotes

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_quote():
    return Quote(
        option_id="TEST_CALL",
        strike=100.0,
        expiry="2026-01-01",
        T=0.1,
        option_type="call",
        bid=9.8,
        ask=10.2,
        bid_size=5,
        ask_size=5,
        fair_value=10.0,
        spread=0.4,
    )


@pytest.fixture
def zero_greeks():
    return InventoryGreeks(
        delta=0.0,
        gamma=0.0,
        vega=0.0,
        theta=0.0,
        vega_by_tenor={},
    )


@pytest.fixture
def mock_engine(zero_greeks, sample_quote):
    quotes = {"TEST_CALL": sample_quote}
    return MockQuotingEngine(mock_greeks=zero_greeks, mock_quotes=quotes)


@pytest.fixture
def risk_manager(mock_engine):
    return RiskManager(
        quoting_engine=mock_engine,
        delta_limit=50000.0,
        gamma_limit=-100.0,
        theta_limit=-100.0,
        vega_limits={"0-7D": 500.0, "8-30D": 1500.0},
        drawdown_limit=-0.02,
        initial_capital=100_000.0,
    )


# ============================================================================
# Test Cases
# ============================================================================

class TestRiskManagerNoBreach:
    """Tests where no limits are breached."""

    def test_quotes_passed_through(self, risk_manager, mock_engine):
        option_specs = [{"id": "TEST_CALL"}]
        positions = {}

        result = risk_manager.get_quotes(option_specs, positions)

        assert result.halted is False
        assert result.quotes == mock_engine.mock_quotes
        assert mock_engine.aggregate_inventory_calls == 1
        assert mock_engine.generate_quotes_calls == 1


class TestRiskManagerHardBreaches:
    """Tests for individual hard limit breaches."""

    def test_delta_breach(self, risk_manager, mock_engine):
        # Set Delta to 60,000 (limit is 50,000)
        mock_engine.mock_greeks = InventoryGreeks(delta=60000.0)
        option_specs = [{"id": "TEST_CALL"}]
        positions = {}

        result = risk_manager.get_quotes(option_specs, positions)

        assert result.halted is True
        assert "Delta" in result.reason
        assert result.excess_delta == 10000.0
        assert mock_engine.generate_quotes_calls == 0  # Should not call if halted

    def test_gamma_breach(self, risk_manager, mock_engine):
        mock_engine.mock_greeks = InventoryGreeks(gamma=-150.0)
        option_specs = [{"id": "TEST_CALL"}]
        positions = {}

        result = risk_manager.get_quotes(option_specs, positions)

        assert result.halted is True
        assert "Gamma" in result.reason
        assert result.excess_gamma == 50.0
        assert mock_engine.generate_quotes_calls == 0

    def test_theta_breach(self, risk_manager, mock_engine):
        mock_engine.mock_greeks = InventoryGreeks(theta=-150.0)
        option_specs = [{"id": "TEST_CALL"}]
        positions = {}

        result = risk_manager.get_quotes(option_specs, positions)

        assert result.halted is True
        assert "Theta" in result.reason
        assert result.excess_theta == 50.0

    def test_vega_breach(self, risk_manager, mock_engine):
        mock_engine.mock_greeks = InventoryGreeks(
            vega_by_tenor={"0-7D": 600.0}
        )
        option_specs = [{"id": "TEST_CALL"}]
        positions = {}

        result = risk_manager.get_quotes(option_specs, positions)

        assert result.halted is True
        assert "Vega" in result.reason
        assert result.excess_vega["0-7D"] == 100.0
        assert mock_engine.generate_quotes_calls == 0


class TestRiskManagerMultipleBreaches:
    """Tests when multiple limits are breached simultaneously."""

    def test_delta_and_gamma_breach(self, risk_manager, mock_engine):
        mock_engine.mock_greeks = InventoryGreeks(
            delta=60000.0,
            gamma=-150.0,
        )
        option_specs = [{"id": "TEST_CALL"}]
        positions = {}

        result = risk_manager.get_quotes(option_specs, positions)

        assert result.halted is True
        assert "Delta" in result.reason
        assert "Gamma" in result.reason
        assert result.excess_delta == 10000.0
        assert result.excess_gamma == 50.0
        assert mock_engine.generate_quotes_calls == 0


class TestRiskManagerSoftLimits:
    """Tests for size reduction when risk is elevated (near limits)."""

    def test_size_reduction_on_elevated_delta(self, risk_manager, mock_engine, sample_quote):
        # 80% of limit = 40,000. Set Delta to 42,000 (elevated)
        mock_engine.mock_greeks = InventoryGreeks(delta=42000.0)
        mock_engine.mock_quotes = {"TEST_CALL": sample_quote}
        option_specs = [{"id": "TEST_CALL"}]
        positions = {}

        result = risk_manager.get_quotes(option_specs, positions)

        # Should not halt; Only sizes shouldbe halved
        assert result.halted is False
        quote = result.quotes["TEST_CALL"]

        # Sizes should be halved (5 -> 2)
        assert quote.bid_size == 2
        assert quote.ask_size == 2
        
        # Prices should remain unchanged
        assert quote.bid == 9.8
        assert quote.ask == 10.2

    def test_size_reduction_on_elevated_gamma(self, risk_manager, mock_engine, sample_quote):
        # 80% of limit = -80. Set Gamma to -90 (elevated)
        mock_engine.mock_greeks = InventoryGreeks(gamma=-90.0)
        mock_engine.mock_quotes = {"TEST_CALL": sample_quote}
        option_specs = [{"id": "TEST_CALL"}]
        positions = {}

        result = risk_manager.get_quotes(option_specs, positions)

        assert result.halted is False

        quote = result.quotes["TEST_CALL"]
        assert quote.bid_size == 2
        assert quote.ask_size == 2


class TestRiskManagerDrawdown:
    """Tests for PnL drawdown limits."""

    def test_drawdown_breach(self, risk_manager, mock_engine, sample_quote):
        option_specs = [{"id": "TEST_CALL"}]
        positions = {}

        # Drawdown limit = 100.000 USD * -2% = -2000 USD. Update PNL by -3000 USD < -2000 USD
        result = risk_manager.get_quotes(option_specs, positions, update_pnl=-3000.0)

        assert result.halted is True
        assert "Drawdown" in result.reason
        assert result.drawdown_excess == 1000.0  # -2000 limit, -3000 PnL → excess 1000
        assert risk_manager.is_halted() is True

    def test_drawdown_reset(self, risk_manager, mock_engine, sample_quote):
        # First, trigger a drawdown
        option_specs = [{"id": "TEST_CALL"}]
        positions = {}
        risk_manager.get_quotes(option_specs, positions, update_pnl=-3000.0)
        assert risk_manager.is_halted() is True

        # Reset daily PnL to emulate a new day
        risk_manager.reset_daily_pnl()
        assert risk_manager.is_halted() is False # New day: System is not longer halted.
        # Capital should be carried forward (100,000 - 3,000 = 97,000)
        assert risk_manager._day_start_capital == 97000.0


class TestRiskManagerLimitSetters:
    """Tests for runtime limit adjustments."""

    def test_set_delta_limit(self, risk_manager, mock_engine):
        risk_manager.set_delta_limit(100000.0)
        assert risk_manager.delta_limit == 100000.0

    def test_set_vega_limit_invalid_tenor(self, risk_manager):
        with pytest.raises(ValueError, match="Invalid tenor"):
            risk_manager.set_vega_limit("INVALID_TENOR", 100.0)

    def test_set_vega_limit_valid_tenor(self, risk_manager):
        risk_manager.set_vega_limit("0-7D", 1000.0)
        assert risk_manager.vega_limits["0-7D"] == 1000.0


class TestRiskManagerInitialization:
    """Tests for initialization with custom limits."""

    def test_invalid_vega_limits_raises_error(self, mock_engine):
        with pytest.raises(ValueError, match="Invalid tenor keys"):
            RiskManager(
                quoting_engine=mock_engine,
                vega_limits={"INVALID_TENOR": 100.0},
            )

    def test_initial_capital_preserved(self, mock_engine):
        rm = RiskManager(mock_engine, initial_capital=250000.0)
        assert rm.initial_capital == 250000.0
        assert rm._current_capital == 250000.0
        assert rm._day_start_capital == 250000.0