import pytest
import numpy as np
from typing import List, Dict

from src.data.base_connector import DataConnector, OptionQuote
from src.pricing_engine import PricingEngine


class MockConnector(DataConnector):
    """Mock data connector that returns synthetic OptionQuote objects."""

    def __init__(self, spot=100.0, expiries=None):
        self.spot = spot
        if expiries is None:
            expiries = ["2026-07-09", "2026-07-10", "2026-07-13"]
        self.expiries = expiries
        self._chains = self._generate_chains()

    def _generate_chains(self):
        chains = {}
        # Simple synthetic smile: IV = 0.2 + 0.1 * (K/F - 1)^2
        for exp in self.expiries:
            T = 7 / 365  # simplified
            forward = self.spot * np.exp(0.05 * T)
            calls = []
            puts = []
            for k in range(80, 121, 2):
                iv = 0.2 + 0.1 * ((k / forward) - 1) ** 2
                # Create a call and put for each strike
                mid = 10.0  # dummy price, we will compute IV from it later (or just set implied_vol)
                # In a real mock, we can just set implied_vol directly to avoid BSM inversion
                calls.append(OptionQuote(
                    strike=float(k),
                    bid=mid * 0.98,
                    ask=mid * 1.02,
                    mid=mid,
                    implied_vol=iv,
                    volume=100,
                    open_interest=100,
                    option_type="call"
                ))
                puts.append(OptionQuote(
                    strike=float(k),
                    bid=mid * 0.98,
                    ask=mid * 1.02,
                    mid=mid,
                    implied_vol=iv,
                    volume=100,
                    open_interest=100,
                    option_type="put"
                ))
            chains[exp] = {"calls": calls, "puts": puts}
        return chains

    def get_available_expiries(self) -> List[str]:
        return self.expiries

    def get_chain_for_expiry(self, expiry: str, use_cache: bool = True) -> Dict[str, List[OptionQuote]]:
        return self._chains[expiry]

    def get_spot_price(self) -> float:
        return self.spot

    def get_surface_data(self, max_expiries: int = 5) -> Dict:
        # Not used in engine, but required by base class
        return {}
    
class TestPricingEngine:
    """Unit tests for PricingEngine using a mock connector."""

    @pytest.fixture
    def mock_engine(self):
        connector = MockConnector(spot=100.0)
        engine = PricingEngine(symbol="TEST", connector=connector, r=0.05, q=0.01)
        return engine

    def test_initialization(self, mock_engine):
        assert mock_engine.symbol == "TEST"
        assert mock_engine.is_calibrated() is False

    def test_calibrate(self, mock_engine):
        mock_engine._calibrate()
        assert mock_engine.is_calibrated() is True
        assert "rho" in mock_engine.get_ssvi_params()
        assert len(mock_engine.get_svi_params()) == 3  # 3 expiries from mock
        assert len(mock_engine.get_sabr_params()) == 3

    def test_get_price(self, mock_engine):
        mock_engine._calibrate()
        expiry = mock_engine.get_svi_params().keys().__iter__().__next__()
        price = mock_engine.get_price(strike=100.0, expiry=expiry, option_type="call")
        assert isinstance(price, float)
        assert price > 0

    def test_get_atm_iv(self, mock_engine):
        mock_engine._calibrate()
        expiry = mock_engine.get_svi_params().keys().__iter__().__next__()
        atm_iv = mock_engine.get_atm_iv(expiry)
        assert isinstance(atm_iv, float)
        assert atm_iv > 0

    def test_background_thread(self, mock_engine):
        mock_engine.start_background_calibration(interval_ms=100)
        assert mock_engine.is_background_running() is True
        mock_engine.stop_background_calibration()
        assert mock_engine.is_background_running() is False
