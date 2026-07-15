import pytest
import numpy as np
from src.models.sabr import sabr_iv, calibrate_sabr

class TestSABRMath:
    def test_sabr_iv_atm(self):
        """Test ATM IV formula."""
        # A simple case: f=100, K=100, T=1, alpha=0.2, beta=0.5, rho=-0.3, nu=0.5
        iv = sabr_iv(100, 100, 1.0, 0.2, 0.5, -0.3, 0.5)
        # Expected value is around 0.2 with small correction.
        # We just check it's positive and in a reasonable range.
        assert iv > 0.01
        assert iv < 0.5

    def test_sabr_iv_otm(self):
        """Test OTM IV."""
        f = 100
        K = 110  # OTM call
        iv = sabr_iv(f, K, 1.0, 0.2, 0.5, -0.3, 0.5)
        # OTM IV should be lower or higher depending on skew.
        # We just check it's valid.
        assert not np.isnan(iv)
        assert iv > 0.0

class TestSABRCalibration:
    def test_calibration_recovers_parameters(self):
        """Generate synthetic SABR smile and verify calibration."""
        np.random.seed(42)
        # True params
        true_alpha, true_beta, true_rho, true_nu = 0.2, 0.5, -0.3, 0.5
        T = 1.0
        f = 100.0
        strikes = np.linspace(70, 130, 20)
        
        # Generate IVs
        ivs = []
        for K in strikes:
            iv = sabr_iv(f, K, T, true_alpha, true_beta, true_rho, true_nu)
            ivs.append(iv)
        ivs = np.array(ivs)
        
        # Calibrate
        result = calibrate_sabr(strikes, ivs, T, f, beta=true_beta)
        assert result.success is True
        assert result.params is not None
        
        alpha, rho, nu = result.params
        # Parameters should be in the ballpark
        assert alpha == pytest.approx(true_alpha, abs=0.05)
        assert rho == pytest.approx(true_rho, abs=0.05)
        assert nu == pytest.approx(true_nu, abs=0.05)
        
        # Check RMSE
        fitted = result.fitted_ivs
        rmse = np.sqrt(np.mean((ivs - fitted) ** 2))
        assert rmse < 0.001  # Very tight fit

    def test_calibration_fails_gracefully(self):
        """Test that calibration fails with insufficient data."""
        result = calibrate_sabr(np.array([100, 101]), np.array([0.2, 0.21]), 1.0, 100)
        assert result.success is False
        assert "At least 4 strikes" in result.message