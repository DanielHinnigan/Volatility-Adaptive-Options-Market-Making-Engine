import pytest
import numpy as np
from src.models.svi import svi_total_variance, svi_iv, calibrate_raw_svi

# ---   Test the formula against hand-calculated values ---
class TestSVIMath:
    def test_svi_total_variance(self):
        # Given a known set of parameters
        a, b, rho, m, sigma = 0.04, 0.15, -0.3, 0.0, 0.1
        k = 0.05  # log-moneyness
        
        # Calculated using the correct equation
        expected = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        
        result = svi_total_variance(k, a, b, rho, m, sigma)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_svi_iv(self):
        # Using the result from above, T = 0.5
        # IV = sqrt(w / T)
        a, b, rho, m, sigma = 0.04, 0.15, -0.3, 0.0, 0.1
        k, T = 0.05, 0.5

        w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

        expected_iv = np.sqrt(w/T)
        
        result = svi_iv(k, a, b, rho, m, sigma, T)
        assert result == pytest.approx(expected_iv, abs=1e-6)


# ---  Test calibration with synthetic data ---
class TestSVICalibration:
    @pytest.fixture
    def synthetic_smile(self):
        """Generates a clean SVI smile with known parameters."""
        # True parameters
        true_params = (0.04, 0.15, -0.3, 0.0, 0.1)
        T = 0.5
        
        # Generate strikes from -0.5 to 0.5 log-moneyness
        k_grid = np.linspace(-0.5, 0.5, 30)
        iv_grid = svi_iv(k_grid, *true_params, T)
        
        # Add tiny noise to simulate market micro-structure (0.1% vol noise)
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, 0.001, size=len(iv_grid))
        iv_grid = iv_grid + noise
        
        return k_grid, iv_grid, T, true_params

    def test_calibration_recovers_smile(self, synthetic_smile):
        k_grid, iv_grid, T, true_params = synthetic_smile
        
        # Calibrate
        fitted_params = calibrate_raw_svi(k_grid, iv_grid, T)
        
        # Assert calibration didn't fail
        assert fitted_params is not None
        assert len(fitted_params) == 5
        
        # Verify that the fitted curve matches the synthetic data
        fitted_ivs = svi_iv(k_grid, *fitted_params, T)
        
        # The RMSE should be extremely small (close to the noise we added)
        rmse = np.sqrt(np.mean((fitted_ivs - iv_grid) ** 2))
        
        # Since noise was 0.001, RMSE should be less than ~0.002
        assert rmse < 0.002, f"RMSE too high: {rmse:.6f}"

    def test_calibration_fails_gracefully(self):
        # Empty arrays should return None
        result = calibrate_raw_svi(np.array([]), np.array([]), 0.5)
        assert result is None
        
        # NaNs should cause failure (or the optimizer to return None)
        k = np.array([0.1, 0.2, 0.3])
        iv = np.array([np.nan, 0.2, 0.3])
        result = calibrate_raw_svi(k, iv, 0.5)

        # The objective function will hit the NaN and return 1e10, optimizer fails
        assert result is None