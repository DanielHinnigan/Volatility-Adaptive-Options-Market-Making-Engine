import pytest
import numpy as np
from src.models.svi import svi_total_variance, svi_iv, calibrate_raw_svi, fit_svi_smile, SVICalibrationResult

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

# --- Test the public interface (fit_svi_smile) ---
class TestSVIPublicInterface:
    """Tests for the high-level SVI calibration interface."""
    @pytest.fixture
    def synthetic_smile_data(self):
        """Generate a clean synthetic smile with known parameters for public API testing."""
        # Known SVI parameters (a realistic equity skew)
        true_params = (0.04, 0.15, -0.3, 0.0, 0.1)
        T = 0.5
        forward = 100.0

        # Generate strikes from 80 to 120 (OTM strikes for calls > 100, puts < 100)
        # We want a nice mix of OTM puts and calls
        strike_grid = np.concatenate([
            np.linspace(80, 95, 10),   # OTM puts (strike < forward)
            np.linspace(105, 120, 10)  # OTM calls (strike > forward)
        ])
        strike_grid = np.sort(strike_grid)

        # Compute log-moneyness and IVs
        k_grid = np.log(strike_grid / forward)
        iv_grid = svi_iv(k_grid, *true_params, T)

        # Add a tiny amount of noise to simulate market micro-structure
        np.random.seed(42)
        noise = np.random.normal(0, 0.001, size=len(iv_grid))
        iv_grid = iv_grid + noise

        return strike_grid, iv_grid, T, forward, true_params

    def test_fit_svi_smile_success(self, synthetic_smile_data):
        """Test that the public interface successfully calibrates a known smile."""
        strikes, ivs, T, forward, true_params = synthetic_smile_data

        # Act
        result = fit_svi_smile(strikes, ivs, T, forward, r=0.05, q=0.0)

        # Assert: Success
        assert result.success is True
        assert result.params is not None
        assert len(result.params) == 5

        # Assert: Message is appropriate
        assert "successful" in result.message.lower()

        # Assert: Fitted IVs are computed correctly and match the input closely
        assert result.fitted_ivs is not None
        assert len(result.fitted_ivs) == len(ivs)

        # RMSE should be very small (fits the synthetic noise)
        rmse = np.sqrt(np.mean((result.fitted_ivs - ivs) ** 2))
        assert rmse < 0.002, f"RMSE too high: {rmse:.6f}"

        # Check that the result stores the input arrays correctly
        np.testing.assert_array_equal(result.strikes, strikes)
        np.testing.assert_array_equal(result.ivs, ivs)

    def test_fit_svi_smile_empty_arrays(self):
        """Test that empty arrays return a failure result, not an exception."""
        result = fit_svi_smile(
            strikes=np.array([]),
            ivs=np.array([]),
            T=0.5,
            forward=100.0,
            r=0.05,
            q=0.0
        )

        assert result.success is False
        assert result.params is None
        assert "empty" in result.message.lower()
        assert result.fitted_ivs is None

    def test_fit_svi_smile_mismatched_length(self):
        """Test that mismatched strike/IV lengths return a failure result."""
        strikes = np.array([100, 105, 110])
        ivs = np.array([0.2, 0.18])  # Only 2 IVs for 3 strikes

        result = fit_svi_smile(strikes, ivs, T=0.5, forward=100.0, r=0.05, q=0.0)

        assert result.success is False
        assert "same length" in result.message.lower()
        assert result.params is None

    def test_fit_svi_smile_invalid_t(self):
        """Test that T <= 0 returns a failure result."""
        strikes = np.array([100, 105])
        ivs = np.array([0.2, 0.18])

        # T = 0 (invalid)
        result = fit_svi_smile(strikes, ivs, T=0.0, forward=100.0, r=0.05, q=0.0)
        assert result.success is False

        # T = -1 (invalid)
        result = fit_svi_smile(strikes, ivs, T=-1.0, forward=100.0, r=0.05, q=0.0)
        assert result.success is False

    def test_fit_svi_smile_invalid_forward(self):
        """Test that forward <= 0 returns a failure result."""
        strikes = np.array([100, 105])
        ivs = np.array([0.2, 0.18])

        result = fit_svi_smile(strikes, ivs, T=0.5, forward=-10.0, r=0.05, q=0.0)
        assert result.success is False
        assert "forward" in result.message.lower()

    def test_fit_svi_smile_returns_correct_type(self):
        """Test that the function always returns an SVICalibrationResult object."""
        result = fit_svi_smile(np.array([100]), np.array([0.2]), T=0.5, forward=100.0, r=0.05, q=0.0)
        assert isinstance(result, SVICalibrationResult)

        # Even on failure (invalid forward), it should return the dataclass
        result = fit_svi_smile(np.array([100]), np.array([0.2]), T=0.5, forward=0.0, r=0.05, q=0.0)
        assert isinstance(result, SVICalibrationResult)

    def test_fit_svi_smile_handles_nans_gracefully(self):
        """Test that NaN values in input cause a clean failure, not a crash."""
        strikes = np.array([100, 105, 110])
        ivs = np.array([0.2, np.nan, 0.18])

        result = fit_svi_smile(strikes, ivs, T=0.5, forward=100.0, r=0.05, q=0.0)

        # The underlying optimizer will hit NaN, fail, and return None.
        # The public interface should catch this and return a failure result.
        assert result.success is False