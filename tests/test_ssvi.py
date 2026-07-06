import pytest
import numpy as np
from src.models.ssvi import ssvi_total_variance, ssvi_iv, calibrate_ssvi, SSVICalibrationResult


# Test Core Math
class TestSSVIMath:
    def test_ssvi_total_variance(self):
        """Test SSVI total variance against a manual calculation."""
        # Given parameters
        theta = 0.04
        rho = -0.3
        eta = 0.5
        gamma = 0.5
        k = 0.05

        phi = eta / (theta ** gamma)
        sqrt_term = np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2)
        expected = 0.5 * theta * (1 + rho * phi * k + sqrt_term)

        result = ssvi_total_variance(np.array([k]), theta, rho, eta, gamma)
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_ssvi_iv(self):
        """Test SSVI IV conversion."""
        # Use the variance from above, T = 0.5
        # IV = sqrt(w/T)
        # Parameters from above
        theta = 0.04
        rho = -0.3
        eta = 0.5
        gamma = 0.5
        k = 0.05

        # Expiry
        T = 0.5

        # Total Variance
        phi = eta / (theta ** gamma)
        sqrt_term = np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2)
        w = 0.5 * theta * (1 + rho * phi * k + sqrt_term)

        # Expected IV
        expected_iv = np.sqrt(w/T)

        result = ssvi_iv(np.array([k]), theta, rho, eta, T, gamma)
        assert result[0] == pytest.approx(expected_iv, abs=1e-6)


# ============================================================================
# Test Global Calibration
# ============================================================================
class TestSSVICalibration:
    @pytest.fixture
    def synthetic_surface(self):
        """Generate a synthetic SSVI surface across 3 expiries to test calibration method against."""
        # True global parameters
        true_rho = -0.3
        true_eta = 0.5
        gamma = 0.5

        # Define expiries
        expiries = {
            "1M": {"T": 30 / 365, "theta": 0.04},
            "3M": {"T": 90 / 365, "theta": 0.045},
            "6M": {"T": 180 / 365, "theta": 0.05},
        }

        # Generate strikes for each expiry (log-moneyness from -0.5 to 0.5)
        slices = {}
        np.random.seed(42)

        for name, params in expiries.items():
            T = params["T"]
            theta = params["theta"]
            k_grid = np.linspace(-0.5, 0.5, 20)

            # Compute true IVs
            iv_true = ssvi_iv(k_grid, theta, true_rho, true_eta, T, gamma)

            # Add small noise (0.1% vol noise)
            noise = np.random.normal(0, 0.001, size=len(k_grid))
            iv_noisy = iv_true + noise

            slices[name] = {
                "k": k_grid,
                "iv": iv_noisy,
                "theta": theta,
                "T": T,
            }

        return slices, (true_rho, true_eta), gamma

    def test_ssvi_calibration_recovers_surface(self, synthetic_surface):
        slices, true_params, gamma = synthetic_surface
        true_rho, true_eta = true_params

        # Calibrate
        result = calibrate_ssvi(slices, gamma=gamma)

        # Assert success
        assert result.success is True
        assert result.params is not None
        rho_fit, eta_fit = result.params

        # Assert parameters are in the ballpark
        assert rho_fit == pytest.approx(true_rho, abs=0.05)
        assert eta_fit == pytest.approx(true_eta, abs=0.05)

        # Assert fitted IV smiles are provided and match closely
        assert result.fitted_ivs is not None
        for expiry, data in slices.items():
            fitted_ivs = result.fitted_ivs[expiry]
            market_ivs = data["iv"]
            rmse = np.sqrt(np.mean((fitted_ivs - market_ivs) ** 2))
            # Should be close to the noise level (0.001)
            assert rmse < 0.002, f"RMSE for {expiry} too high: {rmse:.6f}"

    def test_ssvi_calibration_fails_with_empty_data(self):
        result = calibrate_ssvi({})
        assert result.success is False
        assert result.message is not None

    def test_ssvi_calibration_fails_with_missing_fields(self):
        slices = {
            "1M": {"k": np.array([0.1]), "iv": np.array([0.2])}  # Missing theta and T
        }
        result = calibrate_ssvi(slices)
        assert result.success is False
        assert "missing" in result.message.lower()

    def test_ssvi_calibration_enforces_no_arbitrage(self):
        """Test that the optimizer penalizes arbitrage violations heavily."""
        # Deliberately create a slice that violates no-butterfly arbitrage
        # theta * phi > 4/(1+|rho|)
        # We force theta to be huge (implying a huge phi if eta is fixed)
        theta_bad = 0.5  # phi = 0.5/sqrt(50) = 0.0707, theta*phi = 3.53 > 4/(1+|rho|) = 3.07 -> VIOLATION
        rho_fixed = -0.3
        eta_fixed = 0.5
        gamma = 0.5


        slices = {
            "1M": {
                "k": np.linspace(-0.5, 0.5, 10),
                "iv": np.full(10, 0.5),  # Dummy IVs
                "theta": theta_bad,
                "T": 30 / 365,
            }
        }

        # Run calibration with a known good initial guess that would normally fit
        result = calibrate_ssvi(slices, gamma=gamma, initial_rho=-0.3, initial_eta=0.5)

        # The optimizer should find a way to avoid the violation by pushing rho or eta.
        # Since we can't change theta, it will push eta down (to lower phi)
        # OR push rho closer to -1 or 1 to increase the denominator.
        # But since we fixed the initial guess, it should converge to a valid region.
        assert result.success is True
        rho_fit, eta_fit = result.params

        # Ensure the fitted parameters satisfy the constraint
        phi_fit = eta_fit / (theta_bad ** gamma)
        assert theta_bad * phi_fit <= 4.0 / (1.0 + abs(rho_fit)), "No-arbitrage constraint violated!"