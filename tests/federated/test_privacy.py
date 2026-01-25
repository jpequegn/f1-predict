"""Tests for differential privacy mechanisms."""

import numpy as np

from f1_predict.federated.privacy import (
    DifferentialPrivacy,
    NoiseType,
    PrivacyAccountant,
    PrivacyBudget,
    create_dp_mechanism,
)


class TestPrivacyBudget:
    """Test PrivacyBudget dataclass."""

    def test_create_budget(self):
        """Test creating a privacy budget."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        assert budget.epsilon == 1.0
        assert budget.delta == 1e-5
        assert budget.consumed_epsilon == 0.0
        assert budget.consumed_delta == 0.0

    def test_remaining_budget(self):
        """Test remaining budget calculation."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        assert budget.remaining_epsilon == 1.0

        budget.consumed_epsilon = 0.3
        assert budget.remaining_epsilon == 0.7

    def test_consume_budget(self):
        """Test consuming privacy budget."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)

        # Successful consumption
        assert budget.consume(0.5) is True
        assert budget.consumed_epsilon == 0.5

        # Another consumption
        assert budget.consume(0.3) is True
        assert budget.consumed_epsilon == 0.8

    def test_consume_exceeds_budget(self):
        """Test that consuming more than available fails."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        budget.consumed_epsilon = 0.9

        # Trying to consume more than remaining
        assert budget.consume(0.2) is False
        assert budget.consumed_epsilon == 0.9  # Unchanged

    def test_is_exhausted(self):
        """Test exhaustion detection."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        assert budget.is_exhausted is False

        budget.consumed_epsilon = 1.0
        assert budget.is_exhausted is True

    def test_reset_budget(self):
        """Test resetting consumed budget."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        budget.consumed_epsilon = 0.5
        budget.consumed_delta = 1e-6

        budget.reset()

        assert budget.consumed_epsilon == 0.0
        assert budget.consumed_delta == 0.0


class TestPrivacyAccountant:
    """Test PrivacyAccountant class."""

    def test_create_accountant(self):
        """Test creating a privacy accountant."""
        accountant = PrivacyAccountant(total_epsilon=1.0, total_delta=1e-5)
        assert accountant.budget.epsilon == 1.0
        assert len(accountant.history) == 0

    def test_record_query(self):
        """Test recording privacy queries."""
        accountant = PrivacyAccountant(total_epsilon=1.0)

        result = accountant.record_query(
            epsilon=0.2,
            description="Test query 1",
        )

        assert result is True
        assert len(accountant.history) == 1
        assert accountant.history[0]["epsilon"] == 0.2

    def test_record_multiple_queries(self):
        """Test recording multiple queries."""
        accountant = PrivacyAccountant(total_epsilon=1.0)

        accountant.record_query(0.2, description="Query 1")
        accountant.record_query(0.3, description="Query 2")

        assert len(accountant.history) == 2
        assert accountant.budget.consumed_epsilon == 0.5

    def test_query_exceeds_budget(self):
        """Test that queries exceeding budget are rejected."""
        accountant = PrivacyAccountant(total_epsilon=1.0)

        accountant.record_query(0.8)
        result = accountant.record_query(0.5)  # Would exceed budget

        assert result is False
        assert accountant.budget.consumed_epsilon == 0.8

    def test_get_remaining_budget(self):
        """Test getting remaining budget."""
        accountant = PrivacyAccountant(total_epsilon=1.0, total_delta=1e-5)
        accountant.record_query(0.4)

        remaining_eps, remaining_delta = accountant.get_remaining_budget()

        assert remaining_eps == 0.6
        assert remaining_delta == 1e-5

    def test_composition_bound(self):
        """Test composition bound calculation."""
        accountant = PrivacyAccountant(total_epsilon=10.0, total_delta=1e-5)

        bound = accountant.get_composition_bound(
            num_rounds=10,
            per_round_epsilon=0.5,
        )

        # Basic composition is 10 * 0.5 = 5.0
        # Advanced composition should be tighter
        assert bound <= 5.0
        assert bound > 0


class TestDifferentialPrivacy:
    """Test DifferentialPrivacy class."""

    def test_create_dp(self):
        """Test creating differential privacy mechanism."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        assert dp.epsilon == 1.0
        assert dp.delta == 1e-5
        assert dp.noise_type == NoiseType.GAUSSIAN

    def test_gaussian_noise(self):
        """Test Gaussian noise addition."""
        dp = DifferentialPrivacy(
            epsilon=1.0,
            noise_type=NoiseType.GAUSSIAN,
            random_state=42,
        )

        data = np.zeros((10, 5))
        noisy = dp.add_noise(data)

        # Should add noise (not all zeros anymore)
        assert not np.allclose(noisy, data)
        # Shape should be preserved
        assert noisy.shape == data.shape

    def test_laplacian_noise(self):
        """Test Laplacian noise addition."""
        dp = DifferentialPrivacy(
            epsilon=1.0,
            noise_type=NoiseType.LAPLACIAN,
            random_state=42,
        )

        data = np.zeros((10, 5))
        noisy = dp.add_noise(data)

        assert not np.allclose(noisy, data)
        assert noisy.shape == data.shape

    def test_noise_scale_with_epsilon(self):
        """Test that lower epsilon means more noise."""
        data = np.ones((100, 100))

        dp_low_eps = DifferentialPrivacy(epsilon=0.1, random_state=42)
        dp_high_eps = DifferentialPrivacy(epsilon=10.0, random_state=42)

        noisy_low = dp_low_eps.add_noise(data)
        noisy_high = dp_high_eps.add_noise(data)

        # Lower epsilon should add more noise (larger deviation from original)
        deviation_low = np.std(noisy_low - data)
        deviation_high = np.std(noisy_high - data)

        assert deviation_low > deviation_high

    def test_clip_gradient(self):
        """Test gradient clipping."""
        dp = DifferentialPrivacy(epsilon=1.0, clip_norm=1.0)

        # Large gradient
        large_gradient = np.array([10.0, 10.0, 10.0])
        clipped = dp.clip_gradient(large_gradient)

        # Should be clipped to norm 1.0
        assert np.linalg.norm(clipped) <= 1.0 + 1e-6

        # Small gradient should not be clipped
        small_gradient = np.array([0.1, 0.1, 0.1])
        not_clipped = dp.clip_gradient(small_gradient)
        np.testing.assert_array_almost_equal(not_clipped, small_gradient)

    def test_privatize_gradient(self):
        """Test full gradient privatization."""
        dp = DifferentialPrivacy(
            epsilon=1.0,
            clip_norm=1.0,
            random_state=42,
        )

        gradient = np.array([5.0, 5.0, 5.0])
        private = dp.privatize_gradient(gradient)

        # Should be different from original (noise added)
        assert not np.allclose(private, gradient)
        # Should have reasonable magnitude (clipped + noise)
        assert np.linalg.norm(private) < 10

    def test_privatize_model_update(self):
        """Test privatizing model weight updates."""
        dp = DifferentialPrivacy(epsilon=1.0, random_state=42)

        weights = {
            "layer1": np.ones((10, 5)),
            "layer2": np.ones((5, 3)),
        }

        private_weights = dp.privatize_model_update(weights)

        assert len(private_weights) == 2
        assert "layer1" in private_weights
        assert not np.allclose(private_weights["layer1"], weights["layer1"])

    def test_estimate_privacy_cost(self):
        """Test privacy cost estimation."""
        dp = DifferentialPrivacy(epsilon=1.0)

        cost = dp.estimate_privacy_cost(
            num_samples=1000,
            batch_size=32,
            epochs=10,
        )

        # Should return a positive epsilon
        assert cost > 0


class TestCreateDPMechanism:
    """Test factory function."""

    def test_create_gaussian(self):
        """Test creating Gaussian mechanism."""
        dp = create_dp_mechanism(
            epsilon=1.0,
            delta=1e-5,
            noise_type="gaussian",
        )

        assert dp.noise_type == NoiseType.GAUSSIAN

    def test_create_laplacian(self):
        """Test creating Laplacian mechanism."""
        dp = create_dp_mechanism(
            epsilon=1.0,
            noise_type="laplacian",
        )

        assert dp.noise_type == NoiseType.LAPLACIAN

    def test_create_with_sensitivity(self):
        """Test creating with custom sensitivity."""
        dp = create_dp_mechanism(
            epsilon=1.0,
            sensitivity=2.0,
        )

        assert dp.sensitivity == 2.0
