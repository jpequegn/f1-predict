"""Tests for federated aggregation protocols."""

import numpy as np
import pytest

from f1_predict.federated.aggregation import (
    AggregationResult,
    ByzantineRobustAggregator,
    FederatedAggregator,
    SecureAggregator,
)
from f1_predict.federated.privacy import DifferentialPrivacy


class TestAggregationResult:
    """Test AggregationResult dataclass."""

    def test_create_result(self):
        """Test creating aggregation result."""
        result = AggregationResult(
            aggregated_weights={"layer1": np.array([1.0, 2.0])},
            num_clients=3,
            total_samples=300,
            round_number=1,
            method="fedavg",
        )

        assert result.num_clients == 3
        assert result.total_samples == 300
        assert result.round_number == 1
        assert result.method == "fedavg"

    def test_summary(self):
        """Test summary generation."""
        result = AggregationResult(
            aggregated_weights={"layer1": np.array([1.0])},
            num_clients=5,
            total_samples=500,
            round_number=2,
            method="fedavg",
        )

        summary = result.summary()
        assert "Round 2" in summary
        assert "5" in summary  # num_clients
        assert "500" in summary  # total_samples


class TestFederatedAggregator:
    """Test FederatedAggregator class."""

    @pytest.fixture
    def aggregator(self):
        """Create basic aggregator."""
        return FederatedAggregator()

    @pytest.fixture
    def sample_updates(self):
        """Create sample client updates."""
        return [
            {"layer1": np.array([1.0, 2.0]), "layer2": np.array([3.0])},
            {"layer1": np.array([2.0, 4.0]), "layer2": np.array([6.0])},
            {"layer1": np.array([3.0, 6.0]), "layer2": np.array([9.0])},
        ]

    def test_aggregate_equal_weights(self, aggregator, sample_updates):
        """Test aggregation with equal weights."""
        weights = [100, 100, 100]

        result = aggregator.aggregate(
            client_updates=sample_updates,
            client_weights=weights,
            round_number=1,
        )

        assert result.num_clients == 3
        assert result.method == "fedavg"
        # Average of [1,2], [2,4], [3,6] = [2,4]
        np.testing.assert_array_almost_equal(
            result.aggregated_weights["layer1"],
            np.array([2.0, 4.0]),
        )

    def test_aggregate_weighted(self, aggregator, sample_updates):
        """Test aggregation with different weights."""
        # First client has 3x the weight
        weights = [300, 100, 100]

        result = aggregator.aggregate(
            client_updates=sample_updates,
            client_weights=weights,
            round_number=1,
        )

        # Weighted average should favor first client
        # (300*1 + 100*2 + 100*3) / 500 = 1.6
        assert result.aggregated_weights["layer1"][0] < 2.0

    def test_aggregate_empty_raises(self, aggregator):
        """Test that empty updates raise error."""
        with pytest.raises(ValueError, match="No client updates"):
            aggregator.aggregate([], [], round_number=1)

    def test_aggregate_with_dp(self, sample_updates):
        """Test aggregation with differential privacy."""
        dp = DifferentialPrivacy(epsilon=1.0, random_state=42)
        aggregator = FederatedAggregator(differential_privacy=dp)

        weights = [100, 100, 100]
        result = aggregator.aggregate(
            client_updates=sample_updates,
            client_weights=weights,
            round_number=1,
        )

        # Should still produce valid result (with noise)
        assert result.num_clients == 3
        assert "layer1" in result.aggregated_weights

    def test_aggregate_preserves_layers(self, aggregator, sample_updates):
        """Test that all layers are preserved."""
        weights = [100, 100, 100]
        result = aggregator.aggregate(
            client_updates=sample_updates,
            client_weights=weights,
            round_number=1,
        )

        assert "layer1" in result.aggregated_weights
        assert "layer2" in result.aggregated_weights


class TestSecureAggregator:
    """Test SecureAggregator class."""

    @pytest.fixture
    def aggregator(self):
        """Create secure aggregator."""
        return SecureAggregator(threshold=2)

    @pytest.fixture
    def sample_updates(self):
        """Create sample updates."""
        return [
            {"layer1": np.array([1.0, 2.0])},
            {"layer1": np.array([2.0, 4.0])},
            {"layer1": np.array([3.0, 6.0])},
        ]

    def test_aggregate_secure(self, aggregator, sample_updates):
        """Test secure aggregation."""
        weights = [100, 100, 100]

        result = aggregator.aggregate(
            client_updates=sample_updates,
            client_weights=weights,
            round_number=1,
        )

        assert result.method == "secure_aggregation"
        assert result.num_clients == 3

    def test_aggregate_insufficient_clients(self, sample_updates):
        """Test error with insufficient clients."""
        aggregator = SecureAggregator(threshold=5)

        with pytest.raises(ValueError, match="Not enough clients"):
            aggregator.aggregate(
                client_updates=sample_updates,
                client_weights=[100, 100, 100],
                round_number=1,
            )

    def test_generate_masks(self, aggregator):
        """Test mask generation."""
        client_ids = [1, 2, 3]
        weight_shapes = {"layer1": (10, 5)}

        masks = aggregator.generate_masks(client_ids, weight_shapes)

        assert len(masks) == 3
        # Masks should sum to zero
        total_mask = sum(masks[cid]["layer1"] for cid in client_ids)
        np.testing.assert_array_almost_equal(total_mask, np.zeros((10, 5)))

    def test_apply_mask(self, aggregator):
        """Test applying mask to weights."""
        # Generate masks first
        client_ids = [1, 2]
        aggregator.generate_masks(client_ids, {"layer1": (2,)})

        weights = {"layer1": np.array([1.0, 2.0])}
        masked = aggregator.apply_mask(1, weights)

        # Masked values should be different
        assert "layer1" in masked


class TestByzantineRobustAggregator:
    """Test ByzantineRobustAggregator class."""

    @pytest.fixture
    def sample_updates_with_outlier(self):
        """Create updates with one malicious outlier."""
        return [
            {"layer1": np.array([1.0, 2.0])},
            {"layer1": np.array([1.1, 2.1])},
            {"layer1": np.array([0.9, 1.9])},
            {"layer1": np.array([1.0, 2.0])},
            {"layer1": np.array([100.0, 200.0])},  # Malicious outlier
        ]

    def test_median_aggregation(self, sample_updates_with_outlier):
        """Test median aggregation filters outliers."""
        aggregator = ByzantineRobustAggregator(method="median")

        result = aggregator.aggregate(
            client_updates=sample_updates_with_outlier,
            client_weights=[100] * 5,
            round_number=1,
        )

        # Median should ignore the outlier
        assert result.aggregated_weights["layer1"][0] < 5.0
        assert result.method == "byzantine_median"

    def test_trimmed_mean_aggregation(self, sample_updates_with_outlier):
        """Test trimmed mean aggregation."""
        aggregator = ByzantineRobustAggregator(
            method="trimmed_mean",
            trim_ratio=0.2,
        )

        result = aggregator.aggregate(
            client_updates=sample_updates_with_outlier,
            client_weights=[100] * 5,
            round_number=1,
        )

        # Trimmed mean should reduce outlier impact
        assert result.aggregated_weights["layer1"][0] < 10.0
        assert result.method == "byzantine_trimmed_mean"

    def test_insufficient_clients(self):
        """Test error with too few clients."""
        aggregator = ByzantineRobustAggregator()
        updates = [
            {"layer1": np.array([1.0])},
            {"layer1": np.array([2.0])},
        ]

        with pytest.raises(ValueError, match="at least 3 clients"):
            aggregator.aggregate(updates, [100, 100], round_number=1)

    def test_invalid_method(self):
        """Test error with invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            ByzantineRobustAggregator(method="invalid")

    def test_robust_against_multiple_outliers(self):
        """Test robustness against multiple outliers."""
        aggregator = ByzantineRobustAggregator(
            method="trimmed_mean",
            trim_ratio=0.3,
        )

        updates = [
            {"layer1": np.array([1.0])},
            {"layer1": np.array([1.0])},
            {"layer1": np.array([1.0])},
            {"layer1": np.array([1.0])},
            {"layer1": np.array([1.0])},
            {"layer1": np.array([100.0])},  # Outlier
            {"layer1": np.array([-100.0])},  # Outlier
        ]

        result = aggregator.aggregate(
            client_updates=updates,
            client_weights=[100] * 7,
            round_number=1,
        )

        # Should be close to 1.0 despite outliers
        assert abs(result.aggregated_weights["layer1"][0] - 1.0) < 1.0
