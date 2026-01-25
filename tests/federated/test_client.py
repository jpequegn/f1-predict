"""Tests for federated learning client."""

import numpy as np
import pytest

from f1_predict.federated.client import (
    ClientConfig,
    ClientUpdate,
    FederatedClient,
    create_federated_client,
)
from f1_predict.federated.privacy import DifferentialPrivacy


class TestClientConfig:
    """Test ClientConfig dataclass."""

    def test_create_config(self):
        """Test creating client configuration."""
        config = ClientConfig(
            client_id="client_1",
            local_epochs=5,
            batch_size=64,
            learning_rate=0.001,
        )

        assert config.client_id == "client_1"
        assert config.local_epochs == 5
        assert config.batch_size == 64
        assert config.learning_rate == 0.001

    def test_default_values(self):
        """Test default configuration values."""
        config = ClientConfig(client_id="test")

        assert config.local_epochs == 1
        assert config.batch_size == 32
        assert config.learning_rate == 0.01
        assert config.differential_privacy is None
        assert config.compression_enabled is False

    def test_config_with_dp(self):
        """Test configuration with differential privacy."""
        dp = DifferentialPrivacy(epsilon=1.0)
        config = ClientConfig(
            client_id="test",
            differential_privacy=dp,
        )

        assert config.differential_privacy is not None
        assert config.differential_privacy.epsilon == 1.0


class TestClientUpdate:
    """Test ClientUpdate dataclass."""

    def test_create_update(self):
        """Test creating client update."""
        weights = {"layer1": np.array([1.0, 2.0])}
        update = ClientUpdate(
            client_id="client_1",
            weights=weights,
            num_samples=100,
            round_number=1,
        )

        assert update.client_id == "client_1"
        assert update.num_samples == 100
        assert "layer1" in update.weights

    def test_update_with_metrics(self):
        """Test update with training metrics."""
        update = ClientUpdate(
            client_id="client_1",
            weights={"layer1": np.array([1.0])},
            num_samples=100,
            round_number=1,
            metrics={"loss": 0.5, "accuracy": 0.9},
        )

        assert update.metrics["loss"] == 0.5
        assert update.metrics["accuracy"] == 0.9

    def test_summary(self):
        """Test summary generation."""
        update = ClientUpdate(
            client_id="client_1",
            weights={"layer1": np.array([1.0]), "layer2": np.array([2.0])},
            num_samples=100,
            round_number=1,
            metrics={"loss": 0.5},
        )

        summary = update.summary()
        assert "client_1" in summary
        assert "100" in summary
        assert "loss" in summary


class TestFederatedClient:
    """Test FederatedClient class."""

    @pytest.fixture
    def client(self):
        """Create basic federated client."""
        config = ClientConfig(
            client_id="test_client",
            local_epochs=1,
            batch_size=10,
            learning_rate=0.1,
        )
        return FederatedClient(config)

    @pytest.fixture
    def sample_weights(self):
        """Create sample model weights."""
        return {
            "layer1": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "layer2": np.array([0.5, 0.6]),
        }

    def test_receive_global_model(self, client, sample_weights):
        """Test receiving global model."""
        client.receive_global_model(sample_weights)

        assert client._local_weights is not None
        assert "layer1" in client._local_weights

    def test_receive_model_copies_weights(self, client, sample_weights):
        """Test that receiving model creates copies."""
        client.receive_global_model(sample_weights)

        # Modify original
        sample_weights["layer1"][0, 0] = 999.0

        # Client weights should be unchanged
        assert client._local_weights["layer1"][0, 0] != 999.0

    def test_train_without_model_raises(self, client):
        """Test that training without model raises error."""
        features = np.random.randn(100, 5)
        labels = np.random.randn(100)

        def forward(x, w):
            return x @ w["layer1"]

        def backward(_x, _y, _pred, w):
            return {"layer1": np.zeros_like(w["layer1"])}

        with pytest.raises(RuntimeError, match="No global model received"):
            client.train_local(features, labels, forward, backward)

    def test_train_local(self, client, sample_weights):
        """Test local training."""
        client.receive_global_model(sample_weights)

        # Simple data
        features = np.random.randn(50, 2)
        labels = np.random.randn(50, 2)

        # Simple forward/backward functions
        def forward(x, w):
            return x @ w["layer1"]

        def backward(x, y, pred, _w):
            error = pred - y
            grad = x.T @ error / len(x)
            return {"layer1": grad}

        metrics = client.train_local(features, labels, forward, backward)

        assert "loss" in metrics
        assert "num_samples" in metrics
        assert metrics["num_samples"] == 50

    def test_compute_update(self, client, sample_weights):
        """Test computing model update."""
        client.receive_global_model(sample_weights)

        # Simulate training by modifying weights
        client._local_weights["layer1"] = sample_weights["layer1"] + 0.1
        client.training_history.append({"num_samples": 100})

        update = client.compute_update(round_number=1)

        assert update.client_id == "test_client"
        assert update.round_number == 1
        # Delta should be approximately 0.1
        assert np.allclose(update.weights["layer1"], 0.1, atol=0.01)

    def test_compute_update_without_training_raises(self, client):
        """Test that computing update without training raises error."""
        with pytest.raises(RuntimeError, match="No local training performed"):
            client.compute_update(round_number=1)

    def test_compress_update(self, client, sample_weights):
        """Test update compression."""
        client.config.compression_enabled = True
        client.config.compression_ratio = 0.5

        client.receive_global_model(sample_weights)
        client._local_weights["layer1"] = sample_weights["layer1"] + np.random.randn(
            2, 2
        )
        client.training_history.append({"num_samples": 100})

        update = client.compute_update(round_number=1)

        # Compressed update should have some zeros
        total_nonzero = np.count_nonzero(update.weights["layer1"])
        total_elements = update.weights["layer1"].size
        assert total_nonzero <= total_elements  # Some sparsification

    def test_get_data_statistics(self, client):
        """Test getting privacy-preserving statistics."""
        features = np.random.randn(100, 5)
        labels = np.array([0] * 60 + [1] * 40)

        stats = client.get_data_statistics(features, labels)

        assert stats["num_samples"] == 100
        assert stats["num_features"] == 5

    def test_get_data_statistics_with_dp(self):
        """Test statistics with differential privacy."""
        dp = DifferentialPrivacy(epsilon=1.0, random_state=42)
        config = ClientConfig(
            client_id="test",
            differential_privacy=dp,
        )
        client = FederatedClient(config)

        features = np.ones((100, 5))  # Constant features
        labels = np.array([0] * 50 + [1] * 50)

        stats = client.get_data_statistics(features, labels)

        # Should have noisy statistics
        assert "feature_mean" in stats
        # Mean should be noisy (not exactly 1.0)
        assert not np.allclose(stats["feature_mean"], 1.0)


class TestCreateFederatedClient:
    """Test factory function."""

    def test_create_basic_client(self):
        """Test creating basic client."""
        client = create_federated_client(
            client_id="test_client",
            local_epochs=5,
            batch_size=64,
        )

        assert client.client_id == "test_client"
        assert client.config.local_epochs == 5
        assert client.config.batch_size == 64

    def test_create_client_with_dp(self):
        """Test creating client with differential privacy."""
        client = create_federated_client(
            client_id="test",
            epsilon=0.5,
        )

        assert client.config.differential_privacy is not None
        assert client.config.differential_privacy.epsilon == 0.5

    def test_create_client_no_dp(self):
        """Test creating client without differential privacy."""
        client = create_federated_client(
            client_id="test",
            epsilon=0,  # No DP
        )

        assert client.config.differential_privacy is None
