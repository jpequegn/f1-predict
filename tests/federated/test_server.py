"""Tests for federated learning server."""

import numpy as np
import pytest

from f1_predict.federated.client import ClientUpdate
from f1_predict.federated.server import (
    AggregationMethod,
    FederatedModel,
    FederatedServer,
    ServerConfig,
    TrainingRound,
    create_federated_server,
)


class TestServerConfig:
    """Test ServerConfig dataclass."""

    def test_create_config(self):
        """Test creating server configuration."""
        config = ServerConfig(
            min_clients=3,
            max_clients=50,
            rounds=20,
        )

        assert config.min_clients == 3
        assert config.max_clients == 50
        assert config.rounds == 20

    def test_default_values(self):
        """Test default configuration values."""
        config = ServerConfig()

        assert config.min_clients == 2
        assert config.max_clients == 100
        assert config.rounds == 10
        assert config.client_fraction == 1.0
        assert config.aggregation_method == AggregationMethod.FEDAVG


class TestTrainingRound:
    """Test TrainingRound dataclass."""

    def test_create_round(self):
        """Test creating training round."""
        round_info = TrainingRound(
            round_number=1,
            participating_clients=["client_1", "client_2"],
            metrics={"loss": 0.5},
        )

        assert round_info.round_number == 1
        assert len(round_info.participating_clients) == 2

    def test_summary(self):
        """Test summary generation."""
        round_info = TrainingRound(
            round_number=5,
            participating_clients=["c1", "c2", "c3"],
            metrics={"loss": 0.25, "accuracy": 0.9},
        )

        summary = round_info.summary()
        assert "Round 5" in summary
        assert "3" in summary  # num clients


class TestFederatedModel:
    """Test FederatedModel dataclass."""

    def test_create_model(self):
        """Test creating federated model."""
        weights = {"layer1": np.array([1.0, 2.0])}
        model = FederatedModel(weights=weights)

        assert model.version == 0
        assert model.training_rounds == 0
        assert "layer1" in model.weights

    def test_update_model(self):
        """Test updating model."""
        model = FederatedModel(weights={"layer1": np.array([1.0])})

        new_weights = {"layer1": np.array([2.0])}
        model.update(new_weights, samples=100)

        assert model.version == 1
        assert model.training_rounds == 1
        assert model.total_samples_seen == 100
        assert model.weights["layer1"][0] == 2.0


class TestFederatedServer:
    """Test FederatedServer class."""

    @pytest.fixture
    def server(self):
        """Create basic server."""
        config = ServerConfig(min_clients=2, max_clients=10)
        weights = {"layer1": np.zeros((5, 3)), "layer2": np.zeros(3)}
        return FederatedServer(config, initial_weights=weights)

    @pytest.fixture
    def sample_updates(self):
        """Create sample client updates."""
        return [
            ClientUpdate(
                client_id="client_1",
                weights={"layer1": np.ones((5, 3)) * 0.1, "layer2": np.ones(3) * 0.1},
                num_samples=100,
                round_number=1,
                metrics={"loss": 0.5},
            ),
            ClientUpdate(
                client_id="client_2",
                weights={"layer1": np.ones((5, 3)) * 0.2, "layer2": np.ones(3) * 0.2},
                num_samples=100,
                round_number=1,
                metrics={"loss": 0.4},
            ),
        ]

    def test_initialize_model(self):
        """Test model initialization."""
        config = ServerConfig()
        server = FederatedServer(config)

        weights = {"layer1": np.array([1.0, 2.0])}
        server.initialize_model(weights)

        assert server._model is not None
        assert server._model.version == 0

    def test_register_client(self, server):
        """Test client registration."""
        result = server.register_client("new_client")
        assert result is True
        assert "new_client" in server._registered_clients

    def test_register_max_clients(self):
        """Test max client limit."""
        config = ServerConfig(max_clients=2)
        server = FederatedServer(config)

        assert server.register_client("c1") is True
        assert server.register_client("c2") is True
        assert server.register_client("c3") is False  # Exceeds max

    def test_unregister_client(self, server):
        """Test client unregistration."""
        server.register_client("client_1")
        assert server.unregister_client("client_1") is True
        assert "client_1" not in server._registered_clients

    def test_unregister_nonexistent(self, server):
        """Test unregistering non-existent client."""
        assert server.unregister_client("nonexistent") is False

    def test_select_clients(self, server):
        """Test client selection."""
        server.register_client("c1")
        server.register_client("c2")
        server.register_client("c3")

        selected = server.select_clients()
        assert len(selected) >= server.config.min_clients

    def test_select_clients_insufficient(self, server):
        """Test selection with insufficient clients."""
        server.register_client("c1")  # Only one client
        selected = server.select_clients()
        assert len(selected) == 0  # Not enough clients

    def test_get_global_model(self, server):
        """Test getting global model."""
        model = server.get_global_model()
        assert model is not None
        assert "layer1" in model

    def test_get_global_model_uninitialized(self):
        """Test getting model when not initialized."""
        config = ServerConfig()
        server = FederatedServer(config)
        assert server.get_global_model() is None

    def test_start_round(self, server):
        """Test starting training round."""
        server.register_client("c1")
        server.register_client("c2")

        round_num, selected = server.start_round()

        assert round_num == 1
        assert len(selected) >= 2

    def test_start_round_without_model_raises(self):
        """Test starting round without model."""
        config = ServerConfig()
        server = FederatedServer(config)

        with pytest.raises(RuntimeError, match="Model not initialized"):
            server.start_round()

    def test_aggregate_updates(self, server, sample_updates):
        """Test aggregating client updates."""
        result = server.aggregate_updates(sample_updates)

        assert result.num_clients == 2
        assert result.total_samples == 200
        # Model should be updated
        assert server._model.version == 1

    def test_aggregate_empty_raises(self, server):
        """Test that empty updates raise error."""
        with pytest.raises(ValueError, match="No client updates"):
            server.aggregate_updates([])

    def test_aggregate_updates_model(self, server, sample_updates):
        """Test that aggregation updates global model."""
        initial_weights = server.get_global_model()

        server.aggregate_updates(sample_updates)

        updated_weights = server.get_global_model()
        # Weights should have changed (deltas applied)
        assert not np.allclose(
            initial_weights["layer1"],
            updated_weights["layer1"],
        )

    def test_training_history(self, server, sample_updates):
        """Test training history recording."""
        # Start a round to set _current_round to 1
        server.register_client("c1")
        server.register_client("c2")
        server.start_round()

        server.aggregate_updates(sample_updates)

        assert len(server.training_history) == 1
        assert server.training_history[0].round_number == 1

    def test_get_training_summary(self, server, sample_updates):
        """Test getting training summary."""
        server.register_client("c1")
        server.register_client("c2")
        server.aggregate_updates(sample_updates)

        summary = server.get_training_summary()

        assert summary["total_rounds"] == 1
        assert summary["total_samples_processed"] == 200
        assert summary["registered_clients"] == 2

    def test_training_summary_no_training(self, server):
        """Test summary when no training performed."""
        summary = server.get_training_summary()
        assert summary["status"] == "no_training_performed"


class TestFederatedServerWithPrivacy:
    """Test server with privacy budget."""

    @pytest.fixture
    def private_server(self):
        """Create server with privacy budget."""
        return create_federated_server(
            min_clients=2,
            rounds=10,
            privacy_budget=1.0,
            initial_weights={"layer1": np.zeros(5)},
        )

    def test_privacy_status(self, private_server):
        """Test getting privacy status."""
        status = private_server.get_privacy_status()

        assert status is not None
        assert status["total_epsilon"] == 1.0
        assert status["consumed_epsilon"] == 0.0
        assert status["is_exhausted"] is False

    def test_privacy_consumed_after_aggregation(self, private_server):
        """Test privacy consumption after aggregation."""
        private_server.register_client("c1")
        private_server.register_client("c2")

        updates = [
            ClientUpdate(
                client_id="c1",
                weights={"layer1": np.ones(5) * 0.1},
                num_samples=100,
                round_number=1,
                metrics={},
            ),
            ClientUpdate(
                client_id="c2",
                weights={"layer1": np.ones(5) * 0.1},
                num_samples=100,
                round_number=1,
                metrics={},
            ),
        ]

        private_server.aggregate_updates(updates)

        status = private_server.get_privacy_status()
        assert status["consumed_epsilon"] > 0


class TestCreateFederatedServer:
    """Test factory function."""

    def test_create_basic_server(self):
        """Test creating basic server."""
        server = create_federated_server(
            min_clients=3,
            max_clients=50,
            rounds=20,
        )

        assert server.config.min_clients == 3
        assert server.config.max_clients == 50
        assert server.config.rounds == 20

    def test_create_with_aggregation_method(self):
        """Test creating with different aggregation methods."""
        server = create_federated_server(aggregation_method="byzantine")
        assert server.config.aggregation_method == AggregationMethod.BYZANTINE

    def test_create_with_initial_weights(self):
        """Test creating with initial weights."""
        weights = {"layer1": np.array([1.0, 2.0])}
        server = create_federated_server(initial_weights=weights)

        model = server.get_global_model()
        assert model is not None
        np.testing.assert_array_equal(model["layer1"], weights["layer1"])

    def test_create_with_privacy_budget(self):
        """Test creating with privacy budget."""
        server = create_federated_server(
            rounds=10,
            privacy_budget=5.0,
        )

        status = server.get_privacy_status()
        assert status is not None
        assert status["total_epsilon"] == 5.0
