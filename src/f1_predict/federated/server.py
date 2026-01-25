"""Federated learning server implementation.

This module provides the server-side functionality:
- Client coordination and selection
- Model aggregation orchestration
- Training round management
- Global model distribution
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import structlog

from f1_predict.federated.aggregation import (
    AggregationResult,
    BaseAggregator,
    ByzantineRobustAggregator,
    FederatedAggregator,
    SecureAggregator,
)
from f1_predict.federated.client import ClientUpdate
from f1_predict.federated.privacy import DifferentialPrivacy, PrivacyAccountant

logger = structlog.get_logger(__name__)


class AggregationMethod(Enum):
    """Available aggregation methods."""

    FEDAVG = "fedavg"
    SECURE = "secure"
    BYZANTINE = "byzantine"


@dataclass
class ServerConfig:
    """Configuration for federated server."""

    min_clients: int = 2
    max_clients: int = 100
    rounds: int = 10
    client_fraction: float = 1.0  # Fraction of clients to sample per round
    aggregation_method: AggregationMethod = AggregationMethod.FEDAVG
    differential_privacy: Optional[DifferentialPrivacy] = None
    privacy_budget: Optional[float] = None  # Total privacy budget
    random_state: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingRound:
    """Information about a training round."""

    round_number: int
    participating_clients: list[str]
    aggregation_result: Optional[AggregationResult] = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate summary of training round."""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return (
            f"Training Round {self.round_number}:\n"
            f"  Clients: {len(self.participating_clients)}\n"
            f"  Metrics: {metrics_str}"
        )


@dataclass
class FederatedModel:
    """Container for federated model state."""

    weights: dict[str, np.ndarray]
    version: int = 0
    training_rounds: int = 0
    total_samples_seen: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        new_weights: dict[str, np.ndarray],
        samples: int,
    ) -> None:
        """Update model with new weights.

        Args:
            new_weights: New aggregated weights
            samples: Number of samples used in this round
        """
        self.weights = new_weights
        self.version += 1
        self.training_rounds += 1
        self.total_samples_seen += samples


class FederatedServer:
    """Federated learning server for coordinating distributed training.

    Handles:
    - Client registration and selection
    - Global model management
    - Training round orchestration
    - Secure aggregation coordination
    """

    def __init__(
        self,
        config: ServerConfig,
        initial_weights: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        """Initialize federated server.

        Args:
            config: Server configuration
            initial_weights: Initial global model weights
        """
        self.config = config
        self._rng = np.random.default_rng(config.random_state)
        self.logger = logger.bind(component="FederatedServer")

        # Initialize aggregator based on method
        self._aggregator = self._create_aggregator()

        # Privacy accounting
        self._privacy_accountant: Optional[PrivacyAccountant] = None
        if config.privacy_budget is not None:
            self._privacy_accountant = PrivacyAccountant(
                total_epsilon=config.privacy_budget
            )

        # Model state
        self._model: Optional[FederatedModel] = None
        if initial_weights is not None:
            self._model = FederatedModel(weights=initial_weights)

        # Registered clients
        self._registered_clients: set[str] = set()

        # Training history
        self.training_history: list[TrainingRound] = []

        # Current round
        self._current_round = 0

    def _create_aggregator(self) -> BaseAggregator:
        """Create aggregator based on configuration.

        Returns:
            Configured aggregator instance
        """
        dp = self.config.differential_privacy

        if self.config.aggregation_method == AggregationMethod.SECURE:
            return SecureAggregator(
                threshold=self.config.min_clients,
                differential_privacy=dp,
                random_state=self.config.random_state,
            )
        if self.config.aggregation_method == AggregationMethod.BYZANTINE:
            return ByzantineRobustAggregator(
                method="trimmed_mean",
                trim_ratio=0.1,
                differential_privacy=dp,
                random_state=self.config.random_state,
            )
        return FederatedAggregator(
            differential_privacy=dp,
            random_state=self.config.random_state,
        )

    def initialize_model(
        self,
        weights: dict[str, np.ndarray],
    ) -> None:
        """Initialize or reset global model.

        Args:
            weights: Initial model weights
        """
        self._model = FederatedModel(weights=weights.copy())
        self._current_round = 0
        self.training_history.clear()

        self.logger.info(
            "model_initialized",
            num_layers=len(weights),
        )

    def register_client(self, client_id: str) -> bool:
        """Register a client for federated training.

        Args:
            client_id: Unique client identifier

        Returns:
            True if registration successful
        """
        if len(self._registered_clients) >= self.config.max_clients:
            self.logger.warning(
                "client_registration_rejected",
                client_id=client_id,
                reason="max_clients_reached",
            )
            return False

        self._registered_clients.add(client_id)

        self.logger.info(
            "client_registered",
            client_id=client_id,
            total_clients=len(self._registered_clients),
        )

        return True

    def unregister_client(self, client_id: str) -> bool:
        """Unregister a client from federated training.

        Args:
            client_id: Client to unregister

        Returns:
            True if client was registered and removed
        """
        if client_id in self._registered_clients:
            self._registered_clients.remove(client_id)
            self.logger.info("client_unregistered", client_id=client_id)
            return True
        return False

    def select_clients(self) -> list[str]:
        """Select clients for current training round.

        Uses random sampling based on client_fraction.

        Returns:
            List of selected client IDs
        """
        available_clients = list(self._registered_clients)

        if len(available_clients) < self.config.min_clients:
            self.logger.warning(
                "insufficient_clients",
                available=len(available_clients),
                required=self.config.min_clients,
            )
            return []

        # Sample clients
        num_to_select = max(
            self.config.min_clients,
            int(len(available_clients) * self.config.client_fraction),
        )
        num_to_select = min(num_to_select, len(available_clients))

        selected = self._rng.choice(
            available_clients,
            size=num_to_select,
            replace=False,
        ).tolist()

        self.logger.info(
            "clients_selected",
            num_selected=len(selected),
            total_available=len(available_clients),
        )

        return selected

    def get_global_model(self) -> Optional[dict[str, np.ndarray]]:
        """Get current global model weights.

        Returns:
            Global model weights or None if not initialized
        """
        if self._model is None:
            return None
        return {name: w.copy() for name, w in self._model.weights.items()}

    def start_round(self) -> tuple[int, list[str]]:
        """Start a new training round.

        Returns:
            Tuple of (round_number, selected_clients)
        """
        if self._model is None:
            msg = "Model not initialized. Call initialize_model first."
            raise RuntimeError(msg)

        self._current_round += 1
        selected_clients = self.select_clients()

        self.logger.info(
            "round_started",
            round=self._current_round,
            num_clients=len(selected_clients),
        )

        return self._current_round, selected_clients

    def aggregate_updates(
        self,
        client_updates: list[ClientUpdate],
    ) -> AggregationResult:
        """Aggregate client updates and update global model.

        Args:
            client_updates: List of updates from clients

        Returns:
            Aggregation result
        """
        if not client_updates:
            msg = "No client updates to aggregate"
            raise ValueError(msg)

        if self._model is None:
            msg = "Model not initialized"
            raise RuntimeError(msg)

        self.logger.info(
            "aggregating_updates",
            num_updates=len(client_updates),
            round=self._current_round,
        )

        # Extract weights and sample counts
        weight_updates = [u.weights for u in client_updates]
        sample_counts = [float(u.num_samples) for u in client_updates]

        # Aggregate updates (deltas)
        result = self._aggregator.aggregate(
            client_updates=weight_updates,
            client_weights=sample_counts,
            round_number=self._current_round,
        )

        # Apply aggregated delta to global model
        new_weights = {}
        for name in self._model.weights:
            if name in result.aggregated_weights:
                new_weights[name] = (
                    self._model.weights[name] + result.aggregated_weights[name]
                )
            else:
                new_weights[name] = self._model.weights[name]

        # Update model
        self._model.update(new_weights, result.total_samples)

        # Record privacy cost
        if self._privacy_accountant is not None and self.config.differential_privacy:
            per_round_epsilon = self.config.differential_privacy.epsilon
            self._privacy_accountant.record_query(
                epsilon=per_round_epsilon,
                description=f"Round {self._current_round} aggregation",
            )

        # Compute aggregate metrics
        avg_metrics: dict[str, float] = {}
        for key in client_updates[0].metrics:
            values = [u.metrics.get(key, 0) for u in client_updates]
            avg_metrics[key] = float(np.mean(values))

        # Record training round
        training_round = TrainingRound(
            round_number=self._current_round,
            participating_clients=[u.client_id for u in client_updates],
            aggregation_result=result,
            metrics=avg_metrics,
        )
        self.training_history.append(training_round)

        self.logger.info(
            "round_complete",
            round=self._current_round,
            total_samples=result.total_samples,
            model_version=self._model.version,
        )

        return result

    def get_privacy_status(self) -> Optional[dict[str, Any]]:
        """Get current privacy budget status.

        Returns:
            Privacy status dictionary or None if no privacy accounting
        """
        if self._privacy_accountant is None:
            return None

        remaining_eps, remaining_delta = self._privacy_accountant.get_remaining_budget()

        return {
            "total_epsilon": self._privacy_accountant.budget.epsilon,
            "consumed_epsilon": self._privacy_accountant.budget.consumed_epsilon,
            "remaining_epsilon": remaining_eps,
            "remaining_delta": remaining_delta,
            "num_queries": len(self._privacy_accountant.history),
            "is_exhausted": self._privacy_accountant.budget.is_exhausted,
        }

    def get_training_summary(self) -> dict[str, Any]:
        """Get summary of federated training.

        Returns:
            Training summary dictionary
        """
        if not self.training_history:
            return {"status": "no_training_performed"}

        total_samples = sum(
            r.aggregation_result.total_samples
            for r in self.training_history
            if r.aggregation_result is not None
        )

        avg_clients_per_round = np.mean(
            [len(r.participating_clients) for r in self.training_history]
        )

        return {
            "total_rounds": len(self.training_history),
            "total_samples_processed": total_samples,
            "avg_clients_per_round": float(avg_clients_per_round),
            "registered_clients": len(self._registered_clients),
            "model_version": self._model.version if self._model else 0,
            "privacy_status": self.get_privacy_status(),
        }


def create_federated_server(
    min_clients: int = 2,
    max_clients: int = 100,
    rounds: int = 10,
    aggregation_method: str = "fedavg",
    privacy_budget: Optional[float] = None,
    initial_weights: Optional[dict[str, np.ndarray]] = None,
) -> FederatedServer:
    """Factory function to create a federated server.

    Args:
        min_clients: Minimum clients required per round
        max_clients: Maximum registered clients
        rounds: Total training rounds
        aggregation_method: Aggregation method ("fedavg", "secure", "byzantine")
        privacy_budget: Optional total privacy budget (epsilon)
        initial_weights: Optional initial model weights

    Returns:
        Configured FederatedServer instance
    """
    method_map = {
        "fedavg": AggregationMethod.FEDAVG,
        "secure": AggregationMethod.SECURE,
        "byzantine": AggregationMethod.BYZANTINE,
    }

    agg_method = method_map.get(aggregation_method, AggregationMethod.FEDAVG)

    dp = None
    if privacy_budget is not None:
        # Per-round epsilon for composition
        per_round_epsilon = privacy_budget / rounds
        dp = DifferentialPrivacy(epsilon=per_round_epsilon)

    config = ServerConfig(
        min_clients=min_clients,
        max_clients=max_clients,
        rounds=rounds,
        aggregation_method=agg_method,
        differential_privacy=dp,
        privacy_budget=privacy_budget,
    )

    return FederatedServer(config, initial_weights=initial_weights)
