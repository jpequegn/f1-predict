"""Federated learning client implementation.

This module provides the client-side functionality:
- Local model training
- Update generation and compression
- Privacy-preserving update submission
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import structlog

from f1_predict.federated.privacy import DifferentialPrivacy

logger = structlog.get_logger(__name__)


@dataclass
class ClientConfig:
    """Configuration for federated client."""

    client_id: str
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    differential_privacy: Optional[DifferentialPrivacy] = None
    compression_enabled: bool = False
    compression_ratio: float = 0.1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClientUpdate:
    """Model update from a federated client."""

    client_id: str
    weights: dict[str, np.ndarray]
    num_samples: int
    round_number: int
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate summary of client update."""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return (
            f"Client Update ({self.client_id}):\n"
            f"  Round: {self.round_number}\n"
            f"  Samples: {self.num_samples}\n"
            f"  Layers: {len(self.weights)}\n"
            f"  Metrics: {metrics_str}"
        )


class FederatedClient:
    """Federated learning client for local training.

    Handles:
    - Local model training on private data
    - Gradient/update computation
    - Privacy-preserving update generation
    - Communication with federated server
    """

    def __init__(
        self,
        config: ClientConfig,
        random_state: int = 42,
    ) -> None:
        """Initialize federated client.

        Args:
            config: Client configuration
            random_state: Random seed
        """
        self.config = config
        self.client_id = config.client_id
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.logger = logger.bind(
            component="FederatedClient",
            client_id=config.client_id,
        )

        # Local model weights (initialized when receiving global model)
        self._local_weights: Optional[dict[str, np.ndarray]] = None
        self._initial_weights: Optional[dict[str, np.ndarray]] = None

        # Training history
        self.training_history: list[dict[str, Any]] = []

    def receive_global_model(
        self,
        global_weights: dict[str, np.ndarray],
    ) -> None:
        """Receive global model from server.

        Args:
            global_weights: Current global model weights
        """
        # Deep copy weights
        self._local_weights = {
            name: weight.copy() for name, weight in global_weights.items()
        }
        self._initial_weights = {
            name: weight.copy() for name, weight in global_weights.items()
        }

        self.logger.info(
            "received_global_model",
            num_layers=len(global_weights),
        )

    def train_local(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        model_forward: Callable[[np.ndarray, dict[str, np.ndarray]], np.ndarray],
        model_backward: Callable[
            [np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]],
            dict[str, np.ndarray],
        ],
    ) -> dict[str, float]:
        """Train model locally on client data.

        This is a simplified training loop. In practice, this would
        integrate with the actual model training framework.

        Args:
            features: Local training features
            labels: Local training labels
            model_forward: Forward pass function
            model_backward: Backward pass function (returns gradients)

        Returns:
            Training metrics
        """
        if self._local_weights is None:
            msg = "No global model received. Call receive_global_model first."
            raise RuntimeError(msg)

        self.logger.info(
            "starting_local_training",
            num_samples=len(features),
            epochs=self.config.local_epochs,
        )

        num_samples = len(features)
        batch_size = min(self.config.batch_size, num_samples)

        total_loss = 0.0
        num_batches = 0

        for _epoch in range(self.config.local_epochs):
            # Shuffle data
            indices = self._rng.permutation(num_samples)
            features_shuffled = features[indices]
            labels_shuffled = labels[indices]

            epoch_loss = 0.0

            for i in range(0, num_samples, batch_size):
                batch_x = features_shuffled[i : i + batch_size]
                batch_y = labels_shuffled[i : i + batch_size]

                # Forward pass
                predictions = model_forward(batch_x, self._local_weights)

                # Compute loss (MSE for simplicity)
                loss = np.mean((predictions - batch_y) ** 2)
                epoch_loss += loss

                # Backward pass - get gradients
                gradients = model_backward(
                    batch_x, batch_y, predictions, self._local_weights
                )

                # Apply differential privacy to gradients
                if self.config.differential_privacy is not None:
                    gradients = {
                        name: self.config.differential_privacy.privatize_gradient(grad)
                        for name, grad in gradients.items()
                    }

                # Update weights (SGD)
                for name in self._local_weights:
                    if name in gradients:
                        self._local_weights[name] -= (
                            self.config.learning_rate * gradients[name]
                        )

                num_batches += 1

            total_loss += epoch_loss

        avg_loss = total_loss / max(num_batches, 1)

        metrics = {
            "loss": float(avg_loss),
            "num_samples": num_samples,
            "num_epochs": self.config.local_epochs,
        }

        self.training_history.append(metrics)

        self.logger.info(
            "local_training_complete",
            loss=avg_loss,
            num_batches=num_batches,
        )

        return metrics

    def compute_update(
        self,
        round_number: int,
    ) -> ClientUpdate:
        """Compute model update (delta from initial weights).

        Args:
            round_number: Current training round

        Returns:
            Client update containing weight deltas
        """
        if self._local_weights is None or self._initial_weights is None:
            msg = "No local training performed. Train first."
            raise RuntimeError(msg)

        # Compute weight deltas
        weight_deltas = {}
        for name in self._local_weights:
            delta = self._local_weights[name] - self._initial_weights[name]
            weight_deltas[name] = delta

        # Apply compression if enabled
        if self.config.compression_enabled:
            weight_deltas = self._compress_update(weight_deltas)

        # Get num_samples from last training
        num_samples = (
            self.training_history[-1]["num_samples"] if self.training_history else 0
        )

        update = ClientUpdate(
            client_id=self.client_id,
            weights=weight_deltas,
            num_samples=num_samples,
            round_number=round_number,
            metrics=self.training_history[-1] if self.training_history else {},
        )

        self.logger.info(
            "update_computed",
            round=round_number,
            num_layers=len(weight_deltas),
        )

        return update

    def _compress_update(
        self,
        weights: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Compress model update using top-k sparsification.

        Args:
            weights: Weight updates to compress

        Returns:
            Compressed (sparse) weight updates
        """
        compressed = {}

        for name, weight in weights.items():
            flat = weight.flatten()
            k = max(1, int(len(flat) * self.config.compression_ratio))

            # Keep top-k values by magnitude
            indices = np.argsort(np.abs(flat))[-k:]
            sparse = np.zeros_like(flat)
            sparse[indices] = flat[indices]

            compressed[name] = sparse.reshape(weight.shape)

        self.logger.debug(
            "update_compressed",
            compression_ratio=self.config.compression_ratio,
        )

        return compressed

    def get_data_statistics(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, Any]:
        """Get privacy-preserving statistics about local data.

        Uses differential privacy to protect individual data points.

        Args:
            features: Local feature data
            labels: Local label data

        Returns:
            Anonymized statistics about local dataset
        """
        stats: dict[str, Any] = {
            "num_samples": len(features),
            "num_features": features.shape[1] if len(features.shape) > 1 else 1,
        }

        # Add noisy statistics if DP is configured
        if self.config.differential_privacy is not None:
            dp = self.config.differential_privacy

            # Noisy mean
            noisy_mean = dp.add_noise(np.mean(features, axis=0))
            stats["feature_mean"] = noisy_mean.tolist()

            # Noisy label distribution
            max_categorical_labels = 10
            if len(np.unique(labels)) < max_categorical_labels:
                counts = np.bincount(labels.astype(int))
                noisy_counts = dp.add_noise(counts.astype(float))
                stats["label_distribution"] = np.maximum(0, noisy_counts).tolist()
        else:
            stats["feature_mean"] = np.mean(features, axis=0).tolist()

        return stats


def create_federated_client(
    client_id: str,
    epsilon: float = 1.0,
    local_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.01,
) -> FederatedClient:
    """Factory function to create a federated client.

    Args:
        client_id: Unique client identifier
        epsilon: Privacy budget for differential privacy
        local_epochs: Number of local training epochs
        batch_size: Local training batch size
        learning_rate: Local learning rate

    Returns:
        Configured FederatedClient instance
    """
    dp = DifferentialPrivacy(epsilon=epsilon) if epsilon > 0 else None

    config = ClientConfig(
        client_id=client_id,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        differential_privacy=dp,
    )

    return FederatedClient(config)
