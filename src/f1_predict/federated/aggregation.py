"""Aggregation protocols for federated learning.

This module provides various aggregation strategies:
- FedAvg (Federated Averaging)
- Secure aggregation with cryptographic protocols
- Byzantine-robust aggregation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import structlog

from f1_predict.federated.privacy import DifferentialPrivacy

logger = structlog.get_logger(__name__)


@dataclass
class AggregationResult:
    """Result of model aggregation."""

    aggregated_weights: dict[str, np.ndarray]
    num_clients: int
    total_samples: int
    round_number: int
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate summary of aggregation result."""
        return (
            f"Aggregation Result (Round {self.round_number}):\n"
            f"  Method: {self.method}\n"
            f"  Clients: {self.num_clients}\n"
            f"  Total samples: {self.total_samples}\n"
            f"  Layers aggregated: {len(self.aggregated_weights)}"
        )


class BaseAggregator(ABC):
    """Base class for federated aggregation."""

    def __init__(self, random_state: int = 42) -> None:
        """Initialize aggregator.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.logger = logger.bind(component=self.__class__.__name__)

    @abstractmethod
    def aggregate(
        self,
        client_updates: list[dict[str, np.ndarray]],
        client_weights: list[float],
        round_number: int,
    ) -> AggregationResult:
        """Aggregate model updates from clients.

        Args:
            client_updates: List of weight dictionaries from clients
            client_weights: Weights for each client (e.g., num_samples)
            round_number: Current training round

        Returns:
            Aggregated model weights
        """
        pass


class FederatedAggregator(BaseAggregator):
    """Standard Federated Averaging (FedAvg) aggregator.

    Implements the FedAvg algorithm from McMahan et al. (2017):
    - Weighted average of client models
    - Weights proportional to client data sizes
    """

    def __init__(
        self,
        differential_privacy: Optional[DifferentialPrivacy] = None,
        random_state: int = 42,
    ) -> None:
        """Initialize FedAvg aggregator.

        Args:
            differential_privacy: Optional DP mechanism
            random_state: Random seed
        """
        super().__init__(random_state)
        self.dp = differential_privacy

    def aggregate(
        self,
        client_updates: list[dict[str, np.ndarray]],
        client_weights: list[float],
        round_number: int,
    ) -> AggregationResult:
        """Aggregate using FedAvg.

        Args:
            client_updates: List of weight dictionaries from clients
            client_weights: Number of samples per client
            round_number: Current training round

        Returns:
            Aggregated weights
        """
        if not client_updates:
            msg = "No client updates to aggregate"
            raise ValueError(msg)

        self.logger.info(
            "aggregating_updates",
            num_clients=len(client_updates),
            round=round_number,
        )

        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        # Get all layer names from first client
        layer_names = list(client_updates[0].keys())

        # Aggregate each layer
        aggregated = {}
        for name in layer_names:
            # Weighted average across clients
            layer_sum = np.zeros_like(client_updates[0][name])

            for i, update in enumerate(client_updates):
                layer_sum += normalized_weights[i] * update[name]

            # Apply differential privacy if configured
            if self.dp is not None:
                layer_sum = self.dp.add_noise(layer_sum)

            aggregated[name] = layer_sum

        result = AggregationResult(
            aggregated_weights=aggregated,
            num_clients=len(client_updates),
            total_samples=int(sum(client_weights)),
            round_number=round_number,
            method="fedavg",
        )

        self.logger.info(
            "aggregation_complete",
            num_layers=len(aggregated),
            total_samples=result.total_samples,
        )

        return result


class SecureAggregator(BaseAggregator):
    """Secure aggregation using cryptographic protocols.

    Implements a simplified secure aggregation protocol:
    - Each client adds a random mask to their update
    - Masks cancel out when summed
    - Server only sees the aggregate, not individual updates
    """

    def __init__(
        self,
        threshold: int = 2,
        differential_privacy: Optional[DifferentialPrivacy] = None,
        random_state: int = 42,
    ) -> None:
        """Initialize secure aggregator.

        Args:
            threshold: Minimum clients needed for secure aggregation
            differential_privacy: Optional DP mechanism
            random_state: Random seed
        """
        super().__init__(random_state)
        self.threshold = threshold
        self.dp = differential_privacy
        self._client_masks: dict[int, dict[str, np.ndarray]] = {}

    def generate_masks(
        self,
        client_ids: list[int],
        weight_shapes: dict[str, tuple[int, ...]],
    ) -> dict[int, dict[str, np.ndarray]]:
        """Generate pairwise masks for secure aggregation.

        In a real implementation, this would use Shamir's secret sharing
        or similar cryptographic protocols.

        Args:
            client_ids: List of participating client IDs
            weight_shapes: Dictionary of layer shapes

        Returns:
            Dictionary mapping client ID to their masks
        """
        n_clients = len(client_ids)
        masks: dict[int, dict[str, np.ndarray]] = {cid: {} for cid in client_ids}

        for name, shape in weight_shapes.items():
            # Generate pairwise masks that sum to zero
            # For simplicity, using random masks
            # Real implementation would use cryptographic protocols
            all_masks = []

            for i in range(n_clients - 1):
                mask = self._rng.normal(0, 1, shape)
                all_masks.append(mask)
                masks[client_ids[i]][name] = mask

            # Last client gets negative sum of all other masks
            final_mask = -np.sum(all_masks, axis=0)
            masks[client_ids[-1]][name] = final_mask

        self._client_masks = masks
        return masks

    def apply_mask(
        self,
        client_id: int,
        weights: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Apply mask to client update.

        Args:
            client_id: Client identifier
            weights: Client's weight update

        Returns:
            Masked weights
        """
        if client_id not in self._client_masks:
            return weights

        masked = {}
        for name, weight in weights.items():
            if name in self._client_masks[client_id]:
                masked[name] = weight + self._client_masks[client_id][name]
            else:
                masked[name] = weight

        return masked

    def aggregate(
        self,
        client_updates: list[dict[str, np.ndarray]],
        client_weights: list[float],
        round_number: int,
    ) -> AggregationResult:
        """Aggregate using secure aggregation.

        Args:
            client_updates: List of (potentially masked) weight updates
            client_weights: Number of samples per client
            round_number: Current training round

        Returns:
            Securely aggregated weights
        """
        if len(client_updates) < self.threshold:
            msg = (
                f"Not enough clients for secure aggregation. "
                f"Need {self.threshold}, got {len(client_updates)}"
            )
            raise ValueError(msg)

        self.logger.info(
            "secure_aggregation_start",
            num_clients=len(client_updates),
            round=round_number,
        )

        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        # Get layer names
        layer_names = list(client_updates[0].keys())

        # Sum up all masked updates (masks cancel out)
        aggregated = {}
        for name in layer_names:
            layer_sum = np.zeros_like(client_updates[0][name])

            for i, update in enumerate(client_updates):
                layer_sum += normalized_weights[i] * update[name]

            # Apply differential privacy if configured
            if self.dp is not None:
                layer_sum = self.dp.add_noise(layer_sum)

            aggregated[name] = layer_sum

        # Clear masks after aggregation
        self._client_masks.clear()

        result = AggregationResult(
            aggregated_weights=aggregated,
            num_clients=len(client_updates),
            total_samples=int(sum(client_weights)),
            round_number=round_number,
            method="secure_aggregation",
            metadata={"threshold": self.threshold},
        )

        self.logger.info(
            "secure_aggregation_complete",
            num_layers=len(aggregated),
        )

        return result


class ByzantineRobustAggregator(BaseAggregator):
    """Byzantine-robust aggregation using coordinate-wise median.

    Robust to malicious or faulty clients that may send
    corrupted updates. Uses trimmed mean or coordinate-wise
    median to filter outliers.
    """

    def __init__(
        self,
        method: str = "trimmed_mean",
        trim_ratio: float = 0.1,
        differential_privacy: Optional[DifferentialPrivacy] = None,
        random_state: int = 42,
    ) -> None:
        """Initialize Byzantine-robust aggregator.

        Args:
            method: Aggregation method ("trimmed_mean" or "median")
            trim_ratio: Fraction of updates to trim (for trimmed_mean)
            differential_privacy: Optional DP mechanism
            random_state: Random seed
        """
        super().__init__(random_state)
        if method not in ["trimmed_mean", "median"]:
            msg = f"Unknown method: {method}. Use 'trimmed_mean' or 'median'"
            raise ValueError(msg)
        self.method = method
        self.trim_ratio = trim_ratio
        self.dp = differential_privacy

    def _trimmed_mean(
        self,
        values: np.ndarray,
        trim_ratio: float,
    ) -> np.ndarray:
        """Compute trimmed mean along first axis.

        Args:
            values: Array of shape (n_clients, ...)
            trim_ratio: Fraction to trim from each end

        Returns:
            Trimmed mean
        """
        n = len(values)
        k = int(n * trim_ratio)

        if k == 0:
            return np.mean(values, axis=0)

        # Sort along first axis
        sorted_vals = np.sort(values, axis=0)

        # Trim k values from each end
        trimmed = sorted_vals[k : n - k]

        return np.mean(trimmed, axis=0)

    def aggregate(
        self,
        client_updates: list[dict[str, np.ndarray]],
        client_weights: list[float],
        round_number: int,
    ) -> AggregationResult:
        """Aggregate using Byzantine-robust method.

        Args:
            client_updates: List of weight dictionaries from clients
            client_weights: Number of samples per client (used for logging)
            round_number: Current training round

        Returns:
            Robustly aggregated weights
        """
        min_byzantine_clients = 3
        if len(client_updates) < min_byzantine_clients:
            msg = f"Need at least {min_byzantine_clients} clients for Byzantine-robust"
            raise ValueError(msg)

        self.logger.info(
            "byzantine_aggregation_start",
            num_clients=len(client_updates),
            method=self.method,
            round=round_number,
        )

        layer_names = list(client_updates[0].keys())

        aggregated = {}
        for name in layer_names:
            # Stack updates from all clients
            stacked = np.stack([u[name] for u in client_updates])

            if self.method == "median":
                # Coordinate-wise median
                layer_agg = np.median(stacked, axis=0)
            else:
                # Trimmed mean
                layer_agg = self._trimmed_mean(stacked, self.trim_ratio)

            # Apply differential privacy if configured
            if self.dp is not None:
                layer_agg = self.dp.add_noise(layer_agg)

            aggregated[name] = layer_agg

        result = AggregationResult(
            aggregated_weights=aggregated,
            num_clients=len(client_updates),
            total_samples=int(sum(client_weights)),
            round_number=round_number,
            method=f"byzantine_{self.method}",
            metadata={
                "trim_ratio": self.trim_ratio if self.method == "trimmed_mean" else None
            },
        )

        self.logger.info(
            "byzantine_aggregation_complete",
            num_layers=len(aggregated),
        )

        return result
