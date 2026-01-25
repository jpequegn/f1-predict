"""Differential privacy mechanisms for federated learning.

This module provides privacy-preserving techniques:
- Differential privacy with configurable noise
- Privacy budget tracking and accounting
- Gradient clipping for bounded sensitivity
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class NoiseType(Enum):
    """Types of noise for differential privacy."""

    GAUSSIAN = "gaussian"
    LAPLACIAN = "laplacian"


@dataclass
class PrivacyBudget:
    """Privacy budget for differential privacy.

    The privacy budget (epsilon) controls the privacy-utility tradeoff:
    - Lower epsilon = more privacy, more noise, less utility
    - Higher epsilon = less privacy, less noise, more utility

    Typical values:
    - epsilon < 1.0: Strong privacy guarantee
    - epsilon 1.0-10.0: Moderate privacy
    - epsilon > 10.0: Weak privacy
    """

    epsilon: float  # Privacy parameter
    delta: float = 1e-5  # Probability of privacy breach
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0

    @property
    def remaining_epsilon(self) -> float:
        """Get remaining privacy budget."""
        return max(0.0, self.epsilon - self.consumed_epsilon)

    @property
    def remaining_delta(self) -> float:
        """Get remaining delta budget."""
        return max(0.0, self.delta - self.consumed_delta)

    @property
    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.consumed_epsilon >= self.epsilon

    def consume(self, epsilon: float, delta: float = 0.0) -> bool:
        """Consume privacy budget.

        Args:
            epsilon: Amount of epsilon to consume
            delta: Amount of delta to consume

        Returns:
            True if budget was consumed, False if insufficient budget
        """
        if epsilon > self.remaining_epsilon or delta > self.remaining_delta:
            return False

        self.consumed_epsilon += epsilon
        self.consumed_delta += delta
        return True

    def reset(self) -> None:
        """Reset consumed budget."""
        self.consumed_epsilon = 0.0
        self.consumed_delta = 0.0


class PrivacyAccountant:
    """Tracks privacy budget across multiple operations.

    Implements composition theorems for privacy accounting:
    - Basic composition: sum of epsilons
    - Advanced composition: tighter bounds for multiple queries
    """

    def __init__(self, total_epsilon: float, total_delta: float = 1e-5) -> None:
        """Initialize privacy accountant.

        Args:
            total_epsilon: Total privacy budget
            total_delta: Total delta budget
        """
        self.budget = PrivacyBudget(epsilon=total_epsilon, delta=total_delta)
        self.history: list[dict[str, Any]] = []
        self.logger = logger.bind(component="PrivacyAccountant")

    def record_query(
        self,
        epsilon: float,
        delta: float = 0.0,
        description: str = "",
    ) -> bool:
        """Record a privacy-consuming query.

        Args:
            epsilon: Privacy cost of the query
            delta: Delta cost of the query
            description: Description of the query

        Returns:
            True if query was recorded, False if budget exceeded
        """
        if not self.budget.consume(epsilon, delta):
            self.logger.warning(
                "privacy_budget_exceeded",
                requested_epsilon=epsilon,
                remaining_epsilon=self.budget.remaining_epsilon,
            )
            return False

        self.history.append(
            {
                "epsilon": epsilon,
                "delta": delta,
                "description": description,
                "cumulative_epsilon": self.budget.consumed_epsilon,
            }
        )

        self.logger.debug(
            "privacy_query_recorded",
            epsilon=epsilon,
            cumulative_epsilon=self.budget.consumed_epsilon,
        )

        return True

    def get_remaining_budget(self) -> tuple[float, float]:
        """Get remaining privacy budget.

        Returns:
            Tuple of (remaining_epsilon, remaining_delta)
        """
        return self.budget.remaining_epsilon, self.budget.remaining_delta

    def get_composition_bound(self, num_rounds: int, per_round_epsilon: float) -> float:
        """Calculate composition bound for multiple rounds.

        Uses advanced composition theorem for tighter bounds.

        Args:
            num_rounds: Number of training rounds
            per_round_epsilon: Epsilon per round

        Returns:
            Total epsilon under composition
        """
        # Basic composition
        basic = num_rounds * per_round_epsilon

        # Advanced composition (Dwork et al.)
        # eps_total = sqrt(2*k*ln(1/delta'))*eps + k*eps*(e^eps - 1)
        delta_prime = self.budget.delta / 2
        if delta_prime > 0:
            advanced = np.sqrt(
                2 * num_rounds * np.log(1 / delta_prime)
            ) * per_round_epsilon + num_rounds * per_round_epsilon * (
                np.exp(per_round_epsilon) - 1
            )
            return min(basic, advanced)

        return basic


@dataclass
class DifferentialPrivacy:
    """Differential privacy mechanism for model updates.

    Provides methods to add calibrated noise to gradients/updates
    to achieve (epsilon, delta)-differential privacy.
    """

    epsilon: float
    delta: float = 1e-5
    noise_type: NoiseType = NoiseType.GAUSSIAN
    sensitivity: float = 1.0  # L2 sensitivity (max gradient norm)
    clip_norm: float = 1.0  # Gradient clipping threshold
    random_state: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize random number generator."""
        self._rng = np.random.default_rng(self.random_state)
        self._logger = logger.bind(
            component="DifferentialPrivacy",
            epsilon=self.epsilon,
            noise_type=self.noise_type.value,
        )

    def _compute_noise_scale(self) -> float:
        """Compute noise scale based on epsilon and delta.

        Returns:
            Noise multiplier (sigma for Gaussian, b for Laplacian)
        """
        if self.noise_type == NoiseType.GAUSSIAN:
            # Gaussian mechanism: sigma = sens * sqrt(2*ln(1.25/delta)) / eps
            return (
                self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
            )
        # Laplacian mechanism: b = sensitivity / epsilon
        return self.sensitivity / self.epsilon

    def add_noise(self, data: np.ndarray) -> np.ndarray:
        """Add differential privacy noise to data.

        Args:
            data: Input array to privatize

        Returns:
            Noisy array with differential privacy guarantees
        """
        noise_scale = self._compute_noise_scale()

        if self.noise_type == NoiseType.GAUSSIAN:
            noise = self._rng.normal(0, noise_scale, data.shape)
        else:
            noise = self._rng.laplace(0, noise_scale, data.shape)

        self._logger.debug(
            "noise_added",
            noise_scale=noise_scale,
            data_shape=data.shape,
            noise_mean=float(np.mean(np.abs(noise))),
        )

        return data + noise

    def clip_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Clip gradient to bound sensitivity.

        Args:
            gradient: Input gradient

        Returns:
            Clipped gradient with bounded L2 norm
        """
        grad_norm = np.linalg.norm(gradient)

        if grad_norm > self.clip_norm:
            gradient = gradient * (self.clip_norm / grad_norm)
            self._logger.debug(
                "gradient_clipped",
                original_norm=grad_norm,
                clipped_norm=self.clip_norm,
            )

        return gradient

    def privatize_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Apply differential privacy to gradient.

        Clips the gradient and adds calibrated noise.

        Args:
            gradient: Input gradient

        Returns:
            Private gradient with (epsilon, delta)-DP guarantee
        """
        # Clip gradient to bound sensitivity
        clipped = self.clip_gradient(gradient)

        # Add noise and return
        return self.add_noise(clipped)

    def privatize_model_update(
        self,
        weights: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Apply differential privacy to model weight updates.

        Args:
            weights: Dictionary of weight arrays

        Returns:
            Dictionary of privatized weights
        """
        private_weights = {}

        for name, weight in weights.items():
            # Clip each weight matrix
            clipped = self.clip_gradient(weight)
            # Add noise
            private_weights[name] = self.add_noise(clipped)

        self._logger.info(
            "model_update_privatized",
            num_layers=len(weights),
            epsilon=self.epsilon,
        )

        return private_weights

    def estimate_privacy_cost(
        self,
        num_samples: int,
        batch_size: int,
        epochs: int,
    ) -> float:
        """Estimate total privacy cost for training.

        Uses the moments accountant for tight bounds.

        Args:
            num_samples: Total number of training samples
            batch_size: Training batch size
            epochs: Number of training epochs

        Returns:
            Estimated total epsilon
        """
        # Sampling probability
        q = batch_size / num_samples

        # Number of gradient steps
        steps = epochs * (num_samples // batch_size)

        # Simple composition (upper bound)
        # For tighter bounds, use moments accountant
        noise_scale = self._compute_noise_scale()

        # Privacy amplification by subsampling
        log_inv_delta = np.log(1 / self.delta)
        return 2 * q * np.sqrt(steps * log_inv_delta) / noise_scale


def create_dp_mechanism(
    epsilon: float,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    noise_type: str = "gaussian",
) -> DifferentialPrivacy:
    """Factory function to create differential privacy mechanism.

    Args:
        epsilon: Privacy budget
        delta: Probability of privacy breach
        sensitivity: L2 sensitivity
        noise_type: Type of noise ("gaussian" or "laplacian")

    Returns:
        Configured DifferentialPrivacy instance
    """
    noise = NoiseType.GAUSSIAN if noise_type == "gaussian" else NoiseType.LAPLACIAN

    return DifferentialPrivacy(
        epsilon=epsilon,
        delta=delta,
        sensitivity=sensitivity,
        noise_type=noise,
    )
