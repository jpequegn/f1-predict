"""Model update compression for communication-efficient federated learning.

This module provides compression techniques:
- Top-k sparsification
- Quantization
- Random sparsification
- Gradient accumulation for error feedback
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class CompressionMethod(Enum):
    """Available compression methods."""

    NONE = "none"
    TOP_K = "top_k"
    RANDOM_K = "random_k"
    QUANTIZATION = "quantization"
    THRESHOLD = "threshold"


@dataclass
class CompressionConfig:
    """Configuration for model compression."""

    method: CompressionMethod = CompressionMethod.NONE
    compression_ratio: float = 0.1  # Fraction of values to keep
    num_bits: int = 8  # For quantization
    threshold: float = 0.001  # For threshold compression
    error_feedback: bool = True  # Accumulate compression errors
    random_state: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionResult:
    """Result of compressing model updates."""

    compressed_weights: dict[str, np.ndarray]
    compression_ratio: float
    original_size: int  # In bytes
    compressed_size: int  # In bytes
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def savings(self) -> float:
        """Calculate compression savings as percentage."""
        if self.original_size == 0:
            return 0.0
        return (1 - self.compressed_size / self.original_size) * 100

    def summary(self) -> str:
        """Generate summary of compression result."""
        return (
            f"Compression Result:\n"
            f"  Method: {self.method}\n"
            f"  Original size: {self.original_size / 1024:.2f} KB\n"
            f"  Compressed size: {self.compressed_size / 1024:.2f} KB\n"
            f"  Savings: {self.savings:.1f}%"
        )


class ModelCompressor:
    """Compressor for federated learning model updates.

    Provides multiple compression strategies to reduce
    communication overhead in federated learning.
    """

    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
    ) -> None:
        """Initialize model compressor.

        Args:
            config: Compression configuration
        """
        self.config = config or CompressionConfig()
        self._rng = np.random.default_rng(self.config.random_state)
        self.logger = logger.bind(
            component="ModelCompressor",
            method=self.config.method.value,
        )

        # Error feedback buffer for accumulating compression errors
        self._error_buffer: dict[str, np.ndarray] = {}

    def compress(
        self,
        weights: dict[str, np.ndarray],
    ) -> CompressionResult:
        """Compress model weights/updates.

        Args:
            weights: Dictionary of weight arrays to compress

        Returns:
            Compression result with compressed weights
        """
        if self.config.method == CompressionMethod.NONE:
            return self._no_compression(weights)

        # Apply error feedback if enabled
        if self.config.error_feedback and self._error_buffer:
            weights = self._apply_error_feedback(weights)

        # Compress based on method
        if self.config.method == CompressionMethod.TOP_K:
            compressed, errors = self._top_k_compress(weights)
        elif self.config.method == CompressionMethod.RANDOM_K:
            compressed, errors = self._random_k_compress(weights)
        elif self.config.method == CompressionMethod.QUANTIZATION:
            compressed, errors = self._quantize(weights)
        elif self.config.method == CompressionMethod.THRESHOLD:
            compressed, errors = self._threshold_compress(weights)
        else:
            return self._no_compression(weights)

        # Store errors for next round
        if self.config.error_feedback:
            self._error_buffer = errors

        # Calculate sizes
        original_size = sum(w.nbytes for w in weights.values())
        compressed_size = self._estimate_compressed_size(compressed)

        result = CompressionResult(
            compressed_weights=compressed,
            compression_ratio=self.config.compression_ratio,
            original_size=original_size,
            compressed_size=compressed_size,
            method=self.config.method.value,
        )

        self.logger.info(
            "compression_complete",
            savings=f"{result.savings:.1f}%",
            method=self.config.method.value,
        )

        return result

    def decompress(
        self,
        compressed: dict[str, np.ndarray],
        original_shapes: dict[str, tuple[int, ...]],  # noqa: ARG002
    ) -> dict[str, np.ndarray]:
        """Decompress weights (for quantization).

        For sparse compressions, the values are already in original format.
        For quantization, this converts back to float.

        Args:
            compressed: Compressed weight dictionary
            original_shapes: Original shapes of weights (reserved for future use)

        Returns:
            Decompressed weights
        """
        if self.config.method == CompressionMethod.QUANTIZATION:
            return self._dequantize(compressed)

        # For sparse methods, weights are already in correct format
        return compressed

    def _no_compression(
        self,
        weights: dict[str, np.ndarray],
    ) -> CompressionResult:
        """Return weights without compression."""
        size = sum(w.nbytes for w in weights.values())
        return CompressionResult(
            compressed_weights=weights.copy(),
            compression_ratio=1.0,
            original_size=size,
            compressed_size=size,
            method="none",
        )

    def _top_k_compress(
        self,
        weights: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Compress using top-k sparsification.

        Keeps only the k largest values by magnitude.

        Args:
            weights: Weights to compress

        Returns:
            Tuple of (compressed_weights, compression_errors)
        """
        compressed = {}
        errors = {}

        for name, weight in weights.items():
            flat = weight.flatten()
            k = max(1, int(len(flat) * self.config.compression_ratio))

            # Get indices of top-k by magnitude
            top_k_indices = np.argsort(np.abs(flat))[-k:]

            # Create sparse representation
            sparse = np.zeros_like(flat)
            sparse[top_k_indices] = flat[top_k_indices]

            compressed[name] = sparse.reshape(weight.shape)
            errors[name] = weight - compressed[name]

        return compressed, errors

    def _random_k_compress(
        self,
        weights: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Compress using random sparsification.

        Randomly selects k values to keep.

        Args:
            weights: Weights to compress

        Returns:
            Tuple of (compressed_weights, compression_errors)
        """
        compressed = {}
        errors = {}

        for name, weight in weights.items():
            flat = weight.flatten()
            k = max(1, int(len(flat) * self.config.compression_ratio))

            # Random selection
            indices = self._rng.choice(len(flat), size=k, replace=False)

            # Create sparse representation with scaling
            sparse = np.zeros_like(flat)
            scale = 1.0 / self.config.compression_ratio
            sparse[indices] = flat[indices] * scale

            compressed[name] = sparse.reshape(weight.shape)
            errors[name] = weight - compressed[name] / scale

        return compressed, errors

    def _quantize(
        self,
        weights: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Compress using quantization.

        Reduces precision to num_bits.

        Args:
            weights: Weights to compress

        Returns:
            Tuple of (quantized_weights, quantization_errors)
        """
        compressed = {}
        errors = {}
        num_levels = 2**self.config.num_bits

        for name, weight in weights.items():
            # Get range for quantization
            min_val = weight.min()
            max_val = weight.max()
            range_val = max_val - min_val

            min_range_threshold = 1e-10
            if range_val < min_range_threshold:
                # Constant values
                compressed[name] = weight.copy()
                errors[name] = np.zeros_like(weight)
                continue

            # Quantize to integers
            normalized = (weight - min_val) / range_val
            quantized_int = np.round(normalized * (num_levels - 1)).astype(np.int32)

            # Store as float with metadata for reconstruction
            # In practice, you'd store integers + min/max separately
            dequantized = quantized_int / (num_levels - 1) * range_val + min_val

            compressed[name] = dequantized
            errors[name] = weight - dequantized

        return compressed, errors

    def _dequantize(
        self,
        quantized: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Dequantize weights back to float.

        Note: For this simplified implementation, quantized weights
        are already stored as floats, so this is a passthrough.

        Args:
            quantized: Quantized weight dictionary

        Returns:
            Dequantized weights
        """
        return quantized.copy()

    def _threshold_compress(
        self,
        weights: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Compress by zeroing values below threshold.

        Args:
            weights: Weights to compress

        Returns:
            Tuple of (compressed_weights, compression_errors)
        """
        compressed = {}
        errors = {}

        for name, weight in weights.items():
            mask = np.abs(weight) >= self.config.threshold
            sparse = weight * mask

            compressed[name] = sparse
            errors[name] = weight - sparse

        return compressed, errors

    def _apply_error_feedback(
        self,
        weights: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Apply accumulated compression errors.

        Args:
            weights: Current weights

        Returns:
            Weights with error feedback applied
        """
        corrected = {}

        for name, weight in weights.items():
            if name in self._error_buffer:
                corrected[name] = weight + self._error_buffer[name]
            else:
                corrected[name] = weight

        return corrected

    def _estimate_compressed_size(
        self,
        compressed: dict[str, np.ndarray],
    ) -> int:
        """Estimate size of compressed representation.

        Args:
            compressed: Compressed weights

        Returns:
            Estimated size in bytes
        """
        if self.config.method == CompressionMethod.QUANTIZATION:
            # Quantization reduces bits per value
            total_elements = sum(w.size for w in compressed.values())
            return total_elements * self.config.num_bits // 8

        if self.config.method in [
            CompressionMethod.TOP_K,
            CompressionMethod.RANDOM_K,
        ]:
            # Sparse: need to store indices + values
            total_elements = sum(w.size for w in compressed.values())
            k = int(total_elements * self.config.compression_ratio)
            # 4 bytes for index + 4 bytes for value (float32)
            return k * 8

        if self.config.method == CompressionMethod.THRESHOLD:
            # Count non-zero elements
            nnz = sum(np.count_nonzero(w) for w in compressed.values())
            return nnz * 8  # index + value

        return sum(w.nbytes for w in compressed.values())

    def reset_error_buffer(self) -> None:
        """Clear the error feedback buffer."""
        self._error_buffer.clear()

    def get_compression_stats(self) -> dict[str, Any]:
        """Get compression statistics.

        Returns:
            Statistics dictionary
        """
        is_quantization = self.config.method == CompressionMethod.QUANTIZATION
        return {
            "method": self.config.method.value,
            "compression_ratio": self.config.compression_ratio,
            "num_bits": self.config.num_bits if is_quantization else None,
            "error_feedback_enabled": self.config.error_feedback,
            "error_buffer_size": len(self._error_buffer),
        }


def create_compressor(
    method: str = "top_k",
    compression_ratio: float = 0.1,
    num_bits: int = 8,
    error_feedback: bool = True,
) -> ModelCompressor:
    """Factory function to create a model compressor.

    Args:
        method: Compression method (none, top_k, random_k, quantization, threshold)
        compression_ratio: Fraction of values to keep (for sparse methods)
        num_bits: Number of bits for quantization
        error_feedback: Whether to accumulate compression errors

    Returns:
        Configured ModelCompressor instance
    """
    method_map = {
        "none": CompressionMethod.NONE,
        "top_k": CompressionMethod.TOP_K,
        "random_k": CompressionMethod.RANDOM_K,
        "quantization": CompressionMethod.QUANTIZATION,
        "threshold": CompressionMethod.THRESHOLD,
    }

    comp_method = method_map.get(method, CompressionMethod.TOP_K)

    config = CompressionConfig(
        method=comp_method,
        compression_ratio=compression_ratio,
        num_bits=num_bits,
        error_feedback=error_feedback,
    )

    return ModelCompressor(config)
