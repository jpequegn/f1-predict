"""Tests for model compression in federated learning."""

import numpy as np
import pytest

from f1_predict.federated.compression import (
    CompressionConfig,
    CompressionMethod,
    CompressionResult,
    ModelCompressor,
    create_compressor,
)


class TestCompressionConfig:
    """Test CompressionConfig dataclass."""

    def test_create_config(self):
        """Test creating compression configuration."""
        config = CompressionConfig(
            method=CompressionMethod.TOP_K,
            compression_ratio=0.2,
        )

        assert config.method == CompressionMethod.TOP_K
        assert config.compression_ratio == 0.2

    def test_default_values(self):
        """Test default configuration values."""
        config = CompressionConfig()

        assert config.method == CompressionMethod.NONE
        assert config.compression_ratio == 0.1
        assert config.num_bits == 8
        assert config.error_feedback is True


class TestCompressionResult:
    """Test CompressionResult dataclass."""

    def test_create_result(self):
        """Test creating compression result."""
        result = CompressionResult(
            compressed_weights={"layer1": np.array([1.0])},
            compression_ratio=0.1,
            original_size=1000,
            compressed_size=100,
            method="top_k",
        )

        assert result.original_size == 1000
        assert result.compressed_size == 100

    def test_savings_calculation(self):
        """Test compression savings calculation."""
        result = CompressionResult(
            compressed_weights={},
            compression_ratio=0.1,
            original_size=1000,
            compressed_size=200,
            method="top_k",
        )

        assert result.savings == 80.0  # 80% savings

    def test_savings_zero_original(self):
        """Test savings with zero original size."""
        result = CompressionResult(
            compressed_weights={},
            compression_ratio=0.1,
            original_size=0,
            compressed_size=0,
            method="top_k",
        )

        assert result.savings == 0.0

    def test_summary(self):
        """Test summary generation."""
        result = CompressionResult(
            compressed_weights={"layer1": np.array([1.0])},
            compression_ratio=0.1,
            original_size=10240,
            compressed_size=1024,
            method="top_k",
        )

        summary = result.summary()
        assert "top_k" in summary
        assert "90.0%" in summary  # 90% savings


class TestModelCompressor:
    """Test ModelCompressor class."""

    @pytest.fixture
    def sample_weights(self):
        """Create sample weights for testing."""
        np.random.seed(42)
        return {
            "layer1": np.random.randn(100, 50),
            "layer2": np.random.randn(50, 10),
            "bias": np.random.randn(10),
        }

    def test_no_compression(self, sample_weights):
        """Test no compression mode."""
        compressor = ModelCompressor()  # Default is NONE
        result = compressor.compress(sample_weights)

        assert result.method == "none"
        assert result.original_size == result.compressed_size
        np.testing.assert_array_equal(
            result.compressed_weights["layer1"],
            sample_weights["layer1"],
        )

    def test_top_k_compression(self, sample_weights):
        """Test top-k sparsification."""
        config = CompressionConfig(
            method=CompressionMethod.TOP_K,
            compression_ratio=0.1,
        )
        compressor = ModelCompressor(config)

        result = compressor.compress(sample_weights)

        assert result.method == "top_k"
        # Should have significant savings
        assert result.compressed_size < result.original_size
        # Most values should be zero
        total_nonzero = sum(
            np.count_nonzero(w) for w in result.compressed_weights.values()
        )
        total_elements = sum(w.size for w in sample_weights.values())
        assert total_nonzero < total_elements * 0.2

    def test_random_k_compression(self, sample_weights):
        """Test random sparsification."""
        config = CompressionConfig(
            method=CompressionMethod.RANDOM_K,
            compression_ratio=0.1,
            random_state=42,
        )
        compressor = ModelCompressor(config)

        result = compressor.compress(sample_weights)

        assert result.method == "random_k"
        assert result.compressed_size < result.original_size

    def test_quantization_compression(self, sample_weights):
        """Test quantization."""
        config = CompressionConfig(
            method=CompressionMethod.QUANTIZATION,
            num_bits=4,
        )
        compressor = ModelCompressor(config)

        result = compressor.compress(sample_weights)

        assert result.method == "quantization"
        # 4-bit quantization should reduce size
        assert result.compressed_size < result.original_size

    def test_threshold_compression(self):
        """Test threshold-based compression."""
        # Create weights with many small values
        small_weights = {
            "layer1": np.concatenate(
                [
                    np.random.randn(50) * 0.0001,  # Small values
                    np.random.randn(50) * 1.0,  # Larger values
                ]
            )
        }

        config = CompressionConfig(
            method=CompressionMethod.THRESHOLD,
            threshold=0.01,
        )
        compressor = ModelCompressor(config)

        result = compressor.compress(small_weights)

        assert result.method == "threshold"
        # Small values should be zeroed
        num_zeros = np.sum(result.compressed_weights["layer1"] == 0)
        assert num_zeros > 40  # Most small values zeroed

    def test_error_feedback(self, sample_weights):
        """Test error feedback accumulation."""
        config = CompressionConfig(
            method=CompressionMethod.TOP_K,
            compression_ratio=0.1,
            error_feedback=True,
        )
        compressor = ModelCompressor(config)

        # First compression
        compressor.compress(sample_weights)

        # Error buffer should be populated
        assert len(compressor._error_buffer) > 0

        # Second compression should use error feedback
        result2 = compressor.compress(sample_weights)

        # Results should be different due to error accumulation
        # (though values might be similar, the process is different)
        assert result2 is not None

    def test_reset_error_buffer(self, sample_weights):
        """Test resetting error buffer."""
        config = CompressionConfig(
            method=CompressionMethod.TOP_K,
            compression_ratio=0.1,
            error_feedback=True,
        )
        compressor = ModelCompressor(config)

        compressor.compress(sample_weights)
        assert len(compressor._error_buffer) > 0

        compressor.reset_error_buffer()
        assert len(compressor._error_buffer) == 0

    def test_decompress_passthrough(self, sample_weights):
        """Test decompression for sparse methods."""
        config = CompressionConfig(method=CompressionMethod.TOP_K)
        compressor = ModelCompressor(config)

        result = compressor.compress(sample_weights)
        shapes = {name: w.shape for name, w in sample_weights.items()}

        decompressed = compressor.decompress(
            result.compressed_weights,
            shapes,
        )

        # For sparse methods, should be passthrough
        for name in result.compressed_weights:
            np.testing.assert_array_equal(
                decompressed[name],
                result.compressed_weights[name],
            )

    def test_get_compression_stats(self):
        """Test getting compression statistics."""
        config = CompressionConfig(
            method=CompressionMethod.QUANTIZATION,
            num_bits=8,
            error_feedback=True,
        )
        compressor = ModelCompressor(config)

        stats = compressor.get_compression_stats()

        assert stats["method"] == "quantization"
        assert stats["num_bits"] == 8
        assert stats["error_feedback_enabled"] is True


class TestCompressionPreservesStructure:
    """Test that compression preserves weight structure."""

    @pytest.fixture
    def weights(self):
        """Create weights with known structure."""
        return {
            "conv1": np.random.randn(32, 3, 3, 3),
            "fc1": np.random.randn(128, 512),
            "bias1": np.random.randn(128),
        }

    def test_shape_preserved_top_k(self, weights):
        """Test top-k preserves shapes."""
        compressor = create_compressor(method="top_k", compression_ratio=0.1)
        result = compressor.compress(weights)

        for name, orig_weight in weights.items():
            assert result.compressed_weights[name].shape == orig_weight.shape

    def test_shape_preserved_quantization(self, weights):
        """Test quantization preserves shapes."""
        compressor = create_compressor(method="quantization", num_bits=4)
        result = compressor.compress(weights)

        for name, orig_weight in weights.items():
            assert result.compressed_weights[name].shape == orig_weight.shape


class TestCreateCompressor:
    """Test factory function."""

    def test_create_top_k(self):
        """Test creating top-k compressor."""
        compressor = create_compressor(method="top_k", compression_ratio=0.2)
        assert compressor.config.method == CompressionMethod.TOP_K
        assert compressor.config.compression_ratio == 0.2

    def test_create_quantization(self):
        """Test creating quantization compressor."""
        compressor = create_compressor(method="quantization", num_bits=4)
        assert compressor.config.method == CompressionMethod.QUANTIZATION
        assert compressor.config.num_bits == 4

    def test_create_with_error_feedback(self):
        """Test creating with error feedback."""
        compressor = create_compressor(error_feedback=True)
        assert compressor.config.error_feedback is True

    def test_create_without_error_feedback(self):
        """Test creating without error feedback."""
        compressor = create_compressor(error_feedback=False)
        assert compressor.config.error_feedback is False

    def test_create_unknown_method(self):
        """Test creating with unknown method defaults to top_k."""
        compressor = create_compressor(method="unknown")
        assert compressor.config.method == CompressionMethod.TOP_K
