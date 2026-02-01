"""Tests for advanced CNN architectures."""

import pytest
import torch

from f1_predict.models.advanced_cnn import (
    AdvancedVisionEncoder,
    CustomF1CNN,
    MultiScaleFeatureExtractor,
    TemporalImageEncoder,
)


class TestAdvancedVisionEncoder:
    """Tests for AdvancedVisionEncoder class."""

    @pytest.fixture
    def encoder(self):
        """Create encoder instance."""
        return AdvancedVisionEncoder(
            backbone="resnet18",
            output_dim=256,
            pretrained=False,
        )

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 3, 224, 224)

    def test_init_resnet18(self):
        """Test initialization with ResNet18."""
        encoder = AdvancedVisionEncoder(backbone="resnet18", pretrained=False)
        assert encoder is not None

    def test_init_resnet50(self):
        """Test initialization with ResNet50."""
        encoder = AdvancedVisionEncoder(backbone="resnet50", pretrained=False)
        assert encoder is not None

    def test_init_efficientnet_b0(self):
        """Test initialization with EfficientNet-B0."""
        encoder = AdvancedVisionEncoder(backbone="efficientnet_b0", pretrained=False)
        assert encoder is not None

    def test_init_custom_output_dim(self):
        """Test initialization with custom output dimension."""
        encoder = AdvancedVisionEncoder(
            backbone="resnet18", output_dim=512, pretrained=False
        )
        assert encoder.output_dim == 512

    def test_forward_shape(self, encoder, sample_input):
        """Test forward pass output shape."""
        output = encoder(sample_input)
        assert output.shape == (2, 256)

    def test_forward_with_different_batch_sizes(self, encoder):
        """Test forward with various batch sizes."""
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 224, 224)
            output = encoder(x)
            assert output.shape == (batch_size, 256)

    def test_freeze_backbone(self):
        """Test backbone freezing."""
        encoder = AdvancedVisionEncoder(backbone="resnet18", pretrained=False)
        encoder.freeze_backbone()
        # Check that backbone parameters are frozen
        for param in encoder.backbone.parameters():
            assert not param.requires_grad

    def test_unfreeze_backbone(self):
        """Test backbone unfreezing."""
        encoder = AdvancedVisionEncoder(backbone="resnet18", pretrained=False)
        encoder.freeze_backbone()
        encoder.unfreeze_backbone()
        # Check that backbone parameters are trainable
        for param in encoder.backbone.parameters():
            assert param.requires_grad


class TestCustomF1CNN:
    """Tests for CustomF1CNN class."""

    @pytest.fixture
    def cnn(self):
        """Create CNN instance."""
        return CustomF1CNN(
            input_channels=3,
            output_dim=256,
        )

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 3, 224, 224)

    def test_init_default(self):
        """Test default initialization."""
        cnn = CustomF1CNN()
        assert cnn is not None

    def test_init_custom_channels(self):
        """Test custom input channels."""
        cnn = CustomF1CNN(input_channels=1)
        x = torch.randn(2, 1, 224, 224)
        output = cnn(x)
        assert output.dim() == 2

    def test_forward_shape(self, cnn, sample_input):
        """Test forward pass output shape."""
        output = cnn(sample_input)
        assert output.shape == (2, 256)

    def test_forward_gradient_flow(self, cnn, sample_input):
        """Test gradient flow through network."""
        output = cnn(sample_input)
        loss = output.sum()
        loss.backward()
        # Check gradients exist
        for param in cnn.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestMultiScaleFeatureExtractor:
    """Tests for MultiScaleFeatureExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return MultiScaleFeatureExtractor(
            backbone="resnet18",
            output_dim=256,
            pretrained=False,
        )

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 3, 224, 224)

    def test_init(self):
        """Test initialization."""
        extractor = MultiScaleFeatureExtractor(backbone="resnet18", pretrained=False)
        assert extractor is not None

    def test_forward_returns_dict(self, extractor, sample_input):
        """Test forward returns dictionary of features."""
        features = extractor(sample_input)
        assert isinstance(features, dict)

    def test_forward_multiple_scales(self, extractor, sample_input):
        """Test forward extracts multiple scales."""
        features = extractor(sample_input)
        # Should have multiple feature levels
        assert len(features) >= 1

    def test_get_fused_features(self, extractor, sample_input):
        """Test getting fused features."""
        fused = extractor.get_fused_features(sample_input)
        assert isinstance(fused, torch.Tensor)
        assert fused.shape[0] == 2


class TestTemporalImageEncoder:
    """Tests for TemporalImageEncoder class."""

    @pytest.fixture
    def encoder(self):
        """Create encoder instance."""
        return TemporalImageEncoder(
            cnn_output_dim=256,
            hidden_dim=128,
            num_layers=2,
            pretrained=False,
        )

    @pytest.fixture
    def sample_sequence(self):
        """Create sample image sequence."""
        # batch_size=2, sequence_length=5, channels=3, H=224, W=224
        return torch.randn(2, 5, 3, 224, 224)

    def test_init(self):
        """Test initialization."""
        encoder = TemporalImageEncoder(
            cnn_output_dim=128, hidden_dim=64, num_layers=1, pretrained=False
        )
        assert encoder is not None

    def test_forward_shape(self, encoder, sample_sequence):
        """Test forward pass output shape."""
        output = encoder(sample_sequence)
        assert output.shape == (2, 128)  # batch_size, hidden_dim

    def test_forward_different_sequence_lengths(self, encoder):
        """Test with different sequence lengths."""
        for seq_len in [1, 3, 10]:
            x = torch.randn(2, seq_len, 3, 224, 224)
            output = encoder(x)
            assert output.shape == (2, 128)

    def test_forward_single_batch(self, encoder):
        """Test with single batch item."""
        x = torch.randn(1, 5, 3, 224, 224)
        output = encoder(x)
        assert output.shape == (1, 128)


class TestBackboneCompatibility:
    """Tests for backbone compatibility."""

    @pytest.mark.parametrize("backbone", ["resnet18", "resnet50"])
    def test_supported_backbones(self, backbone):
        """Test all supported backbones work."""
        encoder = AdvancedVisionEncoder(
            backbone=backbone, output_dim=256, pretrained=False
        )
        x = torch.randn(2, 3, 224, 224)
        output = encoder(x)
        assert output.shape == (2, 256)

    def test_invalid_backbone_raises(self):
        """Test invalid backbone raises error."""
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            AdvancedVisionEncoder(backbone="invalid_backbone", pretrained=False)


class TestGradientFlow:
    """Tests for gradient flow through models."""

    def test_encoder_gradients(self):
        """Test gradients flow through encoder."""
        encoder = AdvancedVisionEncoder(
            backbone="resnet18", output_dim=256, pretrained=False
        )
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = encoder(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_custom_cnn_gradients(self):
        """Test gradients flow through custom CNN."""
        cnn = CustomF1CNN(output_dim=256)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = cnn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_temporal_encoder_gradients(self):
        """Test gradients flow through temporal encoder."""
        encoder = TemporalImageEncoder(
            cnn_output_dim=256, hidden_dim=128, pretrained=False
        )
        x = torch.randn(2, 3, 3, 224, 224, requires_grad=True)
        output = encoder(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
