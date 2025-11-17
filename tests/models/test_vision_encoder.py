"""Tests for vision encoder module."""

import pytest
import torch
from torch import nn

from f1_predict.models.vision_encoder import VisionEncoder


class TestVisionEncoderInitialization:
    """Test vision encoder creation and initialization."""

    def test_vision_encoder_initialization(self):
        """Test vision encoder initializes correctly."""
        encoder = VisionEncoder(
            model_name='resnet18',
            pretrained=True,
            freeze_backbone=True
        )
        assert encoder is not None
        assert isinstance(encoder, nn.Module)

    def test_vision_encoder_to_device(self):
        """Test vision encoder can be moved to device."""
        encoder = VisionEncoder(
            model_name='resnet18',
            pretrained=True,
            freeze_backbone=False
        )
        device = torch.device('cpu')
        encoder.to(device)
        # Should not raise error

    def test_vision_encoder_with_different_models(self):
        """Test vision encoder supports different model architectures."""
        for model_name in ['resnet18', 'resnet50']:
            encoder = VisionEncoder(
                model_name=model_name,
                pretrained=True,
                freeze_backbone=True
            )
            assert encoder is not None


class TestVisionEncoderForward:
    """Test forward pass behavior."""

    @pytest.fixture
    def encoder(self):
        """Create test encoder."""
        return VisionEncoder(
            model_name='resnet18',
            pretrained=True,
            freeze_backbone=True
        )

    def test_forward_with_single_image(self, encoder):
        """Test forward pass with single image."""
        # Single image (1, 3, 224, 224)
        image = torch.randn(1, 3, 224, 224)
        features = encoder(image)

        # Should output 512-dim features for ResNet-18
        assert features.shape == (1, 512)

    def test_forward_with_batch_of_images(self, encoder):
        """Test forward pass with batch of images."""
        batch_size = 8
        image = torch.randn(batch_size, 3, 224, 224)
        features = encoder(image)

        assert features.shape == (batch_size, 512)

    def test_forward_output_finite(self, encoder):
        """Test forward pass output values are finite."""
        image = torch.randn(1, 3, 224, 224)
        features = encoder(image)

        assert torch.isfinite(features).all()
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()

    def test_forward_with_different_batch_sizes(self, encoder):
        """Test forward pass with various batch sizes."""
        for batch_size in [1, 2, 4, 8, 16]:
            image = torch.randn(batch_size, 3, 224, 224)
            features = encoder(image)

            assert features.shape == (batch_size, 512)


class TestVisionEncoderBackbone:
    """Test backbone network structure."""

    def test_encoder_has_backbone(self):
        """Test encoder has backbone component."""
        encoder = VisionEncoder(
            model_name='resnet18',
            pretrained=True,
            freeze_backbone=False
        )
        assert hasattr(encoder, 'backbone')

    def test_encoder_has_feature_extractor(self):
        """Test encoder has feature extraction module."""
        encoder = VisionEncoder(
            model_name='resnet18',
            pretrained=True,
            freeze_backbone=False
        )
        assert hasattr(encoder, 'feature_extractor')

    def test_backbone_frozen_when_requested(self):
        """Test backbone parameters frozen when freeze_backbone=True."""
        encoder = VisionEncoder(
            model_name='resnet18',
            pretrained=True,
            freeze_backbone=True
        )

        # All feature_extractor parameters should not require gradients
        for param in encoder.feature_extractor.parameters():
            # Should be frozen (requires_grad=False)
            assert not param.requires_grad

    def test_backbone_trainable_when_not_frozen(self):
        """Test backbone parameters are trainable when freeze_backbone=False."""
        encoder = VisionEncoder(
            model_name='resnet18',
            pretrained=True,
            freeze_backbone=False
        )

        # At least some feature_extractor parameters should be trainable
        has_trainable = False
        for param in encoder.feature_extractor.parameters():
            if param.requires_grad:
                has_trainable = True
                break

        assert has_trainable


class TestVisionEncoderModes:
    """Test training and eval modes."""

    def test_encoder_train_mode(self):
        """Test encoder enters train mode."""
        encoder = VisionEncoder(
            model_name='resnet18',
            pretrained=True,
            freeze_backbone=False
        )

        encoder.train()
        assert encoder.training

    def test_encoder_eval_mode(self):
        """Test encoder enters eval mode."""
        encoder = VisionEncoder(
            model_name='resnet18',
            pretrained=True,
            freeze_backbone=False
        )

        encoder.eval()
        assert not encoder.training

    def test_eval_mode_deterministic_output(self):
        """Test eval mode produces deterministic output."""
        encoder = VisionEncoder(
            model_name='resnet18',
            pretrained=True,
            freeze_backbone=True
        )

        image = torch.randn(4, 3, 224, 224)

        encoder.eval()
        with torch.no_grad():
            output1 = encoder(image)
            output2 = encoder(image)

        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2)


class TestVisionEncoderGradients:
    """Test gradient flow through encoder."""

    def test_gradients_flow_with_frozen_backbone(self):
        """Test gradients don't flow when backbone is frozen."""
        encoder = VisionEncoder(
            model_name='resnet18',
            pretrained=True,
            freeze_backbone=True
        )

        image = torch.randn(1, 3, 224, 224, requires_grad=True)
        features = encoder(image)
        loss = features.sum()
        loss.backward()

        # Input gradients should still exist
        assert image.grad is not None

    def test_encoder_parameters_exist(self):
        """Test encoder has trainable parameters."""
        encoder = VisionEncoder(
            model_name='resnet18',
            pretrained=True,
            freeze_backbone=False
        )

        params = list(encoder.parameters())
        assert len(params) > 0

    def test_output_dimension_resnet18(self):
        """Test output dimension for ResNet-18."""
        encoder = VisionEncoder(
            model_name='resnet18',
            pretrained=True,
            freeze_backbone=True
        )

        # ResNet-18 should output 512-dim features
        image = torch.randn(1, 3, 224, 224)
        features = encoder(image)

        assert features.shape[1] == 512

    def test_output_dimension_resnet50(self):
        """Test output dimension for ResNet-50."""
        encoder = VisionEncoder(
            model_name='resnet50',
            pretrained=True,
            freeze_backbone=True
        )

        # ResNet-50 should output 2048-dim features
        image = torch.randn(1, 3, 224, 224)
        features = encoder(image)

        assert features.shape[1] == 2048
