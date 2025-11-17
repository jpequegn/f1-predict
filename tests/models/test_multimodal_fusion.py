"""Tests for multi-modal fusion model."""

import pytest
import torch
import torch.nn as nn

from f1_predict.models.multimodal_fusion import MultiModalFusionModel


class TestMultiModalFusionModelInitialization:
    """Test model creation and initialization."""

    def test_model_initialization(self):
        """Test model initializes with correct architecture."""
        model = MultiModalFusionModel(
            image_feature_dim=512,
            tabular_input_dim=10,
            hidden_dim=256,
            output_dim=20
        )
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_model_to_device(self):
        """Test model can be moved to device."""
        model = MultiModalFusionModel(
            image_feature_dim=512,
            tabular_input_dim=10,
            hidden_dim=256,
            output_dim=20
        )
        device = torch.device('cpu')
        model.to(device)
        # Should not raise error


class TestMultiModalFusionModelForward:
    """Test forward pass behavior."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return MultiModalFusionModel(
            image_feature_dim=512,
            tabular_input_dim=10,
            hidden_dim=256,
            output_dim=20
        )

    def test_forward_with_both_modalities(self, model):
        """Test forward pass with image and tabular features."""
        batch_size = 4
        image_features = torch.randn(batch_size, 512)
        tabular_features = torch.randn(batch_size, 10)

        output = model(image_features, tabular_features)

        assert output.shape == (batch_size, 20)

    def test_forward_with_missing_images(self, model):
        """Test forward pass when image features are None."""
        batch_size = 4
        image_features = None
        tabular_features = torch.randn(batch_size, 10)

        # Should handle None for images gracefully
        output = model(image_features, tabular_features)

        assert output.shape == (batch_size, 20)

    def test_forward_output_values_reasonable(self, model):
        """Test output values are in reasonable range."""
        image_features = torch.randn(1, 512)
        tabular_features = torch.randn(1, 10)

        output = model(image_features, tabular_features)

        # Output should be finite
        assert torch.isfinite(output).all()
        # Values should not be NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_batch(self, model):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            image_features = torch.randn(batch_size, 512)
            tabular_features = torch.randn(batch_size, 10)

            output = model(image_features, tabular_features)

            assert output.shape == (batch_size, 20)


class TestMultiModalFusionModelGradients:
    """Test gradient flow through model."""

    def test_gradients_flow_through_fusion(self):
        """Test gradients flow through fusion network."""
        model = MultiModalFusionModel(
            image_feature_dim=512,
            tabular_input_dim=10,
            hidden_dim=256,
            output_dim=20
        )

        image_features = torch.randn(4, 512, requires_grad=True)
        tabular_features = torch.randn(4, 10, requires_grad=True)

        output = model(image_features, tabular_features)
        loss = output.sum()
        loss.backward()

        # Gradients should exist
        assert image_features.grad is not None
        assert tabular_features.grad is not None

    def test_fusion_network_parameters_trainable(self):
        """Test fusion network has trainable parameters."""
        model = MultiModalFusionModel(
            image_feature_dim=512,
            tabular_input_dim=10,
            hidden_dim=256,
            output_dim=20
        )

        # Count parameters
        params = list(model.parameters())
        assert len(params) > 0

        # All should be trainable (requires_grad=True)
        for param in params:
            assert param.requires_grad


class TestMultiModalFusionModelArchitecture:
    """Test model architecture components."""

    def test_model_has_tabular_encoder(self):
        """Test model has tabular encoder component."""
        model = MultiModalFusionModel(
            image_feature_dim=512,
            tabular_input_dim=10,
            hidden_dim=256,
            output_dim=20
        )
        assert hasattr(model, 'tabular_encoder')

    def test_model_has_fusion_network(self):
        """Test model has fusion network component."""
        model = MultiModalFusionModel(
            image_feature_dim=512,
            tabular_input_dim=10,
            hidden_dim=256,
            output_dim=20
        )
        assert hasattr(model, 'fusion_network')

    def test_fusion_input_dimension_correct(self):
        """Test fusion network receives correct input dimension."""
        image_dim = 512
        tabular_dim = 10
        encoded_tabular_dim = 128  # Tabular features are encoded to this dimension
        model = MultiModalFusionModel(
            image_feature_dim=image_dim,
            tabular_input_dim=tabular_dim,
            hidden_dim=256,
            output_dim=20
        )

        # Fusion should accept concatenated features (image + encoded_tabular)
        # The tabular features are encoded to 128 dims first, then concatenated
        expected_fusion_input = image_dim + encoded_tabular_dim

        # Verify by checking first layer of fusion network
        first_layer = model.fusion_network[0]
        assert isinstance(first_layer, nn.Linear)
        assert first_layer.in_features == expected_fusion_input


class TestMultiModalFusionModelTrainingMode:
    """Test training and eval modes."""

    def test_model_train_mode(self):
        """Test model enters train mode."""
        model = MultiModalFusionModel(
            image_feature_dim=512,
            tabular_input_dim=10,
            hidden_dim=256,
            output_dim=20
        )

        model.train()
        assert model.training

    def test_model_eval_mode(self):
        """Test model enters eval mode."""
        model = MultiModalFusionModel(
            image_feature_dim=512,
            tabular_input_dim=10,
            hidden_dim=256,
            output_dim=20
        )

        model.eval()
        assert not model.training

    def test_dropout_disabled_in_eval(self):
        """Test dropout is disabled in eval mode."""
        model = MultiModalFusionModel(
            image_feature_dim=512,
            tabular_input_dim=10,
            hidden_dim=256,
            output_dim=20
        )

        image_features = torch.randn(100, 512)
        tabular_features = torch.randn(100, 10)

        # Train mode: outputs might vary due to dropout
        model.train()
        with torch.no_grad():
            output_train_1 = model(image_features, tabular_features)
            output_train_2 = model(image_features, tabular_features)

        # Eval mode: outputs should be identical
        model.eval()
        with torch.no_grad():
            output_eval_1 = model(image_features, tabular_features)
            output_eval_2 = model(image_features, tabular_features)

        # Eval outputs should be identical
        assert torch.allclose(output_eval_1, output_eval_2)
