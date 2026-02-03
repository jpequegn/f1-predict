"""Tests for visual explanation system."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from f1_predict.models.visual_explanations import (
    ExplanationDashboard,
    FeatureAttributor,
    GradCAM,
    MultiModalAttentionVisualizer,
    visualize_prediction_confidence,
)


class SimpleCNN(nn.Module):
    """Simple CNN for testing GradCAM."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SimpleModel(nn.Module):
    """Simple model for testing feature attribution."""

    def __init__(self, input_dim=20, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestGradCAM:
    """Tests for GradCAM class."""

    @pytest.fixture
    def model(self):
        """Create simple CNN model."""
        return SimpleCNN(num_classes=10)

    @pytest.fixture
    def gradcam(self, model):
        """Create GradCAM instance."""
        return GradCAM(model=model, target_layer=model.conv2)

    @pytest.fixture
    def sample_image(self):
        """Create sample image tensor."""
        return torch.randn(1, 3, 32, 32)

    def test_init(self, model):
        """Test GradCAM initialization."""
        gradcam = GradCAM(model=model, target_layer=model.conv2)
        assert gradcam.model is model
        assert gradcam.target_layer is model.conv2

    def test_generate_cam(self, gradcam, sample_image):
        """Test CAM generation."""
        cam = gradcam.generate_cam(sample_image)
        assert isinstance(cam, np.ndarray)
        assert cam.min() >= 0
        assert cam.max() <= 1

    def test_generate_cam_with_target_class(self, gradcam, sample_image):
        """Test CAM generation with specific target class."""
        cam = gradcam.generate_cam(sample_image, target_class=5)
        assert isinstance(cam, np.ndarray)

    def test_generate_cam_auto_target(self, gradcam, sample_image):
        """Test CAM generation with automatic target class."""
        cam = gradcam.generate_cam(sample_image, target_class=None)
        assert isinstance(cam, np.ndarray)

    def test_visualize(self, gradcam, sample_image):
        """Test CAM visualization."""
        cam = gradcam.generate_cam(sample_image)
        # Create dummy image array
        image = np.random.rand(32, 32, 3)
        visualization = gradcam.visualize(image, cam)
        assert visualization.shape == image.shape
        assert visualization.min() >= 0
        assert visualization.max() <= 1


class TestMultiModalAttentionVisualizer:
    """Tests for MultiModalAttentionVisualizer class."""

    @pytest.fixture
    def model(self):
        """Create mock model with attention."""
        model = MagicMock()
        model.named_modules.return_value = []
        model.eval = MagicMock()
        return model

    @pytest.fixture
    def visualizer(self, model):
        """Create visualizer instance."""
        return MultiModalAttentionVisualizer(model=model)

    def test_init(self, model):
        """Test visualizer initialization."""
        visualizer = MultiModalAttentionVisualizer(model=model)
        assert visualizer.model is model

    def test_get_attention_weights(self, visualizer):
        """Test getting attention weights."""
        image_features = torch.randn(1, 256)
        tabular_features = torch.randn(1, 64)
        weights = visualizer.get_attention_weights(image_features, tabular_features)
        assert isinstance(weights, dict)

    def test_visualize_cross_modal_attention(self, visualizer):
        """Test cross-modal attention visualization."""
        image = np.random.rand(224, 224, 3)
        attention_weights = torch.randn(10)
        feature_names = [f"feature_{i}" for i in range(10)]

        fig = visualizer.visualize_cross_modal_attention(
            image=image,
            attention_weights=attention_weights,
            feature_names=feature_names,
        )
        assert fig is not None

    def test_visualize_with_output_path(self, visualizer, tmp_path):
        """Test visualization saving."""
        image = np.random.rand(224, 224, 3)
        attention_weights = torch.randn(10)
        feature_names = [f"feature_{i}" for i in range(10)]
        output_path = str(tmp_path / "attention.png")

        fig = visualizer.visualize_cross_modal_attention(
            image=image,
            attention_weights=attention_weights,
            feature_names=feature_names,
            output_path=output_path,
        )
        assert fig is not None


class TestFeatureAttributor:
    """Tests for FeatureAttributor class."""

    @pytest.fixture
    def model(self):
        """Create simple model."""
        return SimpleModel(input_dim=20, num_classes=10)

    @pytest.fixture
    def attributor(self, model):
        """Create attributor instance."""
        return FeatureAttributor(model=model)

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(1, 20)

    def test_init(self, model):
        """Test attributor initialization."""
        attributor = FeatureAttributor(model=model)
        assert attributor.model is model

    def test_init_with_baseline(self, model):
        """Test initialization with custom baseline."""
        baseline = torch.zeros(1, 20)
        attributor = FeatureAttributor(model=model, baseline=baseline)
        assert attributor.baseline is not None

    def test_integrated_gradients(self, attributor, sample_input):
        """Test integrated gradients computation."""
        attributions = attributor.integrated_gradients(
            input_tensor=sample_input,
            target_class=0,
            steps=10,
        )
        assert attributions.shape == sample_input.shape[1:]

    def test_integrated_gradients_different_steps(self, attributor, sample_input):
        """Test integrated gradients with different step counts."""
        for steps in [5, 20, 50]:
            attributions = attributor.integrated_gradients(
                input_tensor=sample_input,
                target_class=0,
                steps=steps,
            )
            assert attributions is not None

    def test_compute_tabular_importance(self, attributor, sample_input):
        """Test tabular feature importance computation."""
        feature_names = [f"feature_{i}" for i in range(20)]
        importance = attributor.compute_tabular_importance(
            tabular_features=sample_input,
            target_class=0,
            feature_names=feature_names,
        )
        assert isinstance(importance, dict)
        assert len(importance) == 20

    def test_compute_tabular_importance_no_names(self, attributor, sample_input):
        """Test importance computation without feature names."""
        importance = attributor.compute_tabular_importance(
            tabular_features=sample_input,
            target_class=0,
        )
        assert isinstance(importance, dict)


class TestExplanationDashboard:
    """Tests for ExplanationDashboard class."""

    @pytest.fixture
    def model(self):
        """Create mock model."""
        model = MagicMock(spec=nn.Module)
        model.eval = MagicMock()
        model.named_modules = MagicMock(return_value=[])
        model.zero_grad = MagicMock()
        return model

    @pytest.fixture
    def dashboard(self, model, tmp_path):
        """Create dashboard instance."""
        return ExplanationDashboard(
            model=model,
            feature_names=[f"feature_{i}" for i in range(10)],
            output_dir=str(tmp_path),
        )

    def test_init(self, model, tmp_path):
        """Test dashboard initialization."""
        dashboard = ExplanationDashboard(
            model=model,
            feature_names=["f1", "f2"],
            output_dir=str(tmp_path),
        )
        assert dashboard.model is model
        assert len(dashboard.feature_names) == 2

    @patch.object(FeatureAttributor, "compute_tabular_importance")
    def test_generate_explanation(self, mock_importance, dashboard):
        """Test explanation generation."""
        mock_importance.return_value = {"feature_0": 0.5, "feature_1": 0.3}

        image = np.random.rand(224, 224, 3)
        image_tensor = torch.randn(1, 3, 224, 224)
        tabular_features = torch.randn(1, 10)

        explanation = dashboard.generate_explanation(
            image=image,
            image_tensor=image_tensor,
            tabular_features=tabular_features,
            prediction_class=0,
            race_id="test_race",
            driver_id="VER",
        )

        assert "race_id" in explanation
        assert "driver_id" in explanation
        assert "prediction_class" in explanation


class TestVisualizePredictionConfidence:
    """Tests for visualize_prediction_confidence function."""

    def test_basic_visualization(self):
        """Test basic confidence visualization."""
        predictions = torch.randn(1, 10)
        class_names = [f"P{i+1}" for i in range(10)]

        fig = visualize_prediction_confidence(
            predictions=predictions,
            class_names=class_names,
        )
        assert fig is not None

    def test_visualization_with_output_path(self, tmp_path):
        """Test visualization with file output."""
        predictions = torch.randn(1, 20)
        class_names = [f"P{i+1}" for i in range(20)]
        output_path = str(tmp_path / "confidence.png")

        fig = visualize_prediction_confidence(
            predictions=predictions,
            class_names=class_names,
            output_path=output_path,
        )
        assert fig is not None

    def test_visualization_various_class_counts(self):
        """Test visualization with different class counts."""
        for num_classes in [5, 10, 20]:
            predictions = torch.randn(1, num_classes)
            class_names = [f"P{i+1}" for i in range(num_classes)]

            fig = visualize_prediction_confidence(
                predictions=predictions,
                class_names=class_names,
            )
            assert fig is not None


class TestExplanationEdgeCases:
    """Tests for explanation edge cases."""

    def test_gradcam_with_small_image(self):
        """Test GradCAM with small image."""
        model = SimpleCNN()
        gradcam = GradCAM(model=model, target_layer=model.conv2)
        small_image = torch.randn(1, 3, 8, 8)

        cam = gradcam.generate_cam(small_image)
        assert isinstance(cam, np.ndarray)

    def test_gradcam_batch_input(self):
        """Test GradCAM with batch input."""
        model = SimpleCNN()
        gradcam = GradCAM(model=model, target_layer=model.conv2)
        batch = torch.randn(1, 3, 32, 32)  # GradCAM typically needs batch=1

        cam = gradcam.generate_cam(batch)
        assert isinstance(cam, np.ndarray)

    def test_attributor_zero_baseline(self):
        """Test attributor with zero baseline."""
        model = SimpleModel()
        baseline = torch.zeros(1, 20)
        attributor = FeatureAttributor(model=model, baseline=baseline)

        input_tensor = torch.randn(1, 20)
        attributions = attributor.integrated_gradients(
            input_tensor=input_tensor,
            target_class=0,
            steps=10,
        )
        assert attributions is not None

    def test_high_confidence_prediction(self):
        """Test visualization with high confidence prediction."""
        # Create highly confident prediction
        predictions = torch.zeros(1, 10)
        predictions[0, 0] = 10.0  # Very high logit
        class_names = [f"P{i+1}" for i in range(10)]

        fig = visualize_prediction_confidence(
            predictions=predictions,
            class_names=class_names,
        )
        assert fig is not None

    def test_uniform_predictions(self):
        """Test visualization with uniform predictions."""
        predictions = torch.ones(1, 10)  # Uniform logits
        class_names = [f"P{i+1}" for i in range(10)]

        fig = visualize_prediction_confidence(
            predictions=predictions,
            class_names=class_names,
        )
        assert fig is not None
