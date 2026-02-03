"""Visual explanation system for multi-modal F1 predictions.

Implements GradCAM, attention visualization, and feature attribution
for interpreting model predictions and understanding which image
regions and features influence results.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN visualization.

    Highlights regions of the input image that are important for
    the model's prediction using gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ):
        """Initialize GradCAM.

        Args:
            model: CNN model to explain
            target_layer: Target convolutional layer for visualization
        """
        self.model = model
        self.target_layer = target_layer

        # Storage for activations and gradients
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""

        def forward_hook(
            module: nn.Module,  # noqa: ARG001
            input_tensor: tuple,  # noqa: ARG001
            output: torch.Tensor,
        ) -> None:
            self.activations = output.detach()

        def backward_hook(
            module: nn.Module,  # noqa: ARG001
            grad_input: Any,  # noqa: ARG001
            grad_output: Any,
        ) -> None:
            if isinstance(grad_output, tuple) and len(grad_output) > 0:
                self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)  # type: ignore[arg-type]

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate class activation map.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (uses max if None)

        Returns:
            CAM heatmap normalized to [0, 1]
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # Generate CAM
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture gradients/activations")

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU and normalize
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def visualize(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Overlay CAM on original image.

        Args:
            image: Original image (H, W, C) normalized to [0, 1]
            cam: CAM heatmap
            alpha: Blending factor

        Returns:
            Blended visualization
        """
        # Resize CAM to match image size
        from scipy.ndimage import zoom

        h, w = image.shape[:2]
        cam_resized = zoom(cam, (h / cam.shape[0], w / cam.shape[1]))

        # Create heatmap
        cmap_jet = plt.colormaps.get_cmap("jet")
        heatmap = cmap_jet(cam_resized)[:, :, :3]

        # Blend with original image
        blended = alpha * heatmap + (1 - alpha) * image

        return np.clip(blended, 0, 1)


class MultiModalAttentionVisualizer:
    """Visualize attention weights from multi-modal fusion models.

    Shows how the model attends to different parts of images
    and which tabular features are most important.
    """

    def __init__(self, model: nn.Module):
        """Initialize attention visualizer.

        Args:
            model: Multi-modal model with attention layers
        """
        self.model = model

    def get_attention_weights(
        self,
        image_features: torch.Tensor,  # noqa: ARG002 - for interface compatibility
        tabular_features: torch.Tensor,  # noqa: ARG002 - for interface compatibility
    ) -> dict[str, torch.Tensor]:
        """Extract attention weights from model.

        Args:
            image_features: Image feature tensor (unused, kept for interface)
            tabular_features: Tabular feature tensor (unused, kept for interface)

        Returns:
            Dictionary of attention weight tensors
        """
        attention_weights: dict[str, torch.Tensor] = {}

        # Forward pass with attention extraction
        self.model.eval()
        with torch.no_grad():
            # Try to extract attention from model
            # This assumes model has attention modules with stored weights
            for name, module in self.model.named_modules():
                if hasattr(module, "attention_weights") and module.attention_weights is not None:
                    attention_weights[name] = module.attention_weights.clone()

        return attention_weights

    def visualize_cross_modal_attention(
        self,
        image: np.ndarray,
        attention_weights: torch.Tensor,
        feature_names: list[str],
        output_path: Optional[str] = None,
    ) -> Any:
        """Visualize cross-modal attention between image and tabular features.

        Args:
            image: Original image array
            attention_weights: Attention weight tensor
            feature_names: Names of tabular features
            output_path: Optional path to save visualization

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Image with attention overlay
        ax1 = axes[0]
        ax1.imshow(image)
        ax1.set_title("Image Input", fontsize=12, fontweight="bold")
        ax1.axis("off")

        # Attention weights for tabular features
        ax2 = axes[1]
        weights_np = attention_weights.cpu().numpy()

        if weights_np.ndim > 1:
            weights_np = weights_np.mean(axis=0)  # Average over heads

        ax2.barh(range(len(feature_names)), weights_np)
        ax2.set_yticks(range(len(feature_names)))
        ax2.set_yticklabels(feature_names)
        ax2.set_xlabel("Attention Weight", fontsize=10)
        ax2.set_title("Feature Attention", fontsize=12, fontweight="bold")

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=100, bbox_inches="tight")
            logger.info(f"Saved attention visualization to {output_path}")

        return fig


class FeatureAttributor:
    """Compute feature attributions for model predictions.

    Uses integrated gradients and other methods to understand
    feature importance.
    """

    def __init__(self, model: nn.Module, baseline: Optional[torch.Tensor] = None):
        """Initialize feature attributor.

        Args:
            model: Model to explain
            baseline: Baseline input for integrated gradients
        """
        self.model = model
        self.baseline = baseline

    def integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        steps: int = 50,
    ) -> torch.Tensor:
        """Compute integrated gradients attribution.

        Args:
            input_tensor: Input tensor
            target_class: Target class for attribution
            steps: Number of interpolation steps

        Returns:
            Attribution tensor same shape as input
        """
        self.model.eval()

        # Create baseline if not provided
        if self.baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            baseline = self.baseline

        # Generate interpolated inputs
        scaled_inputs = [
            baseline + (float(i) / steps) * (input_tensor - baseline)
            for i in range(steps + 1)
        ]
        scaled_inputs = torch.cat(scaled_inputs, dim=0)
        scaled_inputs.requires_grad = True

        # Forward pass
        outputs = self.model(scaled_inputs)

        # Backward pass
        self.model.zero_grad()
        target_scores = outputs[:, target_class]
        target_scores.sum().backward()

        # Get gradients
        gradients = scaled_inputs.grad
        if gradients is None:
            raise RuntimeError("Gradients not computed")

        # Compute integrated gradients
        avg_gradients = gradients.mean(dim=0, keepdim=True)
        integrated_grad = (input_tensor - baseline) * avg_gradients

        return integrated_grad.squeeze(0)

    def compute_tabular_importance(
        self,
        tabular_features: torch.Tensor,
        target_class: int,
        feature_names: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """Compute importance of each tabular feature.

        Args:
            tabular_features: Tabular input tensor
            target_class: Target class
            feature_names: Optional feature names

        Returns:
            Dictionary mapping feature name to importance score
        """
        attributions = self.integrated_gradients(
            tabular_features, target_class, steps=50
        )
        attributions_np = attributions.abs().cpu().numpy()

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(attributions_np))]

        return {
            name: float(attr)
            for name, attr in zip(feature_names, attributions_np)
        }


class ExplanationDashboard:
    """Generate comprehensive explanation dashboards for predictions.

    Combines multiple explanation methods into a unified visualization.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: list[str],
        output_dir: str = "explanations",
    ):
        """Initialize explanation dashboard.

        Args:
            model: Model to explain
            feature_names: Names of tabular features
            output_dir: Directory for saving visualizations
        """
        self.model = model
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize explainers
        self.attention_viz = MultiModalAttentionVisualizer(model)
        self.feature_attributor = FeatureAttributor(model)

    def generate_explanation(
        self,
        image: np.ndarray,
        image_tensor: torch.Tensor,
        tabular_features: torch.Tensor,
        prediction_class: int,
        race_id: str,
        driver_id: str,
    ) -> dict[str, Any]:
        """Generate comprehensive explanation for a prediction.

        Args:
            image: Original image array
            image_tensor: Preprocessed image tensor
            tabular_features: Tabular features
            prediction_class: Predicted class
            race_id: Race identifier
            driver_id: Driver identifier

        Returns:
            Dictionary with explanation data and file paths
        """
        explanation: dict[str, Any] = {
            "race_id": race_id,
            "driver_id": driver_id,
            "prediction_class": prediction_class,
        }

        # Get attention weights
        attention = self.attention_viz.get_attention_weights(
            image_tensor, tabular_features
        )
        explanation["attention_weights"] = {
            k: v.tolist() for k, v in attention.items()
        }

        # Compute feature importance
        importance = self.feature_attributor.compute_tabular_importance(
            tabular_features, prediction_class, self.feature_names
        )
        explanation["feature_importance"] = importance

        # Generate visualization
        fig = self._create_dashboard_figure(
            image, attention, importance, race_id, driver_id
        )

        # Save figure
        output_path = self.output_dir / f"{race_id}_{driver_id}_explanation.png"
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        explanation["visualization_path"] = str(output_path)

        return explanation

    def _create_dashboard_figure(
        self,
        image: np.ndarray,
        attention: dict[str, torch.Tensor],
        importance: dict[str, float],
        race_id: str,
        driver_id: str,
    ) -> Any:
        """Create dashboard figure.

        Args:
            image: Original image
            attention: Attention weights
            importance: Feature importance scores
            race_id: Race ID
            driver_id: Driver ID

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 8))

        # Layout: 2 rows, 3 columns
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        ax1.set_title("Input Image", fontsize=12, fontweight="bold")
        ax1.axis("off")

        # Feature importance bar chart
        ax2 = fig.add_subplot(gs[0, 1:])
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_importance[:10]]  # Top 10
        values = [x[1] for x in sorted_importance[:10]]
        colors = ["#FF6B6B" if v > 0 else "#4ECDC4" for v in values]
        ax2.barh(range(len(names)), values, color=colors)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names)
        ax2.set_xlabel("Importance Score", fontsize=10)
        ax2.set_title("Top Feature Importance", fontsize=12, fontweight="bold")
        ax2.invert_yaxis()

        # Attention heatmap (if available)
        ax3 = fig.add_subplot(gs[1, 0])
        if attention:
            # Use first attention layer
            first_attn = list(attention.values())[0]
            if first_attn.ndim >= 2:
                attn_np = first_attn.cpu().numpy()
                if attn_np.ndim > 2:
                    attn_np = attn_np.mean(axis=0)  # Average over batch/heads
                ax3.imshow(attn_np, cmap="viridis", aspect="auto")
                ax3.set_title("Attention Heatmap", fontsize=12, fontweight="bold")
        else:
            ax3.text(0.5, 0.5, "No attention\nweights available",
                     ha="center", va="center", fontsize=10)
            ax3.set_title("Attention", fontsize=12, fontweight="bold")
        ax3.axis("off")

        # Summary text
        ax4 = fig.add_subplot(gs[1, 1:])
        ax4.axis("off")

        summary_text = (
            f"Prediction Explanation\n"
            f"{'=' * 30}\n\n"
            f"Race: {race_id}\n"
            f"Driver: {driver_id}\n\n"
            f"Top Contributing Features:\n"
        )
        for name, value in sorted_importance[:5]:
            summary_text += f"  • {name}: {value:.4f}\n"

        ax4.text(
            0.1, 0.9, summary_text,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

        fig.suptitle(
            f"Multi-Modal Prediction Explanation: {race_id} - {driver_id}",
            fontsize=14,
            fontweight="bold",
        )

        return fig


def visualize_prediction_confidence(
    predictions: torch.Tensor,
    class_names: list[str],
    output_path: Optional[str] = None,
) -> Any:
    """Visualize prediction confidence distribution.

    Args:
        predictions: Softmax prediction probabilities
        class_names: Names of classes (e.g., positions 1-20)
        output_path: Optional path to save visualization

    Returns:
        Matplotlib figure
    """
    probs = predictions.softmax(dim=-1).cpu().numpy().flatten()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar chart of probabilities
    x = range(len(class_names))
    cmap = plt.colormaps.get_cmap("RdYlGn")
    colors = cmap(probs)

    bars = ax.bar(x, probs, color=colors)

    # Highlight top prediction
    top_idx = probs.argmax()
    bars[top_idx].set_edgecolor("black")
    bars[top_idx].set_linewidth(2)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_xlabel("Position", fontsize=12)
    ax.set_title("Prediction Confidence Distribution", fontsize=14, fontweight="bold")

    # Add confidence text
    ax.text(
        0.95, 0.95,
        f"Predicted: {class_names[top_idx]}\nConfidence: {probs[top_idx]:.2%}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=100, bbox_inches="tight")

    return fig
