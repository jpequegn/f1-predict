"""Multi-modal fusion model combining image and tabular features."""

from typing import Optional

import torch
from torch import nn


class MultiModalFusionModel(nn.Module):
    """Early fusion model combining image and tabular features.

    Architecture:
    - Image features (512 dim) from pretrained ResNet-18
    - Tabular features encoded to 128 dim
    - Early fusion: concatenate → 640 dims
    - Fusion network: 640 → 256 → 128 → output
    """

    def __init__(
        self,
        image_feature_dim: int = 512,
        tabular_input_dim: int = 10,
        hidden_dim: int = 256,
        output_dim: int = 20,
        dropout_rate: float = 0.3
    ):
        """Initialize fusion model.

        Args:
            image_feature_dim: Dimension of image features from encoder (default: 512 for ResNet-18)
            tabular_input_dim: Dimension of input tabular features
            hidden_dim: Hidden dimension for fusion network
            output_dim: Output dimension (e.g., 20 for position classification)
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()

        self.image_feature_dim = image_feature_dim
        self.tabular_input_dim = tabular_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Tabular encoder: input_dim → 128
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 128)
        )

        # Early fusion network
        # Input: concatenated image_features (512) + tabular_encoded (128) = 640
        fusion_input_dim = image_feature_dim + 128
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim)
        )

    def forward(
        self,
        image_features: Optional[torch.Tensor],
        tabular_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through fusion model.

        Args:
            image_features: (batch_size, 512) or None if images missing
            tabular_features: (batch_size, tabular_input_dim)

        Returns:
            predictions: (batch_size, output_dim)
        """
        batch_size = tabular_features.shape[0]

        # Handle missing images gracefully
        if image_features is None:
            # Use zero vector if images missing
            image_features = torch.zeros(
                batch_size,
                self.image_feature_dim,
                device=tabular_features.device,
                dtype=tabular_features.dtype
            )

        # Encode tabular features
        tabular_encoded = self.tabular_encoder(tabular_features)  # (batch_size, 128)

        # Early fusion: concatenate image and tabular features
        fused = torch.cat([image_features, tabular_encoded], dim=1)  # (batch_size, 640)

        # Pass through fusion network
        output = self.fusion_network(fused)  # (batch_size, output_dim)

        return output
