"""Advanced CNN architectures for F1 multi-modal learning.

Provides multiple CNN backbone options including EfficientNet, multi-scale
feature extraction, and custom F1-specific architectures.
"""

import logging
from typing import Literal, Optional

import torch
from torch import nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
)

logger = logging.getLogger(__name__)


BackboneType = Literal[
    "resnet18", "resnet50", "efficientnet_b0", "efficientnet_b2", "custom_f1"
]


class AdvancedVisionEncoder(nn.Module):
    """Advanced vision encoder with multiple backbone options.

    Supports ResNet, EfficientNet, and custom architectures with
    multi-scale feature extraction capabilities.
    """

    # Feature dimensions for each backbone
    BACKBONE_DIMS: dict[str, int] = {
        "resnet18": 512,
        "resnet50": 2048,
        "efficientnet_b0": 1280,
        "efficientnet_b2": 1408,
        "custom_f1": 512,
    }

    def __init__(
        self,
        backbone: BackboneType = "efficientnet_b0",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        output_dim: Optional[int] = None,
        dropout_rate: float = 0.2,
    ):
        """Initialize advanced vision encoder.

        Args:
            backbone: Backbone architecture to use
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone weights
            output_dim: Optional projection to specific dimension
            dropout_rate: Dropout rate for projection layer
        """
        super().__init__()

        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone
        self.backbone_dim = self.BACKBONE_DIMS[backbone]

        # Build backbone
        self.backbone = self._build_backbone(backbone, pretrained)

        # Freeze if requested
        if freeze_backbone:
            self._freeze_backbone()

        # Optional projection layer
        self.output_dim = output_dim or self.backbone_dim
        if output_dim and output_dim != self.backbone_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.backbone_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
        else:
            self.projection = None

    def _build_backbone(
        self, backbone: BackboneType, pretrained: bool
    ) -> nn.Module:
        """Build backbone network.

        Args:
            backbone: Backbone type
            pretrained: Use pretrained weights

        Returns:
            Backbone module
        """
        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            # Remove classification head
            return nn.Sequential(*list(model.children())[:-1])

        if backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            return nn.Sequential(*list(model.children())[:-1])

        if backbone == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            # Remove classifier, keep features + avgpool
            return nn.Sequential(model.features, model.avgpool)

        if backbone == "efficientnet_b2":
            weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b2(weights=weights)
            return nn.Sequential(model.features, model.avgpool)

        if backbone == "custom_f1":
            return CustomF1CNN()

        # This should never be reached due to type hints, but kept for safety
        raise ValueError(f"Unknown backbone: {backbone}")  # pragma: no cover

    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images.

        Args:
            x: Input tensor (batch_size, 3, H, W)

        Returns:
            Feature tensor (batch_size, output_dim)
        """
        # Extract backbone features
        features = self.backbone(x)

        # Flatten
        features = torch.flatten(features, 1)

        # Project if needed
        if self.projection is not None:
            features = self.projection(features)

        return features

    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.output_dim


class CustomF1CNN(nn.Module):
    """Custom CNN architecture designed for F1 telemetry images.

    Optimized for processing telemetry plots with specific attention
    to temporal patterns and data visualization characteristics.
    """

    def __init__(self, input_channels: int = 3, output_dim: int = 512):
        """Initialize custom F1 CNN.

        Args:
            input_channels: Number of input channels
            output_dim: Output feature dimension
        """
        super().__init__()

        self.output_dim = output_dim

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 5: 14x14 -> 7x7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Projection to output dim
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, C, H, W)

        Returns:
            Feature tensor (batch_size, output_dim)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales for richer representations.

    Uses Feature Pyramid Network (FPN) style architecture to capture
    both fine-grained and high-level features.
    """

    def __init__(
        self,
        backbone: BackboneType = "resnet50",
        pretrained: bool = True,
        output_dim: int = 512,
    ):
        """Initialize multi-scale feature extractor.

        Args:
            backbone: Base backbone architecture
            pretrained: Use pretrained weights
            output_dim: Final output feature dimension
        """
        super().__init__()

        self.output_dim = output_dim

        # Build ResNet backbone with access to intermediate layers
        if backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.layer_dims = [256, 512, 1024, 2048]
        else:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            self.layer_dims = [64, 128, 256, 512]

        # Extract backbone layers
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Lateral connections for FPN
        self.lateral4 = nn.Conv2d(self.layer_dims[3], 256, 1)
        self.lateral3 = nn.Conv2d(self.layer_dims[2], 256, 1)
        self.lateral2 = nn.Conv2d(self.layer_dims[1], 256, 1)

        # Smooth layers
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, 3, padding=1)

        # Global average pool and projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(256 * 3, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale features.

        Args:
            x: Input tensor (batch_size, 3, H, W)

        Returns:
            Feature tensor (batch_size, output_dim)
        """
        # Bottom-up pathway
        c1 = self.stem(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down pathway with lateral connections
        p5 = self.lateral4(c5)
        p4 = self._upsample_add(p5, self.lateral3(c4))
        p3 = self._upsample_add(p4, self.lateral2(c3))

        # Smooth
        p5 = self.smooth4(p5)
        p4 = self.smooth3(p4)
        p3 = self.smooth2(p3)

        # Pool and concatenate
        f5 = torch.flatten(self.avgpool(p5), 1)
        f4 = torch.flatten(self.avgpool(p4), 1)
        f3 = torch.flatten(self.avgpool(p3), 1)

        # Concatenate multi-scale features
        features = torch.cat([f5, f4, f3], dim=1)

        # Project to output dimension
        return self.projection(features)

    def _upsample_add(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Upsample x and add to y."""
        _, _, h, w = y.size()
        return nn.functional.interpolate(
            x, size=(h, w), mode="nearest"
        ) + y


class ImageTypeSpecificEncoder(nn.Module):
    """Encoder that uses different sub-networks for different image types.

    Allows specialized processing for telemetry plots vs race photos
    vs track layouts, then combines features.
    """

    def __init__(
        self,
        num_image_types: int = 4,
        backbone: BackboneType = "efficientnet_b0",
        pretrained: bool = True,
        shared_backbone: bool = True,
        output_dim: int = 512,
    ):
        """Initialize image-type specific encoder.

        Args:
            num_image_types: Number of different image types
            backbone: Backbone architecture
            pretrained: Use pretrained weights
            shared_backbone: Share backbone across types (vs separate)
            output_dim: Output feature dimension
        """
        super().__init__()

        self.num_image_types = num_image_types
        self.shared_backbone = shared_backbone
        self.output_dim = output_dim

        if shared_backbone:
            # Single shared backbone
            self.backbone = AdvancedVisionEncoder(
                backbone=backbone,
                pretrained=pretrained,
                freeze_backbone=True,
            )
            backbone_dim = self.backbone.get_feature_dim()

            # Type-specific heads
            self.type_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(backbone_dim, output_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                )
                for _ in range(num_image_types)
            ])
        else:
            # Separate backbones per type
            self.backbones = nn.ModuleList([
                AdvancedVisionEncoder(
                    backbone=backbone,
                    pretrained=pretrained,
                    freeze_backbone=True,
                    output_dim=output_dim,
                )
                for _ in range(num_image_types)
            ])

    def forward(
        self,
        x: torch.Tensor,
        image_type_idx: int = 0,
    ) -> torch.Tensor:
        """Extract features for a specific image type.

        Args:
            x: Input tensor (batch_size, 3, H, W)
            image_type_idx: Index of image type (0 to num_image_types-1)

        Returns:
            Feature tensor (batch_size, output_dim)
        """
        if self.shared_backbone:
            features = self.backbone(x)
            return self.type_heads[image_type_idx](features)
        return self.backbones[image_type_idx](x)

    def forward_all_types(
        self,
        images: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Extract and aggregate features from multiple image types.

        Args:
            images: Dict mapping image_type_idx to image tensors

        Returns:
            Aggregated feature tensor
        """
        features_list = []

        for type_idx, img in images.items():
            if img is not None:
                feat = self.forward(img, type_idx)
                features_list.append(feat)

        if not features_list:
            # Return zeros if no images
            batch_size = 1
            return torch.zeros(batch_size, self.output_dim)

        # Average features across types
        features = torch.stack(features_list)
        return features.mean(dim=0)


class TemporalImageEncoder(nn.Module):
    """Encode sequences of images (e.g., lap-by-lap telemetry snapshots).

    Combines CNN feature extraction with temporal modeling.
    """

    def __init__(
        self,
        backbone: BackboneType = "efficientnet_b0",
        pretrained: bool = True,
        hidden_dim: int = 256,
        num_layers: int = 2,
        output_dim: int = 512,
    ):
        """Initialize temporal image encoder.

        Args:
            backbone: CNN backbone for frame encoding
            pretrained: Use pretrained weights
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_dim: Final output dimension
        """
        super().__init__()

        self.output_dim = output_dim

        # CNN encoder for individual frames
        self.frame_encoder = AdvancedVisionEncoder(
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=True,
        )
        frame_dim = self.frame_encoder.get_feature_dim()

        # Temporal modeling with LSTM
        self.lstm = nn.LSTM(
            input_size=frame_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0,
        )

        # Projection to output
        self.projection = nn.Linear(hidden_dim * 2, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode image sequence.

        Args:
            x: Input tensor (batch_size, seq_len, 3, H, W)
            lengths: Optional sequence lengths for packing

        Returns:
            Feature tensor (batch_size, output_dim)
        """
        batch_size, seq_len, c, h, w = x.shape

        # Encode each frame
        x = x.view(batch_size * seq_len, c, h, w)
        frame_features = self.frame_encoder(x)
        frame_features = frame_features.view(batch_size, seq_len, -1)

        # Temporal modeling
        if lengths is not None:
            # Pack for variable lengths
            packed = nn.utils.rnn.pack_padded_sequence(
                frame_features,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(frame_features)

        # Use final hidden state (concat forward and backward)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        final_state = torch.cat([h_forward, h_backward], dim=1)

        return self.projection(final_state)
