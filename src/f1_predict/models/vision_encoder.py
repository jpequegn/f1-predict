"""Vision encoder using pretrained ResNet for multi-modal learning."""

from typing import Literal

import torch
from torch import nn
from torchvision import models


class VisionEncoder(nn.Module):
    """ResNet-based vision encoder for extracting image features.

    Uses a pretrained ResNet backbone (18 or 50) with optional fine-tuning.
    Extracts features from the penultimate layer for use in multi-modal fusion.

    Architecture:
    - ResNet backbone (frozen or trainable)
    - Removes classification head
    - Extracts features from avgpool layer: 512-dim (ResNet-18) or 2048-dim (ResNet-50)
    """

    def __init__(
        self,
        model_name: Literal['resnet18', 'resnet50'] = 'resnet18',
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        """Initialize vision encoder.

        Args:
            model_name: ResNet model architecture ('resnet18' or 'resnet50')
            pretrained: Load ImageNet-pretrained weights
            freeze_backbone: Freeze backbone parameters (don't train)
        """
        super().__init__()

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone

        # Load pretrained ResNet
        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(weights='DEFAULT' if pretrained else None)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        # Remove the classification head (fc layer)
        # We'll use avgpool output as features
        self.feature_extractor = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
            self.backbone.avgpool
        )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Feature tensor of shape (batch_size, feature_dim)
            where feature_dim is 512 for ResNet-18 or 2048 for ResNet-50
        """
        # Extract features: (batch_size, feature_dim, 1, 1)
        features = self.feature_extractor(x)

        # Flatten: (batch_size, feature_dim)
        features = torch.flatten(features, 1)

        return features
