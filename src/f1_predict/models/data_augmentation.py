"""Data augmentation system for multi-modal F1 prediction.

Provides comprehensive augmentation for both image and tabular data,
including synthetic telemetry generation and cross-modal consistency.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Optional

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)


@dataclass
class AugmentationParams:
    """Parameters for data augmentation."""

    # Image augmentation
    image_rotation_max: float = 15.0
    image_flip_horizontal: bool = False
    image_flip_vertical: bool = False
    image_brightness_range: tuple[float, float] = (0.9, 1.1)
    image_contrast_range: tuple[float, float] = (0.9, 1.1)
    image_noise_std: float = 0.01
    image_blur_prob: float = 0.1
    image_cutout_prob: float = 0.0
    image_cutout_size: float = 0.1  # Fraction of image size

    # Tabular augmentation
    tabular_noise_std: float = 0.05
    tabular_dropout_prob: float = 0.1
    tabular_mixup_alpha: float = 0.2

    # Multi-modal augmentation
    modality_dropout_prob: float = 0.1  # Probability to drop entire modality
    cross_modal_mix_prob: float = 0.0  # Probability to mix features across samples


@dataclass
class SyntheticTelemetryParams:
    """Parameters for synthetic telemetry generation."""

    # Base lap time parameters
    base_lap_time_mean: float = 90.0
    base_lap_time_std: float = 5.0

    # Variation parameters
    lap_to_lap_noise: float = 0.3
    tire_degradation_rate: float = 0.02
    fuel_effect_per_lap: float = 0.03
    traffic_effect_prob: float = 0.1
    traffic_effect_magnitude: float = 1.0

    # Pit stop parameters
    pit_stop_count_range: tuple[int, int] = (1, 3)
    pit_stop_loss_mean: float = 25.0
    pit_stop_loss_std: float = 2.0

    # Weather effects
    rain_lap_time_increase: float = 5.0
    wet_variability_increase: float = 2.0


class ImageAugmentation:
    """Augmentation transforms for F1 images."""

    def __init__(
        self,
        params: Optional[AugmentationParams] = None,
        seed: Optional[int] = None,
    ):
        """Initialize image augmentation.

        Args:
            params: Augmentation parameters
            seed: Random seed
        """
        self.params = params or AugmentationParams()
        self.rng = np.random.default_rng(seed)

        # Build transform pipeline
        self.transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        """Build torchvision transform pipeline."""
        transform_list = []

        # Geometric transforms
        if self.params.image_rotation_max > 0:
            transform_list.append(
                transforms.RandomRotation(self.params.image_rotation_max)
            )

        if self.params.image_flip_horizontal:
            transform_list.append(transforms.RandomHorizontalFlip(0.5))

        if self.params.image_flip_vertical:
            transform_list.append(transforms.RandomVerticalFlip(0.5))

        # Color transforms
        brightness = self.params.image_brightness_range[1] - 1
        contrast = self.params.image_contrast_range[1] - 1
        transform_list.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
            )
        )

        # Blur
        if self.params.image_blur_prob > 0:
            transform_list.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3)],
                    p=self.params.image_blur_prob,
                )
            )

        # To tensor
        transform_list.append(transforms.ToTensor())

        # Add noise
        if self.params.image_noise_std > 0:
            transform_list.append(
                AddGaussianNoise(std=self.params.image_noise_std)
            )

        # Cutout
        if self.params.image_cutout_prob > 0:
            transform_list.append(
                RandomCutout(
                    prob=self.params.image_cutout_prob,
                    size_fraction=self.params.image_cutout_size,
                )
            )

        # Normalize to ImageNet stats
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

        return transforms.Compose(transform_list)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply augmentation to image.

        Args:
            image: PIL Image

        Returns:
            Augmented tensor
        """
        result = self.transform(image)
        assert isinstance(result, torch.Tensor)
        return result


class AddGaussianNoise:
    """Add Gaussian noise to tensor."""

    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std + self.mean


class RandomCutout:
    """Apply random cutout (erasing) to tensor."""

    def __init__(self, prob: float = 0.5, size_fraction: float = 0.1):
        self.prob = prob
        self.size_fraction = size_fraction

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.prob:
            return tensor

        _, h, w = tensor.shape
        size_h = int(h * self.size_fraction)
        size_w = int(w * self.size_fraction)

        y = torch.randint(0, h - size_h, (1,)).item()
        x = torch.randint(0, w - size_w, (1,)).item()

        tensor[:, y : y + size_h, x : x + size_w] = 0
        return tensor


class TabularAugmentation:
    """Augmentation for tabular features."""

    def __init__(
        self,
        params: Optional[AugmentationParams] = None,
        seed: Optional[int] = None,
    ):
        """Initialize tabular augmentation.

        Args:
            params: Augmentation parameters
            seed: Random seed
        """
        self.params = params or AugmentationParams()
        self.rng = np.random.default_rng(seed)

    def add_noise(self, features: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to features.

        Args:
            features: Input features (batch_size, num_features)

        Returns:
            Noisy features
        """
        noise = torch.randn_like(features) * self.params.tabular_noise_std
        return features + noise

    def feature_dropout(self, features: torch.Tensor) -> torch.Tensor:
        """Randomly zero out some features.

        Args:
            features: Input features

        Returns:
            Features with dropout applied
        """
        mask = torch.rand_like(features) > self.params.tabular_dropout_prob
        return features * mask

    def mixup(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Apply mixup augmentation.

        Args:
            features: Input features (batch_size, num_features)
            labels: Input labels (batch_size,)

        Returns:
            Tuple of (mixed_features, mixed_labels, lambda)
        """
        batch_size = features.shape[0]

        # Sample lambda from Beta distribution
        lam = self.rng.beta(
            self.params.tabular_mixup_alpha,
            self.params.tabular_mixup_alpha,
        )

        # Random permutation
        perm = torch.randperm(batch_size)

        # Mix features
        mixed_features = lam * features + (1 - lam) * features[perm]

        # Mix labels (for soft labels)
        mixed_labels = lam * labels.float() + (1 - lam) * labels[perm].float()

        return mixed_features, mixed_labels, lam

    def __call__(
        self,
        features: torch.Tensor,
        augment_noise: bool = True,
        augment_dropout: bool = True,
    ) -> torch.Tensor:
        """Apply augmentation to features.

        Args:
            features: Input features
            augment_noise: Whether to add noise
            augment_dropout: Whether to apply dropout

        Returns:
            Augmented features
        """
        if augment_noise:
            features = self.add_noise(features)
        if augment_dropout:
            features = self.feature_dropout(features)
        return features


class SyntheticTelemetryGenerator:
    """Generate synthetic telemetry data for training.

    Creates realistic-looking telemetry sequences when real data
    is unavailable or for data augmentation.
    """

    def __init__(
        self,
        params: Optional[SyntheticTelemetryParams] = None,
        seed: Optional[int] = None,
    ):
        """Initialize generator.

        Args:
            params: Generation parameters
            seed: Random seed
        """
        self.params = params or SyntheticTelemetryParams()
        self.rng = np.random.default_rng(seed)

    def generate_race_telemetry(
        self,
        num_laps: int = 50,
        num_drivers: int = 20,
    ) -> dict[str, Any]:
        """Generate synthetic race telemetry.

        Args:
            num_laps: Number of laps in race
            num_drivers: Number of drivers

        Returns:
            Dictionary with telemetry data for all drivers
        """
        drivers = [f"driver_{i:02d}" for i in range(1, num_drivers + 1)]
        telemetry: dict[str, Any] = {"drivers": {}}

        # Generate base pace for each driver (skill level)
        base_paces = self.rng.normal(
            self.params.base_lap_time_mean,
            self.params.base_lap_time_std,
            num_drivers,
        )

        for driver_idx, driver_id in enumerate(drivers):
            driver_data = self._generate_driver_telemetry(
                num_laps=num_laps,
                base_pace=base_paces[driver_idx],
            )
            telemetry["drivers"][driver_id] = driver_data

        return telemetry

    def _generate_driver_telemetry(
        self,
        num_laps: int,
        base_pace: float,
    ) -> dict[str, Any]:
        """Generate telemetry for a single driver.

        Args:
            num_laps: Number of laps
            base_pace: Driver's base lap time

        Returns:
            Dictionary with driver telemetry
        """
        # Determine pit stops
        num_stops = self.rng.integers(
            self.params.pit_stop_count_range[0],
            self.params.pit_stop_count_range[1] + 1,
        )
        pit_laps = sorted(
            self.rng.choice(range(10, num_laps - 5), num_stops, replace=False)
        )

        # Generate lap times
        lap_times = []
        tire_age = 0
        current_compound = self.rng.choice(["soft", "medium", "hard"])
        compounds = [current_compound]

        for lap in range(num_laps):
            # Check for pit stop
            if lap in pit_laps:
                # Add pit stop loss
                pit_loss = self.rng.normal(
                    self.params.pit_stop_loss_mean,
                    self.params.pit_stop_loss_std,
                )
                lap_times.append(base_pace + pit_loss)
                tire_age = 0
                current_compound = self.rng.choice(["soft", "medium", "hard"])
                compounds.append(current_compound)
                continue

            # Calculate lap time
            lap_time = base_pace

            # Tire degradation
            degradation = self.params.tire_degradation_rate * tire_age
            lap_time += degradation

            # Fuel effect (lighter car = faster)
            fuel_benefit = self.params.fuel_effect_per_lap * lap
            lap_time -= fuel_benefit * 0.5  # Partial benefit

            # Random variation
            noise = self.rng.normal(0, self.params.lap_to_lap_noise)
            lap_time += noise

            # Traffic effect
            if self.rng.random() < self.params.traffic_effect_prob:
                lap_time += self.rng.uniform(0, self.params.traffic_effect_magnitude)

            lap_times.append(lap_time)
            tire_age += 1

        return {
            "lap_times": lap_times,
            "pit_laps": list(pit_laps),
            "compounds": compounds,
            "base_pace": base_pace,
        }

    def generate_qualifying_times(
        self,
        num_drivers: int = 20,
        num_sessions: int = 3,
    ) -> dict[str, list[Optional[float]]]:
        """Generate synthetic qualifying times.

        Args:
            num_drivers: Number of drivers
            num_sessions: Number of qualifying sessions (Q1, Q2, Q3)

        Returns:
            Dictionary mapping driver_id to list of session times
        """
        drivers = [f"driver_{i:02d}" for i in range(1, num_drivers + 1)]

        # Base pace order (fastest to slowest)
        base_paces = np.sort(
            self.rng.normal(
                self.params.base_lap_time_mean,
                self.params.base_lap_time_std,
                num_drivers,
            )
        )

        qualifying: dict[str, list[Optional[float]]] = {}

        for driver_idx, driver_id in enumerate(drivers):
            times: list[Optional[float]] = []
            base = base_paces[driver_idx]

            for session in range(num_sessions):
                # Eliminated drivers don't set times in later sessions
                if session == 1 and driver_idx >= 15 or session == 2 and driver_idx >= 10:
                    times.append(None)
                else:
                    # Add session-specific improvement (Q3 fastest)
                    improvement = session * 0.2
                    noise = self.rng.normal(0, 0.15)
                    times.append(base - improvement + noise)

            qualifying[driver_id] = times

        return qualifying


class MultiModalAugmentor:
    """Combined augmentation for multi-modal data.

    Handles synchronized augmentation of images and tabular data,
    including cross-modal consistency and modality dropout.
    """

    def __init__(
        self,
        params: Optional[AugmentationParams] = None,
        seed: Optional[int] = None,
    ):
        """Initialize multi-modal augmentor.

        Args:
            params: Augmentation parameters
            seed: Random seed
        """
        self.params = params or AugmentationParams()
        self.rng = np.random.default_rng(seed)

        self.image_aug = ImageAugmentation(params, seed)
        self.tabular_aug = TabularAugmentation(params, seed)

    def augment(
        self,
        image: Optional[Image.Image],
        tabular: torch.Tensor,
        training: bool = True,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Augment multi-modal sample.

        Args:
            image: Optional PIL image
            tabular: Tabular features tensor
            training: Whether in training mode

        Returns:
            Tuple of (augmented_image, augmented_tabular)
        """
        if not training:
            # No augmentation during eval
            if image is not None:
                # Just basic preprocessing
                basic_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
                image_tensor = basic_transform(image)
                assert isinstance(image_tensor, torch.Tensor)
                return image_tensor, tabular
            return None, tabular

        # Modality dropout
        if self.rng.random() < self.params.modality_dropout_prob:
            image = None

        # Augment each modality
        image_out: Optional[torch.Tensor] = None
        if image is not None:
            image_out = self.image_aug(image)

        tabular_out = self.tabular_aug(tabular)

        return image_out, tabular_out

    def augment_batch(
        self,
        images: list[Optional[Image.Image]],
        tabular: torch.Tensor,
        training: bool = True,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Augment batch of multi-modal samples.

        Args:
            images: List of optional PIL images
            tabular: Tabular features tensor (batch_size, num_features)
            training: Whether in training mode

        Returns:
            Tuple of (stacked_images or None, augmented_tabular)
        """
        augmented_images = []
        has_images = False

        for i, (img, tab) in enumerate(zip(images, tabular)):
            aug_img, aug_tab = self.augment(img, tab.unsqueeze(0), training)
            tabular[i] = aug_tab.squeeze(0)

            if aug_img is not None:
                augmented_images.append(aug_img)
                has_images = True
            else:
                # Placeholder zeros
                augmented_images.append(torch.zeros(3, 224, 224))

        if has_images:
            return torch.stack(augmented_images), tabular
        return None, tabular


class CrossModalConsistencyChecker:
    """Check and enforce consistency between modalities.

    Ensures that image and tabular data are consistent,
    and can impute missing values across modalities.
    """

    def __init__(
        self,
        feature_names: Optional[list[str]] = None,
    ):
        """Initialize consistency checker.

        Args:
            feature_names: Names of tabular features
        """
        self.feature_names = feature_names or []

    def check_consistency(
        self,
        image_metadata: dict[str, Any],
        tabular_features: dict[str, float],
    ) -> tuple[bool, list[str]]:
        """Check consistency between image metadata and tabular features.

        Args:
            image_metadata: Metadata from image (e.g., weather condition from weather image)
            tabular_features: Tabular features dictionary

        Returns:
            Tuple of (is_consistent, list of inconsistencies)
        """
        inconsistencies = []

        # Example checks
        if "temperature" in image_metadata and "temp" in tabular_features:
            img_temp = image_metadata["temperature"]
            tab_temp = tabular_features["temp"]
            if abs(img_temp - tab_temp) > 10:
                inconsistencies.append(
                    f"Temperature mismatch: image={img_temp}, tabular={tab_temp}"
                )

        if "weather_condition" in image_metadata:
            img_condition = image_metadata["weather_condition"]
            # Check if rain in image matches humidity in tabular
            if img_condition == "rain" and tabular_features.get("humidity", 0) < 50:
                inconsistencies.append(
                    "Rain condition in image but low humidity in tabular"
                )

        return len(inconsistencies) == 0, inconsistencies

    def impute_missing(
        self,
        image_features: Optional[torch.Tensor],
        tabular_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Impute missing values using cross-modal information.

        Args:
            image_features: Image features (may be None or have NaN)
            tabular_features: Tabular features (may have NaN)

        Returns:
            Tuple of (imputed_image_features, imputed_tabular_features)
        """
        # Simple mean imputation for tabular
        if torch.isnan(tabular_features).any():
            # Replace NaN with column mean
            for i in range(tabular_features.shape[-1]):
                col = tabular_features[..., i]
                mask = torch.isnan(col)
                if mask.any():
                    mean_val = col[~mask].mean() if (~mask).any() else 0.0
                    col[mask] = mean_val

        # If image features missing, create zero vector
        if image_features is None:
            image_features = torch.zeros(512)  # Standard image feature dim

        return image_features, tabular_features


@dataclass
class AugmentedDataset:
    """Container for augmented dataset with metadata."""

    images: list[Optional[torch.Tensor]]
    tabular: torch.Tensor
    labels: torch.Tensor
    augmentation_applied: list[str] = field(default_factory=list)
    original_indices: list[int] = field(default_factory=list)
