"""Image preprocessing pipeline for multi-modal F1 prediction.

Provides comprehensive image augmentation, normalization, and preprocessing
for various F1 image types including telemetry plots, track layouts, and
weather condition visualizations.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)


class ImageType(Enum):
    """Types of F1 images supported by the preprocessing pipeline."""

    TELEMETRY = "telemetry"  # Speed traces, tire degradation plots
    TRACK_LAYOUT = "track_layout"  # Circuit maps and layouts
    WEATHER = "weather"  # Weather condition visualizations
    RACE_PHOTO = "race_photo"  # Actual race photos (if available)
    GENERIC = "generic"  # Default for unknown image types


@dataclass
class AugmentationConfig:
    """Configuration for image augmentation parameters."""

    # Geometric transforms
    rotation_degrees: float = 15.0
    horizontal_flip_prob: float = 0.0  # Usually disabled for F1 plots
    vertical_flip_prob: float = 0.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    translate_range: tuple[float, float] = (0.05, 0.05)

    # Color transforms
    brightness_range: tuple[float, float] = (0.9, 1.1)
    contrast_range: tuple[float, float] = (0.9, 1.1)
    saturation_range: tuple[float, float] = (0.9, 1.1)
    hue_range: float = 0.02

    # Noise and blur
    gaussian_noise_std: float = 0.01
    gaussian_blur_prob: float = 0.1
    gaussian_blur_kernel: int = 3

    # Dropout/cutout
    random_erasing_prob: float = 0.0
    random_erasing_scale: tuple[float, float] = (0.02, 0.1)

    # Image type specific
    preserve_aspect_ratio: bool = True
    normalize_to_imagenet: bool = True


@dataclass
class ImageTypeConfig:
    """Image type-specific preprocessing configuration."""

    image_type: ImageType
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    target_size: tuple[int, int] = (224, 224)
    channels: int = 3
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)  # ImageNet


# Default configurations for each image type
TELEMETRY_CONFIG = ImageTypeConfig(
    image_type=ImageType.TELEMETRY,
    augmentation=AugmentationConfig(
        rotation_degrees=5.0,  # Minimal rotation for plots
        horizontal_flip_prob=0.0,  # Never flip telemetry
        brightness_range=(0.95, 1.05),  # Subtle brightness changes
        gaussian_noise_std=0.005,  # Minimal noise
        random_erasing_prob=0.0,  # Don't erase parts of plots
    ),
    target_size=(224, 224),
)

TRACK_LAYOUT_CONFIG = ImageTypeConfig(
    image_type=ImageType.TRACK_LAYOUT,
    augmentation=AugmentationConfig(
        rotation_degrees=0.0,  # No rotation for track maps
        horizontal_flip_prob=0.0,  # Preserve track orientation
        scale_range=(0.95, 1.05),  # Minimal scaling
        brightness_range=(0.9, 1.1),
        gaussian_noise_std=0.01,
    ),
    target_size=(224, 224),
)

WEATHER_CONFIG = ImageTypeConfig(
    image_type=ImageType.WEATHER,
    augmentation=AugmentationConfig(
        rotation_degrees=10.0,
        brightness_range=(0.85, 1.15),  # Weather can vary
        contrast_range=(0.85, 1.15),
        gaussian_noise_std=0.02,  # More noise tolerance
    ),
    target_size=(224, 224),
)

RACE_PHOTO_CONFIG = ImageTypeConfig(
    image_type=ImageType.RACE_PHOTO,
    augmentation=AugmentationConfig(
        rotation_degrees=15.0,
        horizontal_flip_prob=0.5,  # Can flip race photos
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        gaussian_blur_prob=0.2,
        random_erasing_prob=0.1,
    ),
    target_size=(224, 224),
)

# Registry of image type configurations
IMAGE_TYPE_CONFIGS: dict[ImageType, ImageTypeConfig] = {
    ImageType.TELEMETRY: TELEMETRY_CONFIG,
    ImageType.TRACK_LAYOUT: TRACK_LAYOUT_CONFIG,
    ImageType.WEATHER: WEATHER_CONFIG,
    ImageType.RACE_PHOTO: RACE_PHOTO_CONFIG,
    ImageType.GENERIC: ImageTypeConfig(image_type=ImageType.GENERIC),
}


class F1ImagePreprocessor:
    """Comprehensive image preprocessing for F1 multi-modal learning.

    Handles image loading, augmentation, normalization, and conversion
    to tensors for PyTorch models. Supports different preprocessing
    strategies for different image types.
    """

    def __init__(
        self,
        image_type: ImageType = ImageType.GENERIC,
        config: Optional[ImageTypeConfig] = None,
        training: bool = True,
    ):
        """Initialize preprocessor.

        Args:
            image_type: Type of F1 image to preprocess
            config: Optional custom configuration (overrides type defaults)
            training: Whether in training mode (enables augmentation)
        """
        self.image_type = image_type
        self.training = training

        # Get config for image type or use provided
        if config is not None:
            self.config = config
        else:
            self.config = IMAGE_TYPE_CONFIGS.get(
                image_type, IMAGE_TYPE_CONFIGS[ImageType.GENERIC]
            )

        # Build transform pipelines
        self._train_transform = self._build_train_transform()
        self._eval_transform = self._build_eval_transform()

    def _build_train_transform(self) -> transforms.Compose:
        """Build training transform pipeline with augmentation."""
        aug = self.config.augmentation
        transform_list = []

        # Resize with aspect ratio preservation
        if aug.preserve_aspect_ratio:
            transform_list.append(
                transforms.Resize(
                    self.config.target_size[0],
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            )
            transform_list.append(transforms.CenterCrop(self.config.target_size))
        else:
            transform_list.append(transforms.Resize(self.config.target_size))

        # Geometric transforms
        if aug.rotation_degrees > 0 or aug.scale_range != (1.0, 1.0):
            transform_list.append(
                transforms.RandomAffine(
                    degrees=aug.rotation_degrees,
                    translate=aug.translate_range,
                    scale=aug.scale_range,
                )
            )

        # Flips
        if aug.horizontal_flip_prob > 0:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=aug.horizontal_flip_prob)
            )
        if aug.vertical_flip_prob > 0:
            transform_list.append(
                transforms.RandomVerticalFlip(p=aug.vertical_flip_prob)
            )

        # Color transforms - ColorJitter expects brightness as a single float
        # representing max delta from 1.0 (both positive and negative)
        brightness_delta = max(
            1.0 - aug.brightness_range[0], aug.brightness_range[1] - 1.0
        )
        contrast_delta = max(1.0 - aug.contrast_range[0], aug.contrast_range[1] - 1.0)
        saturation_delta = max(
            1.0 - aug.saturation_range[0], aug.saturation_range[1] - 1.0
        )

        transform_list.append(
            transforms.ColorJitter(
                brightness=brightness_delta if brightness_delta > 0 else 0,
                contrast=contrast_delta if contrast_delta > 0 else 0,
                saturation=saturation_delta if saturation_delta > 0 else 0,
                hue=aug.hue_range,
            )
        )

        # Blur
        if aug.gaussian_blur_prob > 0:
            transform_list.append(
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=aug.gaussian_blur_kernel,
                            sigma=(0.1, 2.0),
                        )
                    ],
                    p=aug.gaussian_blur_prob,
                )
            )

        # To tensor
        transform_list.append(transforms.ToTensor())

        # Add Gaussian noise
        if aug.gaussian_noise_std > 0:
            transform_list.append(AddGaussianNoise(std=aug.gaussian_noise_std))

        # Normalize
        if aug.normalize_to_imagenet:
            transform_list.append(
                transforms.Normalize(mean=self.config.mean, std=self.config.std)
            )

        # Random erasing (after normalization)
        if aug.random_erasing_prob > 0:
            transform_list.append(
                transforms.RandomErasing(
                    p=aug.random_erasing_prob,
                    scale=aug.random_erasing_scale,
                )
            )

        return transforms.Compose(transform_list)

    def _build_eval_transform(self) -> transforms.Compose:
        """Build evaluation transform pipeline (no augmentation)."""
        transform_list = []

        # Resize
        if self.config.augmentation.preserve_aspect_ratio:
            transform_list.append(
                transforms.Resize(
                    self.config.target_size[0],
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            )
            transform_list.append(transforms.CenterCrop(self.config.target_size))
        else:
            transform_list.append(transforms.Resize(self.config.target_size))

        # To tensor
        transform_list.append(transforms.ToTensor())

        # Normalize
        if self.config.augmentation.normalize_to_imagenet:
            transform_list.append(
                transforms.Normalize(mean=self.config.mean, std=self.config.std)
            )

        return transforms.Compose(transform_list)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Process image through appropriate transform pipeline.

        Args:
            image: PIL Image to process

        Returns:
            Processed tensor of shape (C, H, W)
        """
        transform = self._train_transform if self.training else self._eval_transform
        result = transform(image)
        # Transform pipeline always returns tensor due to ToTensor() in pipeline
        assert isinstance(result, torch.Tensor)
        return result

    def set_training(self, training: bool) -> None:
        """Set training mode.

        Args:
            training: Whether in training mode
        """
        self.training = training

    def get_transform(self) -> transforms.Compose:
        """Get current transform based on training mode."""
        return self._train_transform if self.training else self._eval_transform


class AddGaussianNoise:
    """Add Gaussian noise to tensor."""

    def __init__(self, mean: float = 0.0, std: float = 0.01):
        """Initialize noise transform.

        Args:
            mean: Mean of Gaussian noise
            std: Standard deviation of Gaussian noise
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add noise to tensor.

        Args:
            tensor: Input tensor

        Returns:
            Noisy tensor
        """
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class ImageAugmentor:
    """Advanced image augmentation for F1 images using PIL.

    Provides additional augmentation techniques not available in torchvision.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize augmentor.

        Args:
            seed: Optional random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

    def apply_rain_effect(
        self, image: Image.Image, intensity: float = 0.3
    ) -> Image.Image:
        """Apply synthetic rain effect to image.

        Useful for augmenting weather-related images.

        Args:
            image: Input PIL Image
            intensity: Rain intensity (0.0 to 1.0)

        Returns:
            Image with rain effect applied
        """
        # Convert to numpy
        img_array = np.array(image).astype(np.float32)

        # Create rain streaks
        h, w = img_array.shape[:2]
        rain = self.rng.random((h, w)) < (intensity * 0.02)

        # Add vertical streaks
        for _ in range(int(w * intensity * 0.1)):
            x = self.rng.integers(0, w)
            length = self.rng.integers(5, 20)
            y_start = self.rng.integers(0, max(1, h - length))
            rain[y_start : y_start + length, x] = True

        # Apply rain as slight brightness reduction + streaks
        rain_mask = rain.astype(np.float32)
        if len(img_array.shape) == 3:
            rain_mask = np.stack([rain_mask] * 3, axis=-1)

        # Darken slightly and add white streaks
        img_array = img_array * (1 - intensity * 0.2)
        img_array = img_array + rain_mask * 200

        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def apply_fog_effect(self, image: Image.Image, density: float = 0.3) -> Image.Image:
        """Apply synthetic fog effect to image.

        Args:
            image: Input PIL Image
            density: Fog density (0.0 to 1.0)

        Returns:
            Image with fog effect applied
        """
        # Create fog layer
        img_array = np.array(image).astype(np.float32)
        fog_color = np.array([220, 220, 220], dtype=np.float32)

        # Blend with fog
        blended = img_array * (1 - density) + fog_color * density

        blended = np.clip(blended, 0, 255).astype(np.uint8)
        return Image.fromarray(blended)

    def apply_time_of_day(self, image: Image.Image, time: str = "day") -> Image.Image:
        """Apply time-of-day color adjustment.

        Args:
            image: Input PIL Image
            time: One of 'day', 'sunset', 'night', 'overcast'

        Returns:
            Color-adjusted image
        """
        enhancer = ImageEnhance.Color(image)

        if time == "sunset":
            # Warm orange tint
            image = enhancer.enhance(1.2)
            r, g, b = image.split()
            r = r.point(lambda x: min(255, int(x * 1.1)))
            b = b.point(lambda x: int(x * 0.9))
            return Image.merge("RGB", (r, g, b))

        if time == "night":
            # Dark blue tint
            image = enhancer.enhance(0.8)
            brightness = ImageEnhance.Brightness(image)
            image = brightness.enhance(0.6)
            r, g, b = image.split()
            r = r.point(lambda x: int(x * 0.8))
            b = b.point(lambda x: min(255, int(x * 1.1)))
            return Image.merge("RGB", (r, g, b))

        if time == "overcast":
            # Desaturated
            image = enhancer.enhance(0.7)
            brightness = ImageEnhance.Brightness(image)
            return brightness.enhance(0.9)

        return image  # day - no change

    def apply_motion_blur(
        self,
        image: Image.Image,
        kernel_size: int = 5,
        angle: float = 0.0,  # noqa: ARG002 - reserved for directional blur
    ) -> Image.Image:
        """Apply motion blur effect.

        Args:
            image: Input PIL Image
            kernel_size: Size of blur kernel
            angle: Angle of motion in degrees (reserved for future directional blur)

        Returns:
            Motion-blurred image
        """
        # Simple horizontal motion blur using PIL
        return image.filter(ImageFilter.BoxBlur(kernel_size // 2))


class MultiModalImageProcessor:
    """Process multiple image types for multi-modal learning.

    Manages different preprocessors for different image modalities
    and handles batch processing.
    """

    def __init__(self, training: bool = True):
        """Initialize multi-modal processor.

        Args:
            training: Whether in training mode
        """
        self.training = training
        self.processors: dict[ImageType, F1ImagePreprocessor] = {}

        # Initialize processors for each type
        for image_type in ImageType:
            self.processors[image_type] = F1ImagePreprocessor(
                image_type=image_type, training=training
            )

    def process(
        self, image: Image.Image, image_type: ImageType = ImageType.GENERIC
    ) -> torch.Tensor:
        """Process a single image.

        Args:
            image: PIL Image to process
            image_type: Type of image

        Returns:
            Processed tensor
        """
        processor = self.processors.get(image_type, self.processors[ImageType.GENERIC])
        return processor(image)

    def process_batch(
        self,
        images: list[Image.Image],
        image_types: Optional[list[ImageType]] = None,
    ) -> torch.Tensor:
        """Process a batch of images.

        Args:
            images: List of PIL Images
            image_types: Optional list of image types (same length as images)

        Returns:
            Batch tensor of shape (N, C, H, W)
        """
        if image_types is None:
            image_types = [ImageType.GENERIC] * len(images)

        processed = []
        for img, img_type in zip(images, image_types):
            tensor = self.process(img, img_type)
            processed.append(tensor)

        return torch.stack(processed)

    def set_training(self, training: bool) -> None:
        """Set training mode for all processors.

        Args:
            training: Whether in training mode
        """
        self.training = training
        for processor in self.processors.values():
            processor.set_training(training)


def denormalize_image(
    tensor: torch.Tensor,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Denormalize image tensor for visualization.

    Args:
        tensor: Normalized tensor of shape (C, H, W) or (N, C, H, W)
        mean: Mean used for normalization
        std: Std used for normalization

    Returns:
        Denormalized tensor with values in [0, 1]
    """
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean_t = mean_t.unsqueeze(0)
        std_t = std_t.unsqueeze(0)

    denorm = tensor * std_t + mean_t
    return torch.clamp(denorm, 0, 1)


def tensor_to_pil(
    tensor: torch.Tensor,
    denormalize: bool = True,
) -> Image.Image:
    """Convert tensor to PIL Image.

    Args:
        tensor: Tensor of shape (C, H, W)
        denormalize: Whether to denormalize ImageNet normalization

    Returns:
        PIL Image
    """
    if denormalize:
        tensor = denormalize_image(tensor)

    # Move to CPU if needed
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()

    # Convert to numpy (H, W, C)
    img_array = tensor.permute(1, 2, 0).numpy()
    img_array = (img_array * 255).astype(np.uint8)

    return Image.fromarray(img_array)
