"""Tests for image preprocessing module."""

import numpy as np
from PIL import Image
import pytest
import torch

from f1_predict.models.image_preprocessing import (
    AugmentationConfig,
    F1ImagePreprocessor,
    ImageAugmentor,
    ImageType,
    ImageTypeConfig,
    MultiModalImageProcessor,
)


class TestImageType:
    """Tests for ImageType enum."""

    def test_image_types_exist(self):
        """Test all expected image types exist."""
        assert ImageType.TELEMETRY is not None
        assert ImageType.TRACK_LAYOUT is not None
        assert ImageType.WEATHER is not None
        assert ImageType.RACE_PHOTO is not None
        assert ImageType.GENERIC is not None

    def test_image_type_values(self):
        """Test image type string values."""
        assert ImageType.TELEMETRY.value == "telemetry"
        assert ImageType.TRACK_LAYOUT.value == "track_layout"
        assert ImageType.WEATHER.value == "weather"
        assert ImageType.RACE_PHOTO.value == "race_photo"


class TestAugmentationConfig:
    """Tests for AugmentationConfig dataclass."""

    def test_default_config(self):
        """Test default augmentation configuration."""
        config = AugmentationConfig()
        assert config.rotation_degrees == 15.0
        assert config.horizontal_flip_prob == 0.0
        assert config.gaussian_noise_std == 0.01

    def test_custom_config(self):
        """Test custom augmentation configuration."""
        config = AugmentationConfig(rotation_degrees=30.0, gaussian_noise_std=0.05)
        assert config.rotation_degrees == 30.0
        assert config.gaussian_noise_std == 0.05


class TestImageTypeConfig:
    """Tests for ImageTypeConfig dataclass."""

    def test_default_config(self):
        """Test default image type configuration."""
        config = ImageTypeConfig(image_type=ImageType.GENERIC)
        assert config.target_size == (224, 224)
        assert config.channels == 3

    def test_custom_config(self):
        """Test custom image type configuration."""
        config = ImageTypeConfig(
            image_type=ImageType.TELEMETRY, target_size=(128, 128), channels=1
        )
        assert config.target_size == (128, 128)
        assert config.channels == 1


class TestF1ImagePreprocessor:
    """Tests for F1ImagePreprocessor class."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return F1ImagePreprocessor(image_type=ImageType.GENERIC, training=False)

    @pytest.fixture
    def sample_image(self):
        """Create sample PIL image."""
        return Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )

    def test_init_default(self):
        """Test default initialization."""
        preprocessor = F1ImagePreprocessor()
        assert preprocessor.image_type == ImageType.GENERIC

    def test_init_with_image_type(self):
        """Test initialization with image type."""
        preprocessor = F1ImagePreprocessor(image_type=ImageType.TELEMETRY)
        assert preprocessor.image_type == ImageType.TELEMETRY

    def test_init_training_mode(self):
        """Test training mode initialization."""
        preprocessor = F1ImagePreprocessor(training=True)
        assert preprocessor.training is True

        preprocessor = F1ImagePreprocessor(training=False)
        assert preprocessor.training is False

    def test_call_returns_tensor(self, preprocessor, sample_image):
        """Test preprocessing returns tensor."""
        result = preprocessor(sample_image)
        assert isinstance(result, torch.Tensor)

    def test_call_correct_shape(self, preprocessor, sample_image):
        """Test preprocessed tensor has correct shape."""
        result = preprocessor(sample_image)
        assert result.shape == (3, 224, 224)

    def test_call_normalized_values(self, preprocessor, sample_image):
        """Test tensor values are normalized."""
        result = preprocessor(sample_image)
        # After normalization, values should be roughly in [-3, 3] range
        assert result.min() >= -5
        assert result.max() <= 5

    def test_different_image_types(self, sample_image):
        """Test preprocessing works for all image types."""
        for image_type in ImageType:
            preprocessor = F1ImagePreprocessor(
                image_type=image_type, training=False
            )
            result = preprocessor(sample_image)
            assert result.shape == (3, 224, 224)

    def test_get_transform(self, preprocessor):
        """Test getting transforms."""
        transform = preprocessor.get_transform()
        assert transform is not None

    def test_set_training(self):
        """Test setting training mode."""
        preprocessor = F1ImagePreprocessor(training=True)
        assert preprocessor.training is True

        preprocessor.set_training(False)
        assert preprocessor.training is False

    def test_preprocess_grayscale_image(self, preprocessor):
        """Test preprocessing grayscale images."""
        gray_image = Image.fromarray(
            np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        ).convert("RGB")
        result = preprocessor(gray_image)
        assert result.shape == (3, 224, 224)


class TestImageAugmentor:
    """Tests for ImageAugmentor class."""

    @pytest.fixture
    def augmentor(self):
        """Create augmentor instance."""
        return ImageAugmentor(seed=42)

    @pytest.fixture
    def sample_image(self):
        """Create sample PIL image."""
        return Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )

    def test_init_default(self):
        """Test default initialization."""
        augmentor = ImageAugmentor()
        assert augmentor.rng is not None

    def test_init_with_seed(self):
        """Test initialization with seed."""
        augmentor = ImageAugmentor(seed=42)
        assert augmentor.rng is not None

    def test_apply_rain_effect(self, augmentor, sample_image):
        """Test rain effect application."""
        result = augmentor.apply_rain_effect(sample_image, intensity=0.3)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_apply_fog_effect(self, augmentor, sample_image):
        """Test fog effect application."""
        result = augmentor.apply_fog_effect(sample_image, density=0.3)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_apply_time_of_day(self, augmentor, sample_image):
        """Test time of day effect."""
        for time in ["day", "sunset", "night", "overcast"]:
            result = augmentor.apply_time_of_day(sample_image, time=time)
            assert isinstance(result, Image.Image)

    def test_apply_motion_blur(self, augmentor, sample_image):
        """Test motion blur effect."""
        result = augmentor.apply_motion_blur(sample_image, kernel_size=5)
        assert isinstance(result, Image.Image)


class TestMultiModalImageProcessor:
    """Tests for MultiModalImageProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return MultiModalImageProcessor(training=False)

    @pytest.fixture
    def sample_image(self):
        """Create sample PIL image."""
        return Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )

    @pytest.fixture
    def sample_images(self):
        """Create sample image list."""
        return [
            Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            for _ in range(3)
        ]

    def test_init(self):
        """Test processor initialization."""
        processor = MultiModalImageProcessor()
        assert len(processor.processors) > 0

    def test_process_single_image(self, processor, sample_image):
        """Test processing single image."""
        result = processor.process(sample_image, ImageType.TELEMETRY)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_process_batch(self, processor, sample_images):
        """Test processing batch of images."""
        result = processor.process_batch(sample_images)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3

    def test_process_batch_with_types(self, processor, sample_images):
        """Test batch processing with image types."""
        types = [ImageType.TELEMETRY, ImageType.WEATHER, ImageType.TRACK_LAYOUT]
        result = processor.process_batch(sample_images, image_types=types)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3

    def test_set_training(self, processor):
        """Test setting training mode."""
        processor.set_training(True)
        assert processor.training is True

        processor.set_training(False)
        assert processor.training is False


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_preprocessor_with_rgba_image(self):
        """Test preprocessing RGBA images."""
        preprocessor = F1ImagePreprocessor(training=False)
        rgba_image = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8), mode="RGBA"
        ).convert("RGB")
        result = preprocessor(rgba_image)
        assert result.shape == (3, 224, 224)

    def test_preprocessor_with_large_image(self):
        """Test preprocessing large images."""
        preprocessor = F1ImagePreprocessor(training=False)
        large_image = Image.fromarray(
            np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        )
        result = preprocessor(large_image)
        assert result.shape == (3, 224, 224)

    def test_preprocessor_with_small_image(self):
        """Test preprocessing small images."""
        preprocessor = F1ImagePreprocessor(training=False)
        small_image = Image.fromarray(
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        )
        result = preprocessor(small_image)
        assert result.shape == (3, 224, 224)

    def test_augmentor_various_intensities(self):
        """Test augmentation with various intensities."""
        augmentor = ImageAugmentor(seed=42)
        image = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )

        for intensity in [0.0, 0.5, 1.0]:
            result = augmentor.apply_rain_effect(image, intensity=intensity)
            assert isinstance(result, Image.Image)

    def test_custom_config_preprocessor(self):
        """Test preprocessor with custom config."""
        config = ImageTypeConfig(
            image_type=ImageType.GENERIC,
            target_size=(128, 128),
            augmentation=AugmentationConfig(rotation_degrees=0.0),
        )
        preprocessor = F1ImagePreprocessor(config=config, training=False)
        image = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        result = preprocessor(image)
        assert result.shape == (3, 128, 128)
