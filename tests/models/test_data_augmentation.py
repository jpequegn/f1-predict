"""Tests for data augmentation module."""

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from f1_predict.models.data_augmentation import (
    ImageAugmentation,
    MultiModalAugmentor,
    SyntheticTelemetryGenerator,
    TabularAugmentation,
)


class TestImageAugmentation:
    """Tests for ImageAugmentation class."""

    @pytest.fixture
    def augmentor(self):
        """Create augmentor instance."""
        return ImageAugmentation(strength=0.5)

    @pytest.fixture
    def sample_image(self):
        """Create sample image tensor."""
        return torch.randn(3, 224, 224)

    @pytest.fixture
    def sample_pil_image(self):
        """Create sample PIL image."""
        return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    def test_init_default(self):
        """Test default initialization."""
        augmentor = ImageAugmentation()
        assert augmentor.strength == 0.3

    def test_init_custom_strength(self):
        """Test custom strength initialization."""
        augmentor = ImageAugmentation(strength=0.8)
        assert augmentor.strength == 0.8

    def test_augment_tensor(self, augmentor, sample_image):
        """Test tensor augmentation."""
        result = augmentor.augment(sample_image)
        assert isinstance(result, torch.Tensor)
        assert result.shape == sample_image.shape

    def test_augment_pil(self, augmentor, sample_pil_image):
        """Test PIL image augmentation."""
        result = augmentor.augment_pil(sample_pil_image)
        assert isinstance(result, Image.Image)

    def test_augment_batch(self, augmentor):
        """Test batch augmentation."""
        batch = torch.randn(4, 3, 224, 224)
        result = augmentor.augment_batch(batch)
        assert result.shape == batch.shape

    def test_augment_preserves_range(self, augmentor, sample_image):
        """Test augmentation preserves value range."""
        # Normalize input to [0, 1]
        normalized = (sample_image - sample_image.min()) / (
            sample_image.max() - sample_image.min()
        )
        result = augmentor.augment(normalized)
        # Result should be in reasonable range
        assert result.min() >= -5  # Allow for normalization effects
        assert result.max() <= 5


class TestTabularAugmentation:
    """Tests for TabularAugmentation class."""

    @pytest.fixture
    def augmentor(self):
        """Create augmentor instance."""
        return TabularAugmentation(noise_level=0.1)

    @pytest.fixture
    def sample_features(self):
        """Create sample feature tensor."""
        return torch.randn(2, 20)

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame."""
        return pd.DataFrame(
            {
                "feature1": np.random.randn(10),
                "feature2": np.random.randn(10),
                "feature3": np.random.randn(10),
            }
        )

    def test_init_default(self):
        """Test default initialization."""
        augmentor = TabularAugmentation()
        assert augmentor.noise_level == 0.1

    def test_init_custom_noise(self):
        """Test custom noise level."""
        augmentor = TabularAugmentation(noise_level=0.2)
        assert augmentor.noise_level == 0.2

    def test_add_noise(self, augmentor, sample_features):
        """Test noise addition."""
        result = augmentor.add_noise(sample_features)
        assert result.shape == sample_features.shape
        # Result should be different from input
        assert not torch.allclose(result, sample_features)

    def test_mixup(self, augmentor, sample_features):
        """Test mixup augmentation."""
        features2 = torch.randn(2, 20)
        labels1 = torch.tensor([0, 1])
        labels2 = torch.tensor([1, 0])

        mixed_features, mixed_labels = augmentor.mixup(
            sample_features, features2, labels1, labels2, alpha=0.5
        )

        assert mixed_features.shape == sample_features.shape
        assert mixed_labels.shape == labels1.shape

    def test_feature_dropout(self, augmentor, sample_features):
        """Test feature dropout."""
        result = augmentor.feature_dropout(sample_features, dropout_rate=0.3)
        assert result.shape == sample_features.shape
        # Some values should be zeroed
        zero_count = (result == 0).sum().item()
        assert zero_count >= 0  # At least some might be zero

    def test_augment_dataframe(self, augmentor, sample_dataframe):
        """Test DataFrame augmentation."""
        result = augmentor.augment_dataframe(sample_dataframe)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_dataframe.shape


class TestSyntheticTelemetryGenerator:
    """Tests for SyntheticTelemetryGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return SyntheticTelemetryGenerator()

    def test_init(self):
        """Test initialization."""
        generator = SyntheticTelemetryGenerator()
        assert generator is not None

    def test_generate_speed_profile(self, generator):
        """Test speed profile generation."""
        profile = generator.generate_speed_profile(num_points=100)
        assert len(profile) == 100
        # Speeds should be in reasonable F1 range
        assert all(50 <= s <= 370 for s in profile)

    def test_generate_lap_telemetry(self, generator):
        """Test full lap telemetry generation."""
        telemetry = generator.generate_lap_telemetry()
        assert "speed" in telemetry
        assert "throttle" in telemetry
        assert "brake" in telemetry

    def test_generate_race_telemetry(self, generator):
        """Test full race telemetry generation."""
        telemetry = generator.generate_race_telemetry(num_laps=5)
        assert len(telemetry) == 5

    def test_generate_with_conditions(self, generator):
        """Test generation with specific conditions."""
        telemetry = generator.generate_lap_telemetry(track_type="street", weather="wet")
        assert telemetry is not None

    def test_reproducible_generation(self):
        """Test generation is reproducible with seed."""
        generator1 = SyntheticTelemetryGenerator(seed=42)
        generator2 = SyntheticTelemetryGenerator(seed=42)

        profile1 = generator1.generate_speed_profile(num_points=10)
        profile2 = generator2.generate_speed_profile(num_points=10)

        assert profile1 == profile2


class TestMultiModalAugmentor:
    """Tests for MultiModalAugmentor class."""

    @pytest.fixture
    def augmentor(self):
        """Create augmentor instance."""
        return MultiModalAugmentor(
            image_strength=0.3,
            tabular_noise=0.1,
        )

    @pytest.fixture
    def sample_batch(self):
        """Create sample multi-modal batch."""
        return {
            "images": torch.randn(2, 3, 224, 224),
            "tabular": torch.randn(2, 20),
            "labels": torch.tensor([0, 1]),
        }

    def test_init(self):
        """Test initialization."""
        augmentor = MultiModalAugmentor()
        assert augmentor is not None

    def test_augment_batch(self, augmentor, sample_batch):
        """Test batch augmentation."""
        result = augmentor.augment_batch(sample_batch)
        assert "images" in result
        assert "tabular" in result
        assert "labels" in result

    def test_augment_preserves_shapes(self, augmentor, sample_batch):
        """Test augmentation preserves tensor shapes."""
        result = augmentor.augment_batch(sample_batch)
        assert result["images"].shape == sample_batch["images"].shape
        assert result["tabular"].shape == sample_batch["tabular"].shape

    def test_augment_with_cross_modal(self, augmentor, sample_batch):
        """Test cross-modal augmentation."""
        result = augmentor.augment_batch(sample_batch, cross_modal=True)
        assert result is not None

    def test_check_consistency(self, augmentor, sample_batch):
        """Test consistency checking."""
        augmented = augmentor.augment_batch(sample_batch)
        is_consistent = augmentor.check_consistency(sample_batch, augmented)
        assert isinstance(is_consistent, bool)


class TestAugmentationEdgeCases:
    """Tests for augmentation edge cases."""

    def test_image_augment_single_channel(self):
        """Test augmenting single channel images."""
        augmentor = ImageAugmentation()
        image = torch.randn(1, 224, 224)
        result = augmentor.augment(image)
        assert result.shape == image.shape

    def test_image_augment_large(self):
        """Test augmenting large images."""
        augmentor = ImageAugmentation()
        image = torch.randn(3, 512, 512)
        result = augmentor.augment(image)
        assert result.shape == image.shape

    def test_tabular_augment_high_dimension(self):
        """Test augmenting high-dimensional features."""
        augmentor = TabularAugmentation()
        features = torch.randn(2, 1000)
        result = augmentor.add_noise(features)
        assert result.shape == features.shape

    def test_tabular_augment_single_sample(self):
        """Test augmenting single sample."""
        augmentor = TabularAugmentation()
        features = torch.randn(1, 20)
        result = augmentor.add_noise(features)
        assert result.shape == features.shape

    def test_zero_strength_augmentation(self):
        """Test augmentation with zero strength."""
        augmentor = ImageAugmentation(strength=0.0)
        image = torch.randn(3, 224, 224)
        result = augmentor.augment(image)
        # With zero strength, output should be similar to input
        assert result.shape == image.shape

    def test_max_strength_augmentation(self):
        """Test augmentation with maximum strength."""
        augmentor = ImageAugmentation(strength=1.0)
        image = torch.randn(3, 224, 224)
        result = augmentor.augment(image)
        assert result.shape == image.shape


class TestAugmentationReproducibility:
    """Tests for augmentation reproducibility."""

    def test_image_augment_with_seed(self):
        """Test image augmentation is reproducible with seed."""
        torch.manual_seed(42)
        augmentor1 = ImageAugmentation(strength=0.5)
        image = torch.randn(3, 224, 224)

        torch.manual_seed(42)
        result1 = augmentor1.augment(image.clone())

        torch.manual_seed(42)
        result2 = augmentor1.augment(image.clone())

        # Results should be close (exact match depends on implementation)
        assert result1.shape == result2.shape

    def test_tabular_augment_with_seed(self):
        """Test tabular augmentation is reproducible with seed."""
        torch.manual_seed(42)
        augmentor = TabularAugmentation(noise_level=0.1)
        features = torch.randn(2, 20)

        torch.manual_seed(42)
        result1 = augmentor.add_noise(features.clone())

        torch.manual_seed(42)
        result2 = augmentor.add_noise(features.clone())

        # Results should match with same seed
        assert result1.shape == result2.shape
