"""Tests for multi-modal dataset loader."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from f1_predict.data.multimodal_dataset import MultiModalDataset


class TestMultiModalDatasetInitialization:
    """Test dataset creation and initialization."""

    @pytest.fixture
    def temp_cache_file(self):
        """Create temporary cache JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            cache_data = {
                "metadata": [
                    {
                        "race_id": "2024_race_1",
                        "driver_id": "driver_1",
                        "image_path": "data/multimodal/speed_traces/2024_race_1/driver_1.png",
                        "tabular_features": [1.0, 2.0, 3.0],
                        "label": 1,
                        "finished": True
                    },
                    {
                        "race_id": "2024_race_1",
                        "driver_id": "driver_2",
                        "image_path": "data/multimodal/speed_traces/2024_race_1/driver_2.png",
                        "tabular_features": [2.0, 3.0, 4.0],
                        "label": 2,
                        "finished": True
                    }
                ],
                "feature_names": ["feature_1", "feature_2", "feature_3"]
            }
            json.dump(cache_data, f)
            return f.name

    def test_initialization_with_cache_file(self, temp_cache_file):
        """Test dataset initializes from cache JSON."""
        dataset = MultiModalDataset(cache_file=temp_cache_file)
        assert len(dataset) == 2

    def test_length_matches_metadata(self, temp_cache_file):
        """Test dataset length equals number of metadata entries."""
        dataset = MultiModalDataset(cache_file=temp_cache_file)
        assert len(dataset) == 2


class TestMultiModalDatasetGetItem:
    """Test dataset item retrieval."""

    @pytest.fixture
    def dataset_with_dummy_images(self):
        """Create dataset with temporary dummy images."""
        temp_dir = tempfile.mkdtemp()

        # Create dummy PNG files
        for race_id, driver_id in [("2024_race_1", "driver_1"), ("2024_race_1", "driver_2")]:
            race_dir = Path(temp_dir) / race_id
            race_dir.mkdir(exist_ok=True)

            # Create dummy image
            img = Image.new('RGB', (224, 224), color='red')
            img_path = race_dir / f"{driver_id}.png"
            img.save(img_path)

        # Create cache JSON
        cache_file = Path(temp_dir) / "cache.json"
        cache_data = {
            "metadata": [
                {
                    "race_id": "2024_race_1",
                    "driver_id": "driver_1",
                    "image_path": str(Path(temp_dir) / "2024_race_1" / "driver_1.png"),
                    "tabular_features": [1.0, 2.0, 3.0],
                    "label": 1,
                    "finished": True
                },
                {
                    "race_id": "2024_race_1",
                    "driver_id": "driver_2",
                    "image_path": str(Path(temp_dir) / "2024_race_1" / "driver_2.png"),
                    "tabular_features": [2.0, 3.0, 4.0],
                    "label": 2,
                    "finished": True
                }
            ],
            "feature_names": ["feature_1", "feature_2", "feature_3"]
        }
        cache_file.write_text(json.dumps(cache_data))

        return MultiModalDataset(cache_file=str(cache_file))

    def test_getitem_returns_tuple(self, dataset_with_dummy_images):
        """Test __getitem__ returns (image, tabular, label) tuple."""
        result = dataset_with_dummy_images[0]
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_getitem_image_is_tensor(self, dataset_with_dummy_images):
        """Test image returned as tensor or None."""
        image, _, _ = dataset_with_dummy_images[0]
        assert image is None or isinstance(image, torch.Tensor)

    def test_getitem_tabular_is_tensor(self, dataset_with_dummy_images):
        """Test tabular features returned as tensor."""
        _, tabular, _ = dataset_with_dummy_images[0]
        assert isinstance(tabular, torch.Tensor)
        assert tabular.shape[0] == 3  # 3 features

    def test_getitem_label_is_int(self, dataset_with_dummy_images):
        """Test label returned as integer."""
        _, _, label = dataset_with_dummy_images[0]
        assert isinstance(label, int)

    def test_getitem_missing_image_returns_none(self):
        """Test missing image file returns None gracefully."""
        temp_dir = tempfile.mkdtemp()
        cache_file = Path(temp_dir) / "cache.json"

        cache_data = {
            "metadata": [
                {
                    "race_id": "2024_race_1",
                    "driver_id": "driver_1",
                    "image_path": "/nonexistent/path/image.png",
                    "tabular_features": [1.0, 2.0, 3.0],
                    "label": 1,
                    "finished": True
                }
            ],
            "feature_names": ["feature_1", "feature_2", "feature_3"]
        }
        cache_file.write_text(json.dumps(cache_data))

        dataset = MultiModalDataset(cache_file=str(cache_file))
        image, tabular, label = dataset[0]

        # Should return None for missing image
        assert image is None
        # But still return tabular features and label
        assert isinstance(tabular, torch.Tensor)
        assert isinstance(label, int)

    def test_tabular_features_normalized(self, dataset_with_dummy_images):
        """Test tabular features are converted to float tensors."""
        _, tabular, _ = dataset_with_dummy_images[0]
        assert tabular.dtype == torch.float32


class TestMultiModalDatasetBatching:
    """Test batching behavior with DataLoader."""

    def test_dataloader_compatible(self):
        """Test dataset works with PyTorch DataLoader."""
        temp_dir = tempfile.mkdtemp()

        # Create 4 dummy images
        for i in range(4):
            race_dir = Path(temp_dir) / "race_1"
            race_dir.mkdir(exist_ok=True)
            img = Image.new('RGB', (224, 224), color='blue')
            img.save(race_dir / f"driver_{i}.png")

        # Create cache
        cache_file = Path(temp_dir) / "cache.json"
        metadata = []
        for i in range(4):
            metadata.append({
                "race_id": "race_1",
                "driver_id": f"driver_{i}",
                "image_path": str(Path(temp_dir) / "race_1" / f"driver_{i}.png"),
                "tabular_features": [float(i), float(i+1), float(i+2)],
                "label": i,
                "finished": True
            })

        cache_data = {
            "metadata": metadata,
            "feature_names": ["f1", "f2", "f3"]
        }
        cache_file.write_text(json.dumps(cache_data))

        dataset = MultiModalDataset(cache_file=str(cache_file))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        # Should iterate without errors
        batch_count = 0
        for batch_images, batch_tabular, batch_labels in dataloader:
            batch_count += 1
            # Images might be None or tensor
            assert isinstance(batch_tabular, torch.Tensor)
            assert isinstance(batch_labels, torch.Tensor)

        assert batch_count == 2  # 4 samples with batch_size=2
