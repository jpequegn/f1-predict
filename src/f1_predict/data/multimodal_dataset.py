"""Multi-modal dataset for combining images and tabular features."""

import json
from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MultiModalDataset(Dataset):
    """PyTorch Dataset combining image and tabular features.

    Lazily loads images from disk and tabular features from cache.
    Handles missing images gracefully by returning None.
    """

    def __init__(
        self,
        cache_file: str,
        image_transform: Optional[transforms.Compose] = None
    ):
        """Initialize dataset from cache file.

        Args:
            cache_file: Path to JSON file with metadata
            image_transform: Optional image transformations
        """
        self.cache_file = Path(cache_file)

        # Load metadata
        with open(self.cache_file) as f:
            cache_data = json.load(f)

        self.metadata = cache_data['metadata']
        self.feature_names = cache_data.get('feature_names', [])

        # Default image transforms
        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.image_transform = image_transform

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[Optional[torch.Tensor], torch.Tensor, int]:
        """Get sample at index.

        Returns:
            Tuple of (image_tensor or None, tabular_features_tensor, label)
        """
        entry = self.metadata[idx]

        # Load image (or return None if missing)
        image_tensor = self._load_image(entry['image_path'])

        # Load and convert tabular features
        tabular_features = torch.tensor(
            entry['tabular_features'],
            dtype=torch.float32
        )

        # Get label
        label = entry['label']

        return image_tensor, tabular_features, label

    def _load_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Load image from disk, return None if missing.

        Args:
            image_path: Path to image file

        Returns:
            Tensor of shape (3, 224, 224) or None if file not found
        """
        image_file = Path(image_path)

        if not image_file.exists():
            return None

        try:
            img = Image.open(image_file).convert('RGB')
            return self.image_transform(img)
        except Exception:
            # If image loading fails, return None
            return None
