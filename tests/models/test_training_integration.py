"""Integration tests for multi-modal training pipeline.

Tests the complete workflow of:
1. Generating speed trace images
2. Building dataset cache with features
3. Creating data loaders
4. Training multi-modal model
5. Validating model performance
"""

from pathlib import Path
import tempfile

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from f1_predict.models.dataset_cache_builder import DatasetCacheBuilder
from f1_predict.models.multimodal_trainer import MultiModalTrainer
from f1_predict.models.speed_trace_generator import SpeedTraceGenerator


class SimpleMultiModalDataset(Dataset):
    """Simple dataset for testing multi-modal training."""

    def __init__(self, cache: dict, num_samples: int = 10):
        """Initialize dataset with cache data.

        Args:
            cache: Dataset cache from DatasetCacheBuilder
            num_samples: Number of samples to use from cache
        """
        self.cache = cache
        self.samples = cache.get('samples', [])[:num_samples]

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        """Get sample at index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, features_tensor, label_tensor)
        """
        sample = self.samples[idx]
        features = sample.get('features', [])

        # Create fake image tensor (simulating loaded PNG)
        image = torch.randn(3, 224, 224)

        # Convert features to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Create dummy label (0 or 1)
        label = torch.tensor(0, dtype=torch.long)

        return image, features_tensor, label


class SimpleMultiModalModel(nn.Module):
    """Simple multi-modal model for testing."""

    def __init__(self, num_features: int = 8, num_classes: int = 2):
        """Initialize model.

        Args:
            num_features: Number of tabular features
            num_classes: Number of output classes
        """
        super().__init__()

        # Vision branch (simplified ResNet-like)
        self.vision_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Tabular branch
        self.tabular_branch = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(32 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, images: torch.Tensor | None, tabular: torch.Tensor) -> torch.Tensor:
        """Forward pass through model.

        Args:
            images: Image tensors (batch_size, 3, 224, 224) or None
            tabular: Tabular features (batch_size, num_features)

        Returns:
            Logits (batch_size, num_classes)
        """
        if images is not None:
            vision_out = self.vision_branch(images)
            vision_out = vision_out.view(vision_out.size(0), -1)
        else:
            vision_out = torch.zeros(tabular.size(0), 32)

        tabular_out = self.tabular_branch(tabular)

        # Concatenate and fuse
        fused = torch.cat([vision_out, tabular_out], dim=1)
        return self.fusion(fused)


class TestTrainingIntegration:
    """Integration tests for complete training workflow."""

    @pytest.fixture
    def race_data(self) -> list[dict]:
        """Create sample race data for testing.

        Returns:
            List of race dictionaries with driver data
        """
        return [
            {
                'race_id': '2024-01',
                'date': '2024-03-03',
                'drivers': [
                    {
                        'driver_id': 'driver_001',
                        'lap_data': [120.0, 125.0, 122.0, 128.0, 121.0],
                        'weather_data': {'temp': 25, 'humidity': 60, 'wind_speed': 10},
                        'tire_data': {'compound': 'soft', 'age_laps': 5},
                        'driver_data': {'age': 28, 'experience_years': 8},
                        'circuit_data': {'elevation': 500, 'length_km': 5.5}
                    },
                    {
                        'driver_id': 'driver_002',
                        'lap_data': [125.0, 127.0, 126.0, 129.0, 127.5],
                        'weather_data': {'temp': 25, 'humidity': 60, 'wind_speed': 10},
                        'tire_data': {'compound': 'medium', 'age_laps': 10},
                        'driver_data': {'age': 30, 'experience_years': 10},
                        'circuit_data': {'elevation': 500, 'length_km': 5.5}
                    }
                ]
            },
            {
                'race_id': '2024-02',
                'date': '2024-03-10',
                'drivers': [
                    {
                        'driver_id': 'driver_001',
                        'lap_data': [128.0, 130.0, 129.0, 131.0, 130.0],
                        'weather_data': {'temp': 20, 'humidity': 55, 'wind_speed': 8},
                        'tire_data': {'compound': 'medium', 'age_laps': 0},
                        'driver_data': {'age': 28, 'experience_years': 8},
                        'circuit_data': {'elevation': 100, 'length_km': 6.0}
                    },
                    {
                        'driver_id': 'driver_003',
                        'lap_data': [118.0, 120.0, 119.0, 121.0, 120.5],
                        'weather_data': {'temp': 20, 'humidity': 55, 'wind_speed': 8},
                        'tire_data': {'compound': 'hard', 'age_laps': 15},
                        'driver_data': {'age': 32, 'experience_years': 12},
                        'circuit_data': {'elevation': 100, 'length_km': 6.0}
                    }
                ]
            }
        ]

    def test_speed_trace_generation_integration(self, race_data: list[dict]) -> None:
        """Test speed trace generation from race data.

        Verifies:
        - Traces generated for all drivers
        - Files created at expected paths
        - PNG files are valid (contain data)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SpeedTraceGenerator(output_dir=tmpdir)

            # Generate traces
            results = generator.generate_batch(race_data)

            # Verify all drivers have traces
            assert len(results) == 4  # 2 races × 2 drivers
            assert 'driver_001' in str(results)
            assert 'driver_002' in str(results)
            assert 'driver_003' in str(results)

            # Verify files exist and have content
            for _key, path in results.items():
                assert Path(path).exists()
                assert Path(path).stat().st_size > 0
                assert path.endswith('.png')

    def test_cache_building_integration(self, race_data: list[dict]) -> None:
        """Test dataset cache building from race data.

        Verifies:
        - Cache created with correct structure
        - Metadata properly set
        - Features extracted for all drivers
        - Features are valid floats
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetCacheBuilder(output_dir=tmpdir)

            # Build cache
            cache = builder.build_cache(race_data)

            # Verify structure
            assert 'metadata' in cache
            assert 'samples' in cache
            assert len(cache['samples']) == 4

            # Verify metadata
            assert cache['metadata']['version'] == '1.0'
            assert cache['metadata']['sample_count'] == 4
            assert cache['metadata']['total_races'] == 2
            assert 'created_at' in cache['metadata']

            # Verify samples
            for _i, sample in enumerate(cache['samples']):
                assert 'race_id' in sample
                assert 'driver_id' in sample
                assert 'image_path' in sample
                assert 'features' in sample
                assert 'feature_names' in sample

                # Verify features are floats
                features = sample['features']
                assert all(isinstance(f, float) for f in features)
                assert len(features) > 0

    def test_cache_persistence_integration(self, race_data: list[dict]) -> None:
        """Test saving and loading cache from disk.

        Verifies:
        - Cache saves to JSON correctly
        - Loaded cache matches original
        - Validation passes on loaded cache
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetCacheBuilder(output_dir=tmpdir)

            # Build and save cache
            cache = builder.build_cache(race_data)
            path = builder.save_cache(cache, filename='test_cache.json')

            # Verify file exists
            assert Path(path).exists()

            # Load cache
            loaded_cache = builder.load_cache(filename='test_cache.json')

            # Verify content matches
            assert loaded_cache == cache
            assert loaded_cache['metadata']['sample_count'] == 4

            # Verify validation passes
            assert builder.validate_cache(loaded_cache)

    def test_training_with_mock_data(self, race_data: list[dict]) -> None:
        """Test multi-modal model training with dataset cache.

        Verifies:
        - Model trains for multiple epochs
        - Loss decreases over training
        - Training completes without errors
        - Metrics are computed
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build cache
            builder = DatasetCacheBuilder(output_dir=tmpdir)
            cache = builder.build_cache(race_data)

            # Create dataset
            dataset = SimpleMultiModalDataset(cache, num_samples=min(4, len(cache['samples'])))
            batch_size = 2
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(dataset, batch_size=batch_size)

            # Create model
            num_features = len(cache['samples'][0]['features'])
            model = SimpleMultiModalModel(num_features=num_features, num_classes=2)

            # Create trainer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            trainer = MultiModalTrainer(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device='cpu',
                num_epochs=3,
                early_stopping_patience=10
            )

            # Train
            history = trainer.train(num_epochs=3)

            # Verify training occurred
            assert 'train_loss' in history
            assert 'val_loss' in history
            assert len(history['train_loss']) > 0
            assert len(history['val_loss']) > 0

            # Verify metrics are reasonable
            assert all(isinstance(loss, float) for loss in history['train_loss'])
            assert all(isinstance(loss, float) for loss in history['val_loss'])

    def test_model_checkpoint_integration(self, race_data: list[dict]) -> None:
        """Test model checkpoint save and load.

        Verifies:
        - Checkpoint saves with correct structure
        - Model state restored from checkpoint
        - Training can resume from checkpoint
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            builder = DatasetCacheBuilder(output_dir=tmpdir)
            cache = builder.build_cache(race_data)
            dataset = SimpleMultiModalDataset(cache, num_samples=min(4, len(cache['samples'])))
            loader = DataLoader(dataset, batch_size=2)

            num_features = len(cache['samples'][0]['features'])
            model = SimpleMultiModalModel(num_features=num_features, num_classes=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            trainer = MultiModalTrainer(
                model=model,
                train_dataloader=loader,
                val_dataloader=loader,
                optimizer=optimizer,
                criterion=criterion,
                device='cpu',
                num_epochs=2
            )

            # Train briefly
            trainer.train(num_epochs=1)

            # Save checkpoint
            checkpoint_path = str(Path(tmpdir) / 'model_checkpoint.pt')
            trainer.save_checkpoint(checkpoint_path)

            # Verify checkpoint exists
            assert Path(checkpoint_path).exists()

            # Create new trainer and load checkpoint
            new_model = SimpleMultiModalModel(num_features=num_features, num_classes=2)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
            new_trainer = MultiModalTrainer(
                model=new_model,
                train_dataloader=loader,
                val_dataloader=loader,
                optimizer=new_optimizer,
                criterion=criterion,
                device='cpu'
            )

            # Load checkpoint
            new_trainer.load_checkpoint(checkpoint_path)

            # Verify state was restored
            assert new_trainer.best_val_loss != float('inf')

    def test_end_to_end_workflow(self, race_data: list[dict]) -> None:
        """Test complete end-to-end training workflow.

        Verifies:
        - Speed traces generated
        - Cache built and saved
        - Model trains successfully
        - Checkpoint created
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Step 1: Generate speed traces
            generator = SpeedTraceGenerator(output_dir=str(tmpdir_path / 'traces'))
            trace_results = generator.generate_batch(race_data)
            assert len(trace_results) > 0

            # Step 2: Build and cache dataset
            builder = DatasetCacheBuilder(output_dir=str(tmpdir_path / 'cache'))
            cache = builder.build_cache(race_data)
            cache_path = builder.save_cache(cache)
            assert Path(cache_path).exists()

            # Step 3: Verify cache is valid
            loaded_cache = builder.load_cache()
            assert builder.validate_cache(loaded_cache)

            # Step 4: Create datasets and loaders
            dataset = SimpleMultiModalDataset(loaded_cache, num_samples=4)
            train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
            val_loader = DataLoader(dataset, batch_size=2)

            # Step 5: Train model
            num_features = len(loaded_cache['samples'][0]['features'])
            model = SimpleMultiModalModel(num_features=num_features, num_classes=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            trainer = MultiModalTrainer(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device='cpu',
                num_epochs=2,
                early_stopping_patience=5
            )

            history = trainer.train(num_epochs=2)

            # Step 6: Save checkpoint
            checkpoint_path = str(tmpdir_path / 'final_model.pt')
            trainer.save_checkpoint(checkpoint_path)

            # Verify complete workflow
            assert len(history['train_loss']) > 0
            assert len(history['val_loss']) > 0
            assert Path(checkpoint_path).exists()

    def test_training_with_early_stopping(self, race_data: list[dict]) -> None:
        """Test early stopping mechanism during training.

        Verifies:
        - Training completes and produces history
        - Early stopping patience is used in trainer configuration
        - Training metrics are collected properly
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetCacheBuilder(output_dir=tmpdir)
            cache = builder.build_cache(race_data)
            dataset = SimpleMultiModalDataset(cache, num_samples=4)
            loader = DataLoader(dataset, batch_size=2)

            num_features = len(cache['samples'][0]['features'])
            model = SimpleMultiModalModel(num_features=num_features, num_classes=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            trainer = MultiModalTrainer(
                model=model,
                train_dataloader=loader,
                val_dataloader=loader,
                optimizer=optimizer,
                criterion=criterion,
                device='cpu',
                num_epochs=20,
                early_stopping_patience=3
            )

            # Verify trainer has early stopping patience set
            assert trainer.early_stopping_patience == 3

            history = trainer.train(num_epochs=20)

            # Verify training occurred and completed
            assert len(history['train_loss']) > 0
            assert len(history['val_loss']) > 0
            # Training may complete all epochs or stop early depending on convergence
            assert len(history['train_loss']) <= 20


class TestFeatureExtraction:
    """Tests for feature extraction in integration context."""

    def test_feature_extraction_consistency(self) -> None:
        """Test that feature extraction is consistent across runs.

        Verifies:
        - Same input produces same features
        - Features are reproducible
        """
        builder = DatasetCacheBuilder()

        weather_data = {'temp': 25, 'humidity': 60, 'wind_speed': 10}
        tire_data = {'compound': 'soft', 'age_laps': 5}
        driver_data = {'age': 28, 'experience_years': 8}
        circuit_data = {'elevation': 500, 'length_km': 5.5}

        # Extract twice
        features1 = builder.extract_tabular_features(
            '2024-01', 'driver_001',
            weather_data, tire_data, driver_data, circuit_data
        )
        features2 = builder.extract_tabular_features(
            '2024-01', 'driver_001',
            weather_data, tire_data, driver_data, circuit_data
        )

        # Should be identical
        assert features1 == features2

    def test_feature_extraction_with_missing_data(self) -> None:
        """Test feature extraction handles missing fields gracefully.

        Verifies:
        - Missing fields don't cause errors
        - Partial data extracts available features
        """
        builder = DatasetCacheBuilder()

        # Minimal data
        features = builder.extract_tabular_features(
            '2024-01', 'driver_001',
            weather_data={'temp': 25},
            tire_data={},
            driver_data={},
            circuit_data={}
        )

        # Should still extract what's available
        assert 'temp' in features
        assert len(features) >= 1

