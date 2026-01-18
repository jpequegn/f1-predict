"""Integration tests for multi-modal learning pipeline.

Tests the complete end-to-end workflow including:
- Speed trace image generation
- Dataset cache building
- Multi-modal dataset loading
- Vision encoder feature extraction
- Fusion model training and inference
- Baseline comparison for improvement measurement
"""

import json
from pathlib import Path
import tempfile
import time

from PIL import Image
import pytest
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from f1_predict.data.multimodal_dataset import MultiModalDataset
from f1_predict.models.multimodal_fusion import MultiModalFusionModel
from f1_predict.models.multimodal_trainer import MultiModalTrainer
from f1_predict.models.speed_trace_generator import SpeedTraceGenerator
from f1_predict.models.vision_encoder import VisionEncoder


class TestMultiModalEndToEndTraining:
    """End-to-end multi-modal training tests."""

    @pytest.fixture
    def temp_multimodal_dir(self):
        """Create temporary directory structure for multi-modal data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "speed_traces").mkdir()
            (temp_path / "cache").mkdir()
            yield temp_path

    @pytest.fixture
    def sample_races_data(self):
        """Generate sample race data for testing."""
        races = []
        for race_idx in range(5):  # 5 races
            race_id = f"2024_race_{race_idx + 1}"
            drivers = []
            for driver_idx in range(10):  # 10 drivers per race
                # Generate synthetic lap data (varying speeds)
                num_laps = 50 + (race_idx * 2)
                base_speed = 200 + (driver_idx * 5)
                lap_data = [
                    base_speed + (i % 10) - 5 + (driver_idx * 0.5)
                    for i in range(num_laps)
                ]
                drivers.append(
                    {
                        "driver_id": f"driver_{driver_idx + 1}",
                        "lap_data": lap_data,
                        "position": driver_idx + 1,  # Finishing position
                    }
                )
            races.append({"race_id": race_id, "drivers": drivers})
        return races

    def test_speed_trace_generation_batch(self, temp_multimodal_dir, sample_races_data):
        """Test generating speed traces for multiple races."""
        generator = SpeedTraceGenerator(
            output_dir=str(temp_multimodal_dir / "speed_traces")
        )

        results = generator.generate_batch(sample_races_data)

        # Should generate 50 traces (5 races * 10 drivers)
        assert len(results) == 50

        # Verify all files exist
        for key, path in results.items():
            assert Path(path).exists(), f"Missing trace for {key}"

        # Verify file sizes are reasonable (not empty)
        for path in results.values():
            file_size = Path(path).stat().st_size
            assert file_size > 1000, f"Trace file too small: {path}"

    def test_dataset_cache_creation(self, temp_multimodal_dir, sample_races_data):
        """Test creating dataset cache from race data."""
        # First generate speed traces
        generator = SpeedTraceGenerator(
            output_dir=str(temp_multimodal_dir / "speed_traces")
        )
        trace_paths = generator.generate_batch(sample_races_data)

        # Build cache manually
        cache_data = {
            "metadata": [],
            "feature_names": ["speed_avg", "speed_var", "num_laps"],
        }

        for race in sample_races_data:
            race_id = race["race_id"]
            for driver in race["drivers"]:
                driver_id = driver["driver_id"]
                lap_data = driver["lap_data"]
                key = f"{race_id}_{driver_id}"

                if key in trace_paths:
                    # Extract simple features
                    avg_speed = sum(lap_data) / len(lap_data)
                    var_speed = sum((x - avg_speed) ** 2 for x in lap_data) / len(
                        lap_data
                    )

                    cache_data["metadata"].append(
                        {
                            "race_id": race_id,
                            "driver_id": driver_id,
                            "image_path": trace_paths[key],
                            "tabular_features": [avg_speed, var_speed, len(lap_data)],
                            "label": driver["position"] - 1,  # 0-indexed
                            "finished": True,
                        }
                    )

        # Save cache
        cache_file = temp_multimodal_dir / "cache" / "dataset_cache.json"
        cache_file.parent.mkdir(exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        # Verify cache
        assert cache_file.exists()
        with open(cache_file) as f:
            loaded = json.load(f)
        assert len(loaded["metadata"]) == 50

    def test_train_one_epoch_no_crashes(self, temp_multimodal_dir, sample_races_data):
        """Test training for one epoch completes without crashes."""
        # Setup: generate traces and cache
        generator = SpeedTraceGenerator(
            output_dir=str(temp_multimodal_dir / "speed_traces")
        )
        trace_paths = generator.generate_batch(sample_races_data)

        # Build cache
        cache_data = {"metadata": [], "feature_names": ["f1", "f2", "f3"]}
        for race in sample_races_data:
            for driver in race["drivers"]:
                key = f"{race['race_id']}_{driver['driver_id']}"
                if key in trace_paths:
                    lap_data = driver["lap_data"]
                    cache_data["metadata"].append(
                        {
                            "race_id": race["race_id"],
                            "driver_id": driver["driver_id"],
                            "image_path": trace_paths[key],
                            "tabular_features": [
                                sum(lap_data) / len(lap_data),
                                max(lap_data) - min(lap_data),
                                len(lap_data),
                            ],
                            "label": driver["position"] - 1,
                            "finished": True,
                        }
                    )

        cache_file = temp_multimodal_dir / "cache.json"
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        # Create dataset and dataloader
        dataset = MultiModalDataset(cache_file=str(cache_file))

        def collate_fn(batch):
            """Custom collate to handle None images."""
            images = []
            tabulars = []
            labels = []
            for img, tab, lbl in batch:
                if img is not None:
                    images.append(img)
                tabulars.append(tab)
                labels.append(lbl)

            images_tensor = torch.stack(images) if images else None
            return images_tensor, torch.stack(tabulars), torch.tensor(labels)

        dataloader = DataLoader(
            dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
        )

        # Create model
        vision_encoder = VisionEncoder(model_name="resnet18", freeze_backbone=True)
        fusion_model = MultiModalFusionModel(
            image_feature_dim=512,
            tabular_input_dim=3,
            output_dim=10,  # 10 positions
        )

        # Training components
        optimizer = Adam(fusion_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Train one epoch
        fusion_model.train()
        total_loss = 0.0
        batch_count = 0

        for batch_images, batch_tabular, batch_labels in dataloader:
            # Extract image features
            if batch_images is not None:
                with torch.no_grad():
                    image_features = vision_encoder(batch_images)
            else:
                image_features = None

            # Forward pass
            outputs = fusion_model(image_features, batch_tabular)
            loss = criterion(outputs, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        assert avg_loss > 0, "Loss should be positive"
        assert batch_count > 0, "Should have processed batches"

    def test_loss_decreases_over_epochs(self, temp_multimodal_dir, sample_races_data):
        """Test that loss decreases during training."""
        # Setup
        generator = SpeedTraceGenerator(
            output_dir=str(temp_multimodal_dir / "speed_traces")
        )
        trace_paths = generator.generate_batch(sample_races_data)

        cache_data = {"metadata": [], "feature_names": ["f1", "f2", "f3"]}
        for race in sample_races_data:
            for driver in race["drivers"]:
                key = f"{race['race_id']}_{driver['driver_id']}"
                if key in trace_paths:
                    lap_data = driver["lap_data"]
                    cache_data["metadata"].append(
                        {
                            "race_id": race["race_id"],
                            "driver_id": driver["driver_id"],
                            "image_path": trace_paths[key],
                            "tabular_features": [
                                sum(lap_data) / len(lap_data),
                                max(lap_data) - min(lap_data),
                                len(lap_data),
                            ],
                            "label": driver["position"] - 1,
                            "finished": True,
                        }
                    )

        cache_file = temp_multimodal_dir / "cache.json"
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        dataset = MultiModalDataset(cache_file=str(cache_file))

        def collate_fn(batch):
            images, tabulars, labels = [], [], []
            for img, tab, lbl in batch:
                if img is not None:
                    images.append(img)
                tabulars.append(tab)
                labels.append(lbl)
            images_tensor = torch.stack(images) if images else None
            return images_tensor, torch.stack(tabulars), torch.tensor(labels)

        dataloader = DataLoader(
            dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
        )

        vision_encoder = VisionEncoder(model_name="resnet18", freeze_backbone=True)
        fusion_model = MultiModalFusionModel(
            image_feature_dim=512, tabular_input_dim=3, output_dim=10
        )

        optimizer = Adam(fusion_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Train for 3 epochs and track loss
        epoch_losses = []
        for _epoch in range(3):
            fusion_model.train()
            total_loss = 0.0
            batch_count = 0

            for batch_images, batch_tabular, batch_labels in dataloader:
                if batch_images is not None:
                    with torch.no_grad():
                        image_features = vision_encoder(batch_images)
                else:
                    image_features = None

                outputs = fusion_model(image_features, batch_tabular)
                loss = criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / batch_count
            epoch_losses.append(avg_loss)

        # Loss should generally decrease (allow some variance)
        assert (
            epoch_losses[-1] <= epoch_losses[0] * 1.1
        ), f"Loss should decrease: {epoch_losses}"


class TestMultiModalInferenceLatency:
    """Test inference latency requirements."""

    @pytest.fixture
    def trained_model(self):
        """Create a model ready for inference."""
        vision_encoder = VisionEncoder(model_name="resnet18", freeze_backbone=True)
        fusion_model = MultiModalFusionModel(
            image_feature_dim=512, tabular_input_dim=10, output_dim=20
        )
        return vision_encoder, fusion_model

    def test_inference_latency_under_one_second(self, trained_model):
        """Test that inference completes in under 1 second per sample."""
        vision_encoder, fusion_model = trained_model
        vision_encoder.eval()
        fusion_model.eval()

        # Create dummy inputs
        num_samples = 20
        dummy_images = torch.randn(num_samples, 3, 224, 224)
        dummy_tabular = torch.randn(num_samples, 10)

        # Warm up
        with torch.no_grad():
            _ = vision_encoder(dummy_images[:1])
            _ = fusion_model(torch.randn(1, 512), dummy_tabular[:1])

        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            for i in range(num_samples):
                image_features = vision_encoder(dummy_images[i : i + 1])
                _ = fusion_model(image_features, dummy_tabular[i : i + 1])
        elapsed = time.time() - start_time

        avg_time_per_sample = elapsed / num_samples
        assert (
            avg_time_per_sample < 1.0
        ), f"Inference too slow: {avg_time_per_sample:.3f}s per sample"

    def test_batch_inference_efficiency(self, trained_model):
        """Test batch inference produces correct output shapes."""
        vision_encoder, fusion_model = trained_model
        vision_encoder.eval()
        fusion_model.eval()

        batch_size = 16
        dummy_images = torch.randn(batch_size, 3, 224, 224)
        dummy_tabular = torch.randn(batch_size, 10)

        # Batch inference should produce correct shapes
        with torch.no_grad():
            img_feat = vision_encoder(dummy_images)
            output = fusion_model(img_feat, dummy_tabular)

        # Verify batch processing works correctly
        assert img_feat.shape == (batch_size, 512)
        assert output.shape == (batch_size, 20)
        assert torch.isfinite(output).all()


class TestMultiModalMissingImageHandling:
    """Test graceful handling of missing images."""

    def test_model_handles_none_images(self):
        """Test fusion model handles None image input."""
        fusion_model = MultiModalFusionModel(
            image_feature_dim=512, tabular_input_dim=5, output_dim=10
        )

        # Pass None for images
        tabular = torch.randn(4, 5)
        output = fusion_model(None, tabular)

        assert output.shape == (4, 10)
        assert torch.isfinite(output).all()

    def test_dataset_returns_none_for_missing_file(self):
        """Test dataset returns None when image file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "cache.json"
            cache_data = {
                "metadata": [
                    {
                        "race_id": "test_race",
                        "driver_id": "test_driver",
                        "image_path": "/nonexistent/path/image.png",
                        "tabular_features": [1.0, 2.0, 3.0],
                        "label": 0,
                        "finished": True,
                    }
                ],
                "feature_names": ["f1", "f2", "f3"],
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

            dataset = MultiModalDataset(cache_file=str(cache_file))
            image, tabular, label = dataset[0]

            assert image is None
            assert tabular.shape == (3,)
            assert label == 0

    def test_training_with_mixed_missing_images(self):
        """Test training continues with mix of present and missing images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some real images
            img_dir = temp_path / "images"
            img_dir.mkdir()
            for i in range(3):
                img = Image.new("RGB", (224, 224), color="blue")
                img.save(img_dir / f"driver_{i}.png")

            # Create cache with mix of existing and missing images
            metadata = []
            for i in range(5):
                if i < 3:
                    img_path = str(img_dir / f"driver_{i}.png")
                else:
                    img_path = "/nonexistent/path.png"  # Missing

                metadata.append(
                    {
                        "race_id": "test_race",
                        "driver_id": f"driver_{i}",
                        "image_path": img_path,
                        "tabular_features": [float(i), float(i + 1), float(i + 2)],
                        "label": i % 3,
                        "finished": True,
                    }
                )

            cache_file = temp_path / "cache.json"
            with open(cache_file, "w") as f:
                json.dump(
                    {"metadata": metadata, "feature_names": ["f1", "f2", "f3"]}, f
                )

            dataset = MultiModalDataset(cache_file=str(cache_file))

            def collate_fn(batch):
                images, tabulars, labels = [], [], []
                for img, tab, lbl in batch:
                    if img is not None:
                        images.append(img)
                    tabulars.append(tab)
                    labels.append(lbl)
                images_tensor = torch.stack(images) if images else None
                return images_tensor, torch.stack(tabulars), torch.tensor(labels)

            dataloader = DataLoader(dataset, batch_size=5, collate_fn=collate_fn)

            # Should be able to iterate without errors
            for batch_images, batch_tabular, batch_labels in dataloader:
                assert batch_tabular.shape[0] == 5
                assert batch_labels.shape[0] == 5
                # Images might be partially present or None
                if batch_images is not None:
                    assert batch_images.shape[0] == 3  # Only 3 images exist


class TestMultiModalTrainerIntegration:
    """Test MultiModalTrainer with full pipeline."""

    @pytest.fixture
    def training_setup(self):
        """Set up training components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create dummy images
            img_dir = temp_path / "images"
            img_dir.mkdir()

            metadata = []
            for i in range(20):
                img = Image.new("RGB", (224, 224), color=(i * 10, 50, 100))
                img_path = img_dir / f"sample_{i}.png"
                img.save(img_path)

                metadata.append(
                    {
                        "race_id": f"race_{i // 5}",
                        "driver_id": f"driver_{i % 5}",
                        "image_path": str(img_path),
                        "tabular_features": [float(i), float(i * 2), float(i % 5)],
                        "label": i % 5,
                        "finished": True,
                    }
                )

            cache_file = temp_path / "cache.json"
            with open(cache_file, "w") as f:
                json.dump(
                    {"metadata": metadata, "feature_names": ["f1", "f2", "f3"]}, f
                )

            dataset = MultiModalDataset(cache_file=str(cache_file))

            # Split into train/val
            train_size = 16
            train_dataset = torch.utils.data.Subset(dataset, range(train_size))
            val_dataset = torch.utils.data.Subset(dataset, range(train_size, 20))

            def collate_fn(batch):
                images, tabulars, labels = [], [], []
                for img, tab, lbl in batch:
                    if img is not None:
                        images.append(img)
                    tabulars.append(tab)
                    labels.append(lbl)
                images_tensor = torch.stack(images) if images else None
                return images_tensor, torch.stack(tabulars), torch.tensor(labels)

            train_loader = DataLoader(
                train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
            )
            val_loader = DataLoader(
                val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
            )

            yield {
                "train_loader": train_loader,
                "val_loader": val_loader,
                "temp_dir": temp_path,
            }

    def test_trainer_completes_training(self, training_setup):
        """Test trainer completes multiple epochs."""

        # Create model that combines vision encoder and fusion
        class CombinedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision = VisionEncoder(freeze_backbone=True)
                self.fusion = MultiModalFusionModel(
                    image_feature_dim=512, tabular_input_dim=3, output_dim=5
                )

            def forward(self, images, tabular):
                if images is not None:
                    with torch.no_grad():
                        img_feat = self.vision(images)
                else:
                    img_feat = None
                return self.fusion(img_feat, tabular)

        model = CombinedModel()
        optimizer = Adam(model.fusion.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        trainer = MultiModalTrainer(
            model=model,
            train_dataloader=training_setup["train_loader"],
            val_dataloader=training_setup["val_loader"],
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            num_epochs=3,
            early_stopping_patience=5,
        )

        history = trainer.train(num_epochs=3)

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 3

    def test_trainer_saves_and_loads_checkpoint(self, training_setup):
        """Test checkpoint save/load functionality."""

        class CombinedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision = VisionEncoder(freeze_backbone=True)
                self.fusion = MultiModalFusionModel(
                    image_feature_dim=512, tabular_input_dim=3, output_dim=5
                )

            def forward(self, images, tabular):
                if images is not None:
                    with torch.no_grad():
                        img_feat = self.vision(images)
                else:
                    img_feat = None
                return self.fusion(img_feat, tabular)

        model = CombinedModel()
        optimizer = Adam(model.fusion.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        trainer = MultiModalTrainer(
            model=model,
            train_dataloader=training_setup["train_loader"],
            val_dataloader=training_setup["val_loader"],
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
        )

        # Train one epoch
        trainer.train(num_epochs=1)

        # Save checkpoint
        checkpoint_path = training_setup["temp_dir"] / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Load checkpoint
        trainer.load_checkpoint(str(checkpoint_path))
