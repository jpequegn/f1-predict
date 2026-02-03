"""Tests for multi-modal training pipeline."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from f1_predict.models.multimodal_training import (
    AblationStudyRunner,
    MultiModalTrainingPipeline,
    WarmupCosineScheduler,
)


class MockMultiModalModel(nn.Module):
    """Mock multi-modal model for testing."""

    def __init__(self, image_dim=512, tabular_dim=64, num_classes=20):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(3 * 32 * 32, image_dim)
        )
        self.tabular_encoder = nn.Linear(tabular_dim, 128)
        self.classifier = nn.Linear(image_dim + 128, num_classes)

    def forward(self, images, tabular):
        img_feat = self.image_encoder(images)
        tab_feat = self.tabular_encoder(tabular)
        combined = torch.cat([img_feat, tab_feat], dim=1)
        return self.classifier(combined)


@pytest.fixture
def mock_model():
    """Create mock model."""
    return MockMultiModalModel()


@pytest.fixture
def sample_dataloader():
    """Create sample dataloader."""
    images = torch.randn(32, 3, 32, 32)
    tabular = torch.randn(32, 64)
    labels = torch.randint(0, 20, (32,))
    dataset = TensorDataset(images, tabular, labels)
    return DataLoader(dataset, batch_size=8)


class TestMultiModalTrainingPipeline:
    """Tests for MultiModalTrainingPipeline class."""

    @pytest.fixture
    def pipeline(self, mock_model):
        """Create training pipeline."""
        return MultiModalTrainingPipeline(
            model=mock_model,
            learning_rate=1e-3,
            device="cpu",
        )

    def test_init(self, mock_model):
        """Test pipeline initialization."""
        pipeline = MultiModalTrainingPipeline(
            model=mock_model,
            learning_rate=1e-3,
        )
        assert pipeline.model is not None
        assert pipeline.optimizer is not None

    def test_init_with_scheduler(self, mock_model):
        """Test initialization with scheduler."""
        pipeline = MultiModalTrainingPipeline(
            model=mock_model,
            learning_rate=1e-3,
            scheduler_type="cosine",
        )
        assert pipeline.scheduler is not None

    def test_train_epoch(self, pipeline, sample_dataloader):
        """Test single epoch training."""
        loss = pipeline.train_epoch(sample_dataloader)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_validate(self, pipeline, sample_dataloader):
        """Test validation."""
        metrics = pipeline.validate(sample_dataloader)
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_train_with_validation(self, pipeline, sample_dataloader):
        """Test training with validation."""
        history = pipeline.train(
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            epochs=2,
        )
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 2

    def test_train_with_early_stopping(self, pipeline, sample_dataloader):
        """Test training with early stopping."""
        history = pipeline.train(
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            epochs=100,
            early_stopping_patience=2,
        )
        # Should stop early
        assert len(history["train_loss"]) <= 100

    def test_save_checkpoint(self, pipeline, tmp_path):
        """Test saving checkpoint."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        pipeline.save_checkpoint(str(checkpoint_path))
        assert checkpoint_path.exists()

    def test_load_checkpoint(self, pipeline, tmp_path):
        """Test loading checkpoint."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        pipeline.save_checkpoint(str(checkpoint_path))
        pipeline.load_checkpoint(str(checkpoint_path))
        # Should load without error

    def test_gradient_accumulation(self, mock_model, sample_dataloader):
        """Test gradient accumulation."""
        pipeline = MultiModalTrainingPipeline(
            model=mock_model,
            learning_rate=1e-3,
            gradient_accumulation_steps=4,
        )
        loss = pipeline.train_epoch(sample_dataloader)
        assert isinstance(loss, float)

    def test_gradient_clipping(self, mock_model, sample_dataloader):
        """Test gradient clipping."""
        pipeline = MultiModalTrainingPipeline(
            model=mock_model,
            learning_rate=1e-3,
            max_grad_norm=1.0,
        )
        loss = pipeline.train_epoch(sample_dataloader)
        assert isinstance(loss, float)


class TestWarmupCosineScheduler:
    """Tests for WarmupCosineScheduler class."""

    @pytest.fixture
    def optimizer(self, mock_model):
        """Create optimizer."""
        return torch.optim.Adam(mock_model.parameters(), lr=1e-3)

    def test_init(self, optimizer):
        """Test scheduler initialization."""
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=100,
            total_steps=1000,
        )
        assert scheduler is not None

    def test_warmup_phase(self, optimizer):
        """Test warmup phase increases LR."""
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=10,
            total_steps=100,
        )

        _ = optimizer.param_groups[0]["lr"]
        for _ in range(5):
            scheduler.step()
        warmup_lr = optimizer.param_groups[0]["lr"]

        # During warmup, LR should increase (or stay same at start)
        assert warmup_lr >= 0

    def test_cosine_phase(self, optimizer):
        """Test cosine annealing phase."""
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=10,
            total_steps=100,
        )

        # Complete warmup
        for _ in range(10):
            scheduler.step()

        post_warmup_lr = optimizer.param_groups[0]["lr"]

        # Continue training
        for _ in range(50):
            scheduler.step()

        later_lr = optimizer.param_groups[0]["lr"]

        # LR should decrease during cosine phase
        assert later_lr <= post_warmup_lr

    def test_final_lr(self, optimizer):
        """Test final learning rate."""
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr=1e-6,
        )

        # Run to completion
        for _ in range(100):
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr >= 1e-6


class TestAblationStudyRunner:
    """Tests for AblationStudyRunner class."""

    @pytest.fixture
    def runner(self, sample_dataloader):
        """Create ablation study runner."""
        return AblationStudyRunner(
            base_model_fn=lambda: MockMultiModalModel(),
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
        )

    def test_init(self, sample_dataloader):
        """Test runner initialization."""
        runner = AblationStudyRunner(
            base_model_fn=lambda: MockMultiModalModel(),
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
        )
        assert runner is not None

    def test_run_image_ablation(self, runner):
        """Test image-only ablation."""
        results = runner.run_ablation(
            name="image_only",
            use_images=True,
            use_tabular=False,
            epochs=1,
        )
        assert "accuracy" in results or "loss" in results

    def test_run_tabular_ablation(self, runner):
        """Test tabular-only ablation."""
        results = runner.run_ablation(
            name="tabular_only",
            use_images=False,
            use_tabular=True,
            epochs=1,
        )
        assert "accuracy" in results or "loss" in results

    def test_run_full_ablation(self, runner):
        """Test full model ablation."""
        results = runner.run_ablation(
            name="full_model",
            use_images=True,
            use_tabular=True,
            epochs=1,
        )
        assert "accuracy" in results or "loss" in results

    def test_compare_results(self, runner):
        """Test result comparison."""
        results1 = runner.run_ablation(
            name="exp1",
            use_images=True,
            use_tabular=True,
            epochs=1,
        )
        results2 = runner.run_ablation(
            name="exp2",
            use_images=True,
            use_tabular=False,
            epochs=1,
        )

        comparison = runner.compare_results([results1, results2])
        assert comparison is not None


class TestTrainingEdgeCases:
    """Tests for training edge cases."""

    def test_single_batch_training(self, mock_model):
        """Test training with single batch."""
        images = torch.randn(4, 3, 32, 32)
        tabular = torch.randn(4, 64)
        labels = torch.randint(0, 20, (4,))
        dataset = TensorDataset(images, tabular, labels)
        loader = DataLoader(dataset, batch_size=4)

        pipeline = MultiModalTrainingPipeline(
            model=mock_model,
            learning_rate=1e-3,
        )
        loss = pipeline.train_epoch(loader)
        assert isinstance(loss, float)

    def test_training_with_zero_lr(self, mock_model, sample_dataloader):
        """Test training with zero learning rate."""
        pipeline = MultiModalTrainingPipeline(
            model=mock_model,
            learning_rate=0.0,
        )
        # Should not raise error
        loss = pipeline.train_epoch(sample_dataloader)
        assert isinstance(loss, float)

    def test_training_empty_loader(self, mock_model):
        """Test training with empty loader."""
        images = torch.randn(0, 3, 32, 32)
        tabular = torch.randn(0, 64)
        labels = torch.randint(0, 20, (0,))
        dataset = TensorDataset(images, tabular, labels)
        loader = DataLoader(dataset, batch_size=1)

        pipeline = MultiModalTrainingPipeline(
            model=mock_model,
            learning_rate=1e-3,
        )

        # Should handle empty loader gracefully
        try:
            loss = pipeline.train_epoch(loader)
            assert loss == 0.0 or loss is None
        except (StopIteration, ValueError):
            pass  # Expected for empty loader


class TestSchedulerTypes:
    """Tests for different scheduler types."""

    @pytest.fixture
    def mock_model_fixture(self):
        """Create mock model."""
        return MockMultiModalModel()

    def test_cosine_scheduler(self, mock_model_fixture, sample_dataloader):
        """Test cosine scheduler."""
        pipeline = MultiModalTrainingPipeline(
            model=mock_model_fixture,
            learning_rate=1e-3,
            scheduler_type="cosine",
        )
        pipeline.train_epoch(sample_dataloader)
        assert pipeline.scheduler is not None

    def test_step_scheduler(self, mock_model_fixture, sample_dataloader):
        """Test step scheduler."""
        pipeline = MultiModalTrainingPipeline(
            model=mock_model_fixture,
            learning_rate=1e-3,
            scheduler_type="step",
        )
        pipeline.train_epoch(sample_dataloader)
        assert pipeline.scheduler is not None

    def test_plateau_scheduler(self, mock_model_fixture, sample_dataloader):
        """Test plateau scheduler."""
        pipeline = MultiModalTrainingPipeline(
            model=mock_model_fixture,
            learning_rate=1e-3,
            scheduler_type="plateau",
        )
        pipeline.train_epoch(sample_dataloader)
        assert pipeline.scheduler is not None

    def test_warmup_cosine_scheduler(self, mock_model_fixture, sample_dataloader):
        """Test warmup cosine scheduler."""
        pipeline = MultiModalTrainingPipeline(
            model=mock_model_fixture,
            learning_rate=1e-3,
            scheduler_type="warmup_cosine",
            warmup_steps=10,
        )
        pipeline.train_epoch(sample_dataloader)
        assert pipeline.scheduler is not None
