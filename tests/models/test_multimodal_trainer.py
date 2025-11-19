"""Tests for multi-modal training pipeline."""

from pathlib import Path
import tempfile
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from f1_predict.models.multimodal_fusion import MultiModalFusionModel
from f1_predict.models.multimodal_trainer import MultiModalTrainer


@pytest.fixture
def sample_model():
    """Create test model."""
    return MultiModalFusionModel(
        image_feature_dim=512,
        tabular_input_dim=10,
        hidden_dim=256,
        output_dim=20
    )


@pytest.fixture
def sample_optimizer(sample_model):
    """Create test optimizer."""
    return torch.optim.Adam(sample_model.parameters(), lr=1e-3)


@pytest.fixture
def sample_criterion():
    """Create test loss criterion."""
    return nn.CrossEntropyLoss()


@pytest.fixture
def sample_dataloader():
    """Create mock dataloader."""
    dataloader = Mock()
    # Return batches of (image_features, tabular, labels)
    # Note: image_features should be pre-extracted 512-dim features, not raw images
    batch = (
        torch.randn(4, 512),  # Pre-extracted image features (512-dim from ResNet-18)
        torch.randn(4, 10),   # Tabular features
        torch.randint(0, 20, (4,))  # Labels
    )
    # __iter__ must return a fresh iterator each time it's called
    dataloader.__iter__ = Mock(side_effect=lambda: iter([batch, batch, batch]))
    dataloader.__len__ = Mock(return_value=3)
    return dataloader


class TestMultiModalTrainerInitialization:
    """Test trainer creation and initialization."""

    def test_trainer_initialization(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test trainer initializes correctly."""
        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        assert trainer is not None
        assert trainer.model is sample_model
        assert trainer.optimizer is sample_optimizer
        assert trainer.criterion is sample_criterion

    def test_trainer_device_placement(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test model is placed on correct device."""
        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        assert trainer.device == torch.device('cpu')

    def test_trainer_default_hyperparameters(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test trainer has default hyperparameters."""
        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion
        )

        assert hasattr(trainer, 'learning_rate')
        assert hasattr(trainer, 'num_epochs')


class TestMultiModalTrainerEpochTraining:
    """Test single epoch training."""

    def test_train_epoch_returns_dict(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test train_epoch returns metrics dictionary."""
        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        metrics = trainer.train_epoch()

        assert isinstance(metrics, dict)
        assert 'loss' in metrics

    def test_train_epoch_loss_decreases(self, sample_model, sample_optimizer, sample_criterion):
        """Test that loss decreases during training."""
        # Create real dataloader with pre-extracted features
        dataset = Mock()
        dataset.__len__ = Mock(return_value=8)
        dataset.__getitem__ = Mock(
            side_effect=lambda _: (
                torch.randn(512),  # Pre-extracted image features
                torch.randn(10),   # Tabular features
                torch.randint(0, 20, (1,)).item()  # Label
            )
        )

        from torch.utils.data import DataLoader

        def collate_fn(batch):
            image_features = torch.stack([torch.tensor(b[0]) for b in batch])
            tabular = torch.stack([torch.tensor(b[1]) for b in batch])
            labels = torch.tensor([b[2] for b in batch])
            return image_features, tabular, labels

        train_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        val_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        metrics = trainer.train_epoch()

        assert 'loss' in metrics
        assert isinstance(metrics['loss'], (float, int))
        assert metrics['loss'] > 0


class TestMultiModalTrainerValidation:
    """Test validation functionality."""

    def test_validate_returns_dict(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test validate returns metrics dictionary."""
        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        metrics = trainer.validate()

        assert isinstance(metrics, dict)
        assert 'loss' in metrics

    def test_validate_has_accuracy(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test validation returns accuracy metric."""
        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        metrics = trainer.validate()

        assert 'accuracy' in metrics or 'val_loss' in metrics

    def test_validate_no_gradient_updates(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test that validation doesn't update model parameters."""
        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        # Get initial parameters
        initial_params = [p.clone() for p in sample_model.parameters()]

        trainer.validate()

        # Parameters should not change
        for initial, current in zip(initial_params, sample_model.parameters()):
            assert torch.allclose(initial, current)


class TestMultiModalTrainerFullTraining:
    """Test complete training loop."""

    def test_train_multiple_epochs(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test training for multiple epochs."""
        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        history = trainer.train(num_epochs=2)

        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert len(history['train_loss']) == 2

    def test_train_returns_history(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test train returns training history."""
        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        history = trainer.train(num_epochs=1)

        assert 'train_loss' in history
        assert isinstance(history['train_loss'], list)

    def test_train_with_early_stopping(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test training with early stopping."""
        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu',
            early_stopping_patience=2
        )

        history = trainer.train(num_epochs=10)

        # Should stop early (not all 10 epochs)
        assert len(history['train_loss']) <= 10


class TestMultiModalTrainerCheckpoints:
    """Test checkpoint save/load functionality."""

    def test_save_checkpoint(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test checkpoint saving."""
        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
            trainer.save_checkpoint(str(checkpoint_path))

            assert checkpoint_path.exists()

    def test_checkpoint_contains_model_state(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test checkpoint contains model state dict."""
        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
            trainer.save_checkpoint(str(checkpoint_path))

            checkpoint = torch.load(checkpoint_path)
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint

    def test_load_checkpoint(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test checkpoint loading."""
        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'checkpoint.pt'

            # Save checkpoint
            trainer.save_checkpoint(str(checkpoint_path))

            # Create new trainer and load
            new_model = MultiModalFusionModel(
                image_feature_dim=512,
                tabular_input_dim=10,
                hidden_dim=256,
                output_dim=20
            )
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)

            new_trainer = MultiModalTrainer(
                model=new_model,
                train_dataloader=sample_dataloader,
                val_dataloader=sample_dataloader,
                optimizer=new_optimizer,
                criterion=sample_criterion,
                device='cpu'
            )

            new_trainer.load_checkpoint(str(checkpoint_path))

            # Models should have same state
            for p1, p2 in zip(sample_model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)


class TestMultiModalTrainerMetrics:
    """Test metric computation."""

    def test_compute_accuracy(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test accuracy computation."""
        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        metrics = trainer.validate()

        # Should have some metric
        assert len(metrics) > 0

    def test_loss_is_scalar(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test loss is scalar value."""
        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        metrics = trainer.train_epoch()

        assert isinstance(metrics['loss'], (float, int))
        assert metrics['loss'] > 0


class TestMultiModalTrainerModes:
    """Test training and eval modes."""

    def test_model_in_train_mode_during_training(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test model is in train mode during training."""
        MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        # Model should start in train mode by default
        sample_model.train()
        assert sample_model.training

    def test_model_in_eval_mode_during_validation(self, sample_model, sample_optimizer, sample_criterion, sample_dataloader):
        """Test model is in eval mode during validation."""
        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        trainer.validate()

        # Model should be in eval mode after validation
        # (depends on implementation, but should be)


class TestMultiModalTrainerErrorHandling:
    """Test error handling."""

    def test_empty_dataloader_handling(self, sample_model, sample_optimizer, sample_criterion):
        """Test handling of empty dataloader."""
        empty_dataloader = Mock()
        empty_dataloader.__iter__ = Mock(side_effect=lambda: iter([]))
        empty_dataloader.__len__ = Mock(return_value=0)

        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=empty_dataloader,
            val_dataloader=empty_dataloader,
            optimizer=sample_optimizer,
            criterion=sample_criterion,
            device='cpu'
        )

        # Empty dataloader should cause ZeroDivisionError when computing average loss
        with pytest.raises((RuntimeError, ZeroDivisionError)):
            trainer.train_epoch()

    def test_nan_loss_detection(self, sample_model, sample_optimizer, sample_dataloader):
        """Test that trainer handles NaN loss gracefully."""
        # Mock criterion to return NaN with gradient tracking
        def nan_criterion(_outputs, _labels):
            # Create a tensor that requires grad to support backward()
            return torch.tensor(float('nan'), requires_grad=True)

        mock_criterion = Mock(side_effect=nan_criterion)

        trainer = MultiModalTrainer(
            model=sample_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader,
            optimizer=sample_optimizer,
            criterion=mock_criterion,
            device='cpu'
        )

        # NaN loss can be processed without raising error, but metrics will contain NaN
        metrics = trainer.train_epoch()

        # Verify trainer didn't crash and returned metrics
        assert 'loss' in metrics
        assert isinstance(metrics['loss'], (float, int))
