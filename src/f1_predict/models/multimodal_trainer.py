"""Training orchestration for multi-modal models."""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class MultiModalTrainer:
    """Orchestrates training, validation, and checkpointing for multi-modal models.

    Handles the training loop, metrics computation, early stopping, and checkpoint
    management for multi-modal fusion models combining vision and tabular features.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        num_epochs: int = 20,
        early_stopping_patience: int = 5
    ) -> None:
        """Initialize trainer.

        Args:
            model: Multi-modal fusion model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: PyTorch optimizer
            criterion: Loss function (e.g., CrossEntropyLoss)
            device: Device to use ('cpu' or 'cuda')
            learning_rate: Learning rate (informational)
            num_epochs: Default number of epochs
            early_stopping_patience: Epochs to wait before early stopping
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience

        # Move model to device
        self.model.to(self.device)

        # Training state
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary with epoch metrics (loss, accuracy, etc.)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in self.train_dataloader:
            images, tabular, labels = batch

            # Move to device
            if images is not None:
                images = images.to(self.device)
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images, tabular)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(self.train_dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def validate(self) -> dict[str, float]:
        """Validate model on validation set.

        Returns:
            Dictionary with validation metrics (loss, accuracy, etc.)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                images, tabular, labels = batch

                # Move to device
                if images is not None:
                    images = images.to(self.device)
                tabular = tabular.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images, tabular)
                loss = self.criterion(outputs, labels)

                # Metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def train(self, num_epochs: Optional[int] = None) -> dict[str, list[float]]:
        """Train for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train (uses default if None)

        Returns:
            Training history with lists of metrics per epoch
        """
        if num_epochs is None:
            num_epochs = self.num_epochs

        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        for epoch in range(num_epochs):
            # Train epoch
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])

            # Validate
            val_metrics = self.validate()
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])

            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            # Logging
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

        return history

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

    def get_model(self) -> nn.Module:
        """Get the underlying model.

        Returns:
            The PyTorch model
        """
        return self.model
