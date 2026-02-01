"""Enhanced multi-modal training pipeline for F1 prediction.

Provides comprehensive training orchestration with learning rate scheduling,
mixed precision training, gradient accumulation, metrics tracking, and
ablation study support.
"""

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import time
from typing import Any, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

# Use Any for optimizer to avoid torch version compatibility issues
OptimizerType = Any
SchedulerType = Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for multi-modal training."""

    # Basic training
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32

    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # cosine, step, plateau, warmup_cosine
    lr_warmup_epochs: int = 5
    lr_step_size: int = 10
    lr_gamma: float = 0.1
    lr_patience: int = 5  # For plateau scheduler
    lr_min: float = 1e-6

    # Gradient handling
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # float16 or bfloat16

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    save_every_n_epochs: int = 5

    # Logging
    log_every_n_steps: int = 100
    eval_every_n_epochs: int = 1

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Ablation study
    ablation_mode: Optional[str] = None  # None, 'image_only', 'tabular_only'


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    epoch: int = 0
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_steps: int = 0


@dataclass
class TrainingHistory:
    """Full training history."""

    train_losses: list[float] = field(default_factory=list)
    train_accuracies: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    best_metrics: TrainingMetrics = field(default_factory=TrainingMetrics)


class MultiModalTrainingPipeline:
    """Enhanced training pipeline for multi-modal models.

    Features:
    - Mixed precision training with automatic mixed precision (AMP)
    - Gradient accumulation for effective larger batch sizes
    - Learning rate scheduling (cosine, step, plateau, warmup)
    - Comprehensive metrics tracking and logging
    - Checkpointing and model saving
    - Ablation study support (image-only, tabular-only modes)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
        optimizer: Optional[OptimizerType] = None,
        criterion: Optional[nn.Module] = None,
    ):
        """Initialize training pipeline.

        Args:
            model: Multi-modal model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            optimizer: Optional optimizer (created if not provided)
            criterion: Optional loss function (CrossEntropyLoss if not provided)
        """
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)

        # Model
        self.model = model.to(self.device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer
        if optimizer is None:
            self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = optimizer

        # Loss function
        self.criterion = criterion or nn.CrossEntropyLoss()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision - use torch.cuda.amp for compatibility
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)  # type: ignore[attr-defined]

        # Training state
        self.history = TrainingHistory()
        self.current_metrics = TrainingMetrics()
        self.global_step = 0

        # Checkpointing
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        self.patience_counter = 0

        # Callbacks
        self.callbacks: list[Callable[[TrainingMetrics], None]] = []

    def _create_scheduler(self) -> Optional[SchedulerType]:
        """Create learning rate scheduler based on config."""
        scheduler_type = self.config.lr_scheduler.lower()

        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.lr_min,
            )
        if scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )
        if scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.lr_gamma,
                patience=self.config.lr_patience,
                min_lr=self.config.lr_min,
            )
        if scheduler_type == "warmup_cosine":
            return WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=self.config.lr_warmup_epochs,
                total_epochs=self.config.num_epochs,
                min_lr=self.config.lr_min,
            )
        if scheduler_type == "none":
            return None
        logger.warning(f"Unknown scheduler type: {scheduler_type}")
        return None

    def train(
        self,
        num_epochs: Optional[int] = None,
    ) -> TrainingHistory:
        """Run full training loop.

        Args:
            num_epochs: Number of epochs (uses config if None)

        Returns:
            Training history with all metrics
        """
        num_epochs = num_epochs or self.config.num_epochs

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.config.use_amp}")

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Training epoch
            train_metrics = self._train_epoch(epoch)

            # Validation
            val_metrics = self._validate(epoch)

            # Update metrics
            self.current_metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_accuracy=train_metrics["accuracy"],
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
                learning_rate=self._get_lr(),
                epoch_time=time.time() - epoch_start,
                best_val_loss=self.history.best_metrics.best_val_loss,
                best_epoch=self.history.best_metrics.best_epoch,
                total_steps=self.global_step,
            )

            # Update history
            self._update_history()

            # Learning rate scheduling
            self._step_scheduler(val_metrics["loss"])

            # Check for improvement
            is_best = val_metrics["loss"] < self.history.best_metrics.best_val_loss
            if is_best:
                self.history.best_metrics = TrainingMetrics(
                    epoch=epoch,
                    train_loss=train_metrics["loss"],
                    train_accuracy=train_metrics["accuracy"],
                    val_loss=val_metrics["loss"],
                    val_accuracy=val_metrics["accuracy"],
                    learning_rate=self._get_lr(),
                    epoch_time=self.current_metrics.epoch_time,
                    best_val_loss=val_metrics["loss"],
                    best_epoch=epoch,
                    total_steps=self.global_step,
                )
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Checkpointing
            self._checkpoint(epoch, is_best)

            # Callbacks
            for callback in self.callbacks:
                callback(self.current_metrics)

            # Logging
            self._log_epoch(epoch, train_metrics, val_metrics)

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        logger.info(f"Training complete. Best epoch: {self.history.best_metrics.best_epoch}")
        return self.history

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            images, tabular, labels = self._unpack_batch(batch)

            # Apply ablation mode
            images, tabular = self._apply_ablation(images, tabular)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):  # type: ignore[attr-defined]
                outputs = self.model(images, tabular)
                loss = self.criterion(outputs, labels)

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Step-level logging
            if (batch_idx + 1) % self.config.log_every_n_steps == 0:
                logger.debug(
                    f"Epoch {epoch}, Step {batch_idx + 1}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {"loss": avg_loss, "accuracy": accuracy}

    def _validate(self, epoch: int) -> dict[str, float]:  # noqa: ARG002
        """Validate model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images, tabular, labels = self._unpack_batch(batch)

                # Apply ablation mode
                images, tabular = self._apply_ablation(images, tabular)

                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):  # type: ignore[attr-defined]
                    outputs = self.model(images, tabular)
                    loss = self.criterion(outputs, labels)

                # Metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {"loss": avg_loss, "accuracy": accuracy}

    def _unpack_batch(
        self,
        batch: tuple,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Unpack batch and move to device.

        Args:
            batch: Tuple of (images, tabular, labels)

        Returns:
            Tuple of tensors on device
        """
        images, tabular, labels = batch

        if images is not None:
            images = images.to(self.device)
        tabular = tabular.to(self.device)
        labels = labels.to(self.device)

        return images, tabular, labels

    def _apply_ablation(
        self,
        images: Optional[torch.Tensor],
        tabular: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Apply ablation mode to inputs.

        Args:
            images: Image tensor
            tabular: Tabular tensor

        Returns:
            Modified tensors based on ablation mode
        """
        if self.config.ablation_mode == "image_only":
            # Zero out tabular features
            tabular = torch.zeros_like(tabular)
        elif self.config.ablation_mode == "tabular_only":
            # Zero out image features
            images = None

        return images, tabular

    def _get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def _step_scheduler(self, val_loss: float) -> None:
        """Step learning rate scheduler.

        Args:
            val_loss: Validation loss for plateau scheduler
        """
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _update_history(self) -> None:
        """Update training history."""
        self.history.train_losses.append(self.current_metrics.train_loss)
        self.history.train_accuracies.append(self.current_metrics.train_accuracy)
        self.history.val_losses.append(self.current_metrics.val_loss)
        self.history.val_accuracies.append(self.current_metrics.val_accuracy)
        self.history.learning_rates.append(self.current_metrics.learning_rate)
        self.history.epoch_times.append(self.current_metrics.epoch_time)

    def _checkpoint(self, epoch: int, is_best: bool) -> None:
        """Save checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.history.best_metrics.best_val_loss,
            "history": {
                "train_losses": self.history.train_losses,
                "val_losses": self.history.val_losses,
                "train_accuracies": self.history.train_accuracies,
                "val_accuracies": self.history.val_accuracies,
            },
            "config": self.config,
        }

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

        # Save periodic checkpoints
        if not self.config.save_best_only and (epoch + 1) % self.config.save_every_n_epochs == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Log epoch results.

        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        logger.info(
            f"Epoch {epoch + 1}/{self.config.num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"LR: {self._get_lr():.6f}, "
            f"Time: {self.current_metrics.epoch_time:.1f}s"
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def add_callback(self, callback: Callable[[TrainingMetrics], None]) -> None:
        """Add training callback.

        Args:
            callback: Function called after each epoch with metrics
        """
        self.callbacks.append(callback)


class WarmupCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Learning rate scheduler with warmup followed by cosine annealing."""

    def __init__(
        self,
        optimizer: OptimizerType,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        """Initialize scheduler.

        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Get current learning rate for each param group."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        # Cosine annealing
        progress = (self.last_epoch - self.warmup_epochs) / (
            self.total_epochs - self.warmup_epochs
        )
        return [
            self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


class AblationStudyRunner:
    """Run ablation studies to understand modality contributions."""

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
        base_config: Optional[TrainingConfig] = None,
    ):
        """Initialize ablation study runner.

        Args:
            model_factory: Function to create new model instances
            train_loader: Training data loader
            val_loader: Validation data loader
            base_config: Base training configuration
        """
        self.model_factory = model_factory
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.base_config = base_config or TrainingConfig()

    def run_ablations(self) -> dict[str, TrainingHistory]:
        """Run full ablation study.

        Returns:
            Dictionary mapping ablation mode to training history
        """
        ablation_modes = [None, "image_only", "tabular_only"]
        results: dict[str, TrainingHistory] = {}

        for mode in ablation_modes:
            mode_name = mode or "full_multimodal"
            logger.info(f"Running ablation: {mode_name}")

            # Create fresh model
            model = self.model_factory()

            # Create config with ablation mode
            config = TrainingConfig(
                **{**self.base_config.__dict__, "ablation_mode": mode}
            )
            config.checkpoint_dir = f"checkpoints/ablation_{mode_name}"

            # Create trainer
            trainer = MultiModalTrainingPipeline(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                config=config,
            )

            # Train
            history = trainer.train()
            results[mode_name] = history

            logger.info(
                f"Ablation {mode_name} complete. "
                f"Best val loss: {history.best_metrics.val_loss:.4f}"
            )

        # Print comparison
        self._print_comparison(results)

        return results

    def _print_comparison(self, results: dict[str, TrainingHistory]) -> None:
        """Print ablation comparison table."""
        logger.info("\n=== Ablation Study Results ===")
        logger.info(f"{'Mode':<20} {'Best Val Loss':<15} {'Best Val Acc':<15} {'Best Epoch':<10}")
        logger.info("-" * 60)

        for mode, history in results.items():
            best = history.best_metrics
            logger.info(
                f"{mode:<20} {best.val_loss:<15.4f} {best.val_accuracy:<15.4f} {best.best_epoch:<10}"
            )


def create_training_pipeline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config_overrides: Optional[dict[str, Any]] = None,
) -> MultiModalTrainingPipeline:
    """Factory function to create training pipeline.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config_overrides: Optional config overrides

    Returns:
        Configured training pipeline
    """
    config = TrainingConfig()
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return MultiModalTrainingPipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
