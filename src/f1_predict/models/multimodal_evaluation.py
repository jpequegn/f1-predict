"""Evaluation and benchmarking for multi-modal F1 prediction models.

Provides comprehensive evaluation metrics, comparison between multi-modal
and single-modal approaches, ablation analysis, and inference benchmarking.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    # Classification metrics
    accuracy: float = 0.0
    top_3_accuracy: float = 0.0
    top_5_accuracy: float = 0.0

    # Regression metrics (for position prediction)
    mae: float = 0.0  # Mean Absolute Error
    rmse: float = 0.0  # Root Mean Square Error
    median_error: float = 0.0

    # Additional metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Confidence metrics
    avg_confidence: float = 0.0
    confidence_calibration: float = 0.0

    # Sample counts
    total_samples: int = 0
    correct_predictions: int = 0


@dataclass
class InferenceMetrics:
    """Container for inference performance metrics."""

    # Timing
    total_time_ms: float = 0.0
    avg_time_per_sample_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0

    # Memory
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0

    # Model info
    num_parameters: int = 0
    model_size_mb: float = 0.0


@dataclass
class AblationResults:
    """Results from ablation study."""

    full_multimodal: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    image_only: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    tabular_only: EvaluationMetrics = field(default_factory=EvaluationMetrics)

    # Computed improvements
    image_contribution: float = 0.0  # How much images improve over tabular-only
    tabular_contribution: float = 0.0  # How much tabular improves over image-only
    fusion_synergy: float = 0.0  # Extra improvement from fusion


class MultiModalEvaluator:
    """Comprehensive evaluator for multi-modal models.

    Computes various metrics and provides comparison tools
    for understanding model performance.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize evaluator.

        Args:
            model: Model to evaluate
            device: Device to use
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        dataloader: DataLoader,
        num_classes: int = 20,
    ) -> EvaluationMetrics:
        """Evaluate model on dataset.

        Args:
            dataloader: Data loader
            num_classes: Number of output classes

        Returns:
            EvaluationMetrics with computed metrics
        """
        all_predictions: list[int] = []
        all_labels: list[int] = []
        all_probs: list[np.ndarray] = []

        with torch.no_grad():
            for batch in dataloader:
                images, tabular, labels = self._unpack_batch(batch)

                outputs = self.model(images, tabular)
                probs = torch.softmax(outputs, dim=1)

                predictions = outputs.argmax(dim=1)

                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(probs.cpu().numpy())

        # Compute metrics
        metrics = self._compute_metrics(
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probs),
            num_classes,
        )

        return metrics

    def _unpack_batch(
        self,
        batch: tuple,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Unpack and move batch to device."""
        images, tabular, labels = batch

        if images is not None:
            images = images.to(self.device)
        tabular = tabular.to(self.device)
        labels = labels.to(self.device)

        return images, tabular, labels

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probs: np.ndarray,
        num_classes: int,
    ) -> EvaluationMetrics:
        """Compute evaluation metrics.

        Args:
            predictions: Predicted class indices
            labels: True class indices
            probs: Prediction probabilities
            num_classes: Number of classes

        Returns:
            EvaluationMetrics
        """
        metrics = EvaluationMetrics()
        metrics.total_samples = len(labels)

        # Accuracy
        correct = (predictions == labels).sum()
        metrics.correct_predictions = int(correct)
        metrics.accuracy = correct / len(labels)

        # Top-k accuracy
        top_3_correct = sum(
            labels[i] in np.argsort(probs[i])[-3:]
            for i in range(len(labels))
        )
        top_5_correct = sum(
            labels[i] in np.argsort(probs[i])[-5:]
            for i in range(len(labels))
        )
        metrics.top_3_accuracy = top_3_correct / len(labels)
        metrics.top_5_accuracy = top_5_correct / len(labels)

        # Position-based metrics (treating as regression)
        errors = np.abs(predictions - labels)
        metrics.mae = float(errors.mean())
        metrics.rmse = float(np.sqrt((errors ** 2).mean()))
        metrics.median_error = float(np.median(errors))

        # Confidence metrics
        max_probs = probs.max(axis=1)
        metrics.avg_confidence = float(max_probs.mean())

        # Calibration: how often confidence matches accuracy
        # Group by confidence bins
        bins = np.linspace(0, 1, 11)
        calibration_errors = []
        for i in range(len(bins) - 1):
            mask = (max_probs >= bins[i]) & (max_probs < bins[i + 1])
            if mask.sum() > 0:
                bin_confidence = max_probs[mask].mean()
                bin_accuracy = (predictions[mask] == labels[mask]).mean()
                calibration_errors.append(abs(bin_confidence - bin_accuracy))
        metrics.confidence_calibration = float(np.mean(calibration_errors)) if calibration_errors else 0.0

        # Precision, Recall, F1 (macro average)
        precision_per_class = []
        recall_per_class = []
        for c in range(num_classes):
            pred_c = predictions == c
            true_c = labels == c
            tp = (pred_c & true_c).sum()
            fp = (pred_c & ~true_c).sum()
            fn = (~pred_c & true_c).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precision_per_class.append(precision)
            recall_per_class.append(recall)

        metrics.precision = float(np.mean(precision_per_class))
        metrics.recall = float(np.mean(recall_per_class))

        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (
                metrics.precision + metrics.recall
            )

        return metrics


class InferenceBenchmarker:
    """Benchmark inference performance of models.

    Measures timing, memory usage, and throughput.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize benchmarker.

        Args:
            model: Model to benchmark
            device: Device to use
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def benchmark(
        self,
        dataloader: DataLoader,
        warmup_batches: int = 5,
        num_runs: int = 3,
    ) -> InferenceMetrics:
        """Run inference benchmark.

        Args:
            dataloader: Data loader for benchmarking
            warmup_batches: Number of warmup batches
            num_runs: Number of benchmark runs

        Returns:
            InferenceMetrics with timing and memory data
        """
        metrics = InferenceMetrics()

        # Model info
        metrics.num_parameters = sum(p.numel() for p in self.model.parameters())
        metrics.model_size_mb = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024 * 1024)

        # Warmup
        logger.info(f"Running {warmup_batches} warmup batches...")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= warmup_batches:
                    break
                images, tabular, _ = self._unpack_batch(batch)
                _ = self.model(images, tabular)

        # Synchronize if using CUDA
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Benchmark runs
        all_times = []
        total_samples = 0

        for run in range(num_runs):
            logger.info(f"Benchmark run {run + 1}/{num_runs}")
            run_start = time.perf_counter()
            samples_in_run = 0

            with torch.no_grad():
                for batch in dataloader:
                    images, tabular, _ = self._unpack_batch(batch)
                    batch_size = tabular.shape[0]

                    _ = self.model(images, tabular)

                    samples_in_run += batch_size

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            run_time = (time.perf_counter() - run_start) * 1000  # ms
            all_times.append(run_time)
            total_samples = samples_in_run

        # Compute metrics
        avg_run_time = float(np.mean(all_times))
        metrics.total_time_ms = avg_run_time
        metrics.avg_time_per_sample_ms = avg_run_time / total_samples if total_samples > 0 else 0.0
        metrics.throughput_samples_per_sec = (
            (total_samples / (avg_run_time / 1000)) if avg_run_time > 0 else 0.0
        )

        # Memory metrics
        if self.device.type == "cuda":
            metrics.peak_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            metrics.avg_memory_mb = float(torch.cuda.memory_allocated() / (1024 * 1024))

        return metrics

    def _unpack_batch(
        self,
        batch: tuple,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Unpack and move batch to device."""
        images, tabular, labels = batch

        if images is not None:
            images = images.to(self.device)
        tabular = tabular.to(self.device)
        labels = labels.to(self.device)

        return images, tabular, labels


class CrossValidationEvaluator:
    """K-fold cross-validation for multi-modal models."""

    def __init__(
        self,
        model_factory: Any,
        n_folds: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize cross-validation evaluator.

        Args:
            model_factory: Factory function to create models
            n_folds: Number of folds
            device: Device to use
        """
        self.model_factory = model_factory
        self.n_folds = n_folds
        self.device = device

    def cross_validate(
        self,
        dataset: Any,
        train_fn: Any,
        num_classes: int = 20,
    ) -> dict[str, Any]:
        """Run cross-validation.

        Args:
            dataset: Full dataset
            train_fn: Function to train model given train loader
            num_classes: Number of classes

        Returns:
            Dictionary with fold results and aggregate metrics
        """
        from torch.utils.data import Subset

        # Create fold indices
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        fold_size = len(indices) // self.n_folds

        fold_metrics: list[EvaluationMetrics] = []

        for fold in range(self.n_folds):
            logger.info(f"Cross-validation fold {fold + 1}/{self.n_folds}")

            # Split indices
            val_start = fold * fold_size
            val_end = val_start + fold_size
            val_indices = indices[val_start:val_end]
            train_indices = indices[:val_start] + indices[val_end:]

            # Create subsets
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)

            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32)

            # Create and train model
            model = self.model_factory()
            train_fn(model, train_loader)

            # Evaluate
            evaluator = MultiModalEvaluator(model, self.device)
            metrics = evaluator.evaluate(val_loader, num_classes)
            fold_metrics.append(metrics)

            logger.info(f"Fold {fold + 1} accuracy: {metrics.accuracy:.4f}")

        # Aggregate results
        results = {
            "fold_metrics": fold_metrics,
            "mean_accuracy": np.mean([m.accuracy for m in fold_metrics]),
            "std_accuracy": np.std([m.accuracy for m in fold_metrics]),
            "mean_mae": np.mean([m.mae for m in fold_metrics]),
            "std_mae": np.std([m.mae for m in fold_metrics]),
        }

        logger.info(
            f"Cross-validation complete: "
            f"Accuracy = {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}"
        )

        return results


class ModelComparator:
    """Compare multiple models on the same dataset."""

    def __init__(
        self,
        models: dict[str, nn.Module],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize model comparator.

        Args:
            models: Dictionary mapping model names to models
            device: Device to use
        """
        self.models = models
        self.device = device

    def compare(
        self,
        dataloader: DataLoader,
        num_classes: int = 20,
    ) -> dict[str, EvaluationMetrics]:
        """Compare all models on the same data.

        Args:
            dataloader: Data loader
            num_classes: Number of classes

        Returns:
            Dictionary mapping model names to metrics
        """
        results: dict[str, EvaluationMetrics] = {}

        for name, model in self.models.items():
            logger.info(f"Evaluating model: {name}")
            evaluator = MultiModalEvaluator(model, self.device)
            metrics = evaluator.evaluate(dataloader, num_classes)
            results[name] = metrics

            logger.info(
                f"{name}: Accuracy={metrics.accuracy:.4f}, "
                f"MAE={metrics.mae:.2f}, "
                f"Top-3={metrics.top_3_accuracy:.4f}"
            )

        return results

    def generate_comparison_table(
        self,
        results: dict[str, EvaluationMetrics],
    ) -> str:
        """Generate comparison table as string.

        Args:
            results: Results dictionary

        Returns:
            Formatted table string
        """
        lines = [
            "=" * 80,
            f"{'Model':<20} {'Accuracy':>10} {'Top-3':>10} {'MAE':>8} {'RMSE':>8} {'F1':>8}",
            "-" * 80,
        ]

        for name, metrics in results.items():
            lines.append(
                f"{name:<20} "
                f"{metrics.accuracy:>10.4f} "
                f"{metrics.top_3_accuracy:>10.4f} "
                f"{metrics.mae:>8.2f} "
                f"{metrics.rmse:>8.2f} "
                f"{metrics.f1_score:>8.4f}"
            )

        lines.append("=" * 80)

        return "\n".join(lines)


def run_ablation_study(
    full_model: nn.Module,
    dataloader: DataLoader,
    image_only_model: Optional[nn.Module] = None,
    tabular_only_model: Optional[nn.Module] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_classes: int = 20,
) -> AblationResults:
    """Run ablation study comparing multi-modal to single-modal.

    Args:
        full_model: Full multi-modal model
        dataloader: Test data loader
        image_only_model: Optional image-only model
        tabular_only_model: Optional tabular-only model
        device: Device to use
        num_classes: Number of classes

    Returns:
        AblationResults with comparison data
    """
    results = AblationResults()

    # Evaluate full model
    logger.info("Evaluating full multi-modal model...")
    evaluator = MultiModalEvaluator(full_model, device)
    results.full_multimodal = evaluator.evaluate(dataloader, num_classes)

    # Evaluate image-only if provided
    if image_only_model is not None:
        logger.info("Evaluating image-only model...")
        evaluator = MultiModalEvaluator(image_only_model, device)
        results.image_only = evaluator.evaluate(dataloader, num_classes)

    # Evaluate tabular-only if provided
    if tabular_only_model is not None:
        logger.info("Evaluating tabular-only model...")
        evaluator = MultiModalEvaluator(tabular_only_model, device)
        results.tabular_only = evaluator.evaluate(dataloader, num_classes)

    # Compute contributions
    if image_only_model and tabular_only_model:
        # How much images add over tabular
        results.image_contribution = (
            results.full_multimodal.accuracy - results.tabular_only.accuracy
        )

        # How much tabular adds over images
        results.tabular_contribution = (
            results.full_multimodal.accuracy - results.image_only.accuracy
        )

        # Synergy: extra improvement from fusion
        max_single = max(results.image_only.accuracy, results.tabular_only.accuracy)
        results.fusion_synergy = results.full_multimodal.accuracy - max_single

    return results


def save_evaluation_report(
    metrics: EvaluationMetrics,
    inference: InferenceMetrics,
    output_path: str,
) -> None:
    """Save evaluation report to file.

    Args:
        metrics: Evaluation metrics
        inference: Inference metrics
        output_path: Path to save report
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    report = f"""
Multi-Modal Model Evaluation Report
===================================

Classification Metrics
----------------------
Accuracy:       {metrics.accuracy:.4f}
Top-3 Accuracy: {metrics.top_3_accuracy:.4f}
Top-5 Accuracy: {metrics.top_5_accuracy:.4f}
Precision:      {metrics.precision:.4f}
Recall:         {metrics.recall:.4f}
F1 Score:       {metrics.f1_score:.4f}

Position Prediction Metrics
---------------------------
Mean Absolute Error: {metrics.mae:.2f} positions
RMSE:                {metrics.rmse:.2f} positions
Median Error:        {metrics.median_error:.2f} positions

Confidence Metrics
------------------
Average Confidence:      {metrics.avg_confidence:.4f}
Calibration Error:       {metrics.confidence_calibration:.4f}

Inference Performance
---------------------
Total Time:              {inference.total_time_ms:.2f} ms
Time per Sample:         {inference.avg_time_per_sample_ms:.3f} ms
Throughput:              {inference.throughput_samples_per_sec:.1f} samples/sec
Model Parameters:        {inference.num_parameters:,}
Model Size:              {inference.model_size_mb:.2f} MB
Peak Memory:             {inference.peak_memory_mb:.2f} MB

Sample Statistics
-----------------
Total Samples:      {metrics.total_samples}
Correct:            {metrics.correct_predictions}
"""

    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"Saved evaluation report to {output_path}")
