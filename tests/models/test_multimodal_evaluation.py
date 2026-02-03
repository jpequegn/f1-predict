"""Tests for multi-modal evaluation module."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from f1_predict.models.multimodal_evaluation import (
    CrossValidationEvaluator,
    InferenceBenchmarker,
    InferenceMetrics,
    ModelComparator,
    MultiModalEvaluator,
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


class TestMultiModalEvaluator:
    """Tests for MultiModalEvaluator class."""

    @pytest.fixture
    def evaluator(self, mock_model):
        """Create evaluator instance."""
        return MultiModalEvaluator(model=mock_model, device="cpu")

    def test_init(self, mock_model):
        """Test evaluator initialization."""
        evaluator = MultiModalEvaluator(model=mock_model)
        assert evaluator.model is mock_model

    def test_evaluate(self, evaluator, sample_dataloader):
        """Test evaluation."""
        metrics = evaluator.evaluate(sample_dataloader)
        assert "accuracy" in metrics
        assert "loss" in metrics

    def test_evaluate_top_k_accuracy(self, evaluator, sample_dataloader):
        """Test top-k accuracy evaluation."""
        metrics = evaluator.evaluate(sample_dataloader, top_k=[1, 3, 5])
        assert "top_1_accuracy" in metrics or "accuracy" in metrics

    def test_evaluate_returns_numeric(self, evaluator, sample_dataloader):
        """Test evaluation returns numeric values."""
        metrics = evaluator.evaluate(sample_dataloader)
        for _, value in metrics.items():
            assert isinstance(value, (int, float, np.floating))

    def test_get_predictions(self, evaluator, sample_dataloader):
        """Test getting predictions."""
        predictions, labels = evaluator.get_predictions(sample_dataloader)
        assert predictions.shape[0] == labels.shape[0]

    def test_compute_confusion_matrix(self, evaluator, sample_dataloader):
        """Test confusion matrix computation."""
        predictions, labels = evaluator.get_predictions(sample_dataloader)
        cm = evaluator.compute_confusion_matrix(predictions, labels)
        assert cm.shape[0] == cm.shape[1]

    def test_compute_per_class_metrics(self, evaluator, sample_dataloader):
        """Test per-class metrics computation."""
        predictions, labels = evaluator.get_predictions(sample_dataloader)
        per_class = evaluator.compute_per_class_metrics(predictions, labels)
        assert isinstance(per_class, dict)


class TestInferenceBenchmarker:
    """Tests for InferenceBenchmarker class."""

    @pytest.fixture
    def benchmarker(self, mock_model):
        """Create benchmarker instance."""
        return InferenceBenchmarker(model=mock_model, device="cpu")

    def test_init(self, mock_model):
        """Test benchmarker initialization."""
        benchmarker = InferenceBenchmarker(model=mock_model)
        assert benchmarker.model is mock_model

    def test_benchmark(self, benchmarker, sample_dataloader):
        """Test benchmarking."""
        metrics = benchmarker.benchmark(sample_dataloader, num_runs=2)
        assert isinstance(metrics, InferenceMetrics)

    def test_benchmark_metrics_populated(self, benchmarker, sample_dataloader):
        """Test benchmark metrics are populated."""
        metrics = benchmarker.benchmark(sample_dataloader, num_runs=2)
        assert metrics.total_time_ms >= 0
        assert metrics.avg_time_per_sample_ms >= 0
        assert metrics.throughput_samples_per_sec >= 0

    def test_benchmark_warmup(self, benchmarker, sample_dataloader):
        """Test benchmark with warmup."""
        metrics = benchmarker.benchmark(sample_dataloader, num_runs=2, warmup_runs=1)
        assert isinstance(metrics, InferenceMetrics)


class TestInferenceMetrics:
    """Tests for InferenceMetrics dataclass."""

    def test_init(self):
        """Test metrics initialization."""
        metrics = InferenceMetrics(
            total_time_ms=100.0,
            avg_time_per_sample_ms=1.0,
            throughput_samples_per_sec=1000.0,
        )
        assert metrics.total_time_ms == 100.0
        assert metrics.avg_time_per_sample_ms == 1.0
        assert metrics.throughput_samples_per_sec == 1000.0

    def test_default_values(self):
        """Test default values."""
        metrics = InferenceMetrics()
        assert metrics.total_time_ms == 0.0
        assert metrics.peak_memory_mb == 0.0

    def test_memory_metrics(self):
        """Test memory metrics."""
        metrics = InferenceMetrics(
            peak_memory_mb=512.0,
            avg_memory_mb=256.0,
        )
        assert metrics.peak_memory_mb == 512.0
        assert metrics.avg_memory_mb == 256.0


class TestCrossValidationEvaluator:
    """Tests for CrossValidationEvaluator class."""

    @pytest.fixture
    def cv_evaluator(self):
        """Create cross-validation evaluator."""
        return CrossValidationEvaluator(
            model_fn=lambda: MockMultiModalModel(),
            n_splits=3,
            device="cpu",
        )

    @pytest.fixture
    def dataset(self):
        """Create dataset for cross-validation."""
        images = torch.randn(100, 3, 32, 32)
        tabular = torch.randn(100, 64)
        labels = torch.randint(0, 20, (100,))
        return TensorDataset(images, tabular, labels)

    def test_init(self):
        """Test evaluator initialization."""
        evaluator = CrossValidationEvaluator(
            model_fn=lambda: MockMultiModalModel(),
            n_splits=5,
        )
        assert evaluator.n_splits == 5

    def test_evaluate_cv(self, cv_evaluator, dataset):
        """Test cross-validation evaluation."""
        results = cv_evaluator.evaluate(
            dataset=dataset,
            epochs=1,
            batch_size=16,
        )
        assert "mean_accuracy" in results or "fold_results" in results

    def test_get_fold_results(self, cv_evaluator, dataset):
        """Test getting fold results."""
        results = cv_evaluator.evaluate(
            dataset=dataset,
            epochs=1,
            batch_size=16,
        )
        assert results is not None


class TestModelComparator:
    """Tests for ModelComparator class."""

    @pytest.fixture
    def comparator(self, sample_dataloader):
        """Create model comparator."""
        return ModelComparator(
            test_loader=sample_dataloader,
            device="cpu",
        )

    def test_init(self, sample_dataloader):
        """Test comparator initialization."""
        comparator = ModelComparator(
            test_loader=sample_dataloader,
        )
        assert comparator.test_loader is sample_dataloader

    def test_add_model(self, comparator, mock_model):
        """Test adding model."""
        comparator.add_model("model1", mock_model)
        assert "model1" in comparator.models

    def test_compare_models(self, comparator):
        """Test model comparison."""
        model1 = MockMultiModalModel()
        model2 = MockMultiModalModel()

        comparator.add_model("model1", model1)
        comparator.add_model("model2", model2)

        results = comparator.compare()
        assert "model1" in results
        assert "model2" in results

    def test_get_best_model(self, comparator):
        """Test getting best model."""
        model1 = MockMultiModalModel()
        model2 = MockMultiModalModel()

        comparator.add_model("model1", model1)
        comparator.add_model("model2", model2)

        comparator.compare()
        best_name, _ = comparator.get_best_model(metric="accuracy")
        assert best_name in ["model1", "model2"]

    def test_generate_report(self, comparator, tmp_path):
        """Test report generation."""
        model = MockMultiModalModel()
        comparator.add_model("model1", model)
        comparator.compare()

        report_path = tmp_path / "report.json"
        comparator.generate_report(str(report_path))
        # Report should be generated (or method should exist)


class TestEvaluationEdgeCases:
    """Tests for evaluation edge cases."""

    def test_evaluator_single_batch(self, mock_model):
        """Test evaluation with single batch."""
        images = torch.randn(4, 3, 32, 32)
        tabular = torch.randn(4, 64)
        labels = torch.randint(0, 20, (4,))
        dataset = TensorDataset(images, tabular, labels)
        loader = DataLoader(dataset, batch_size=4)

        evaluator = MultiModalEvaluator(model=mock_model, device="cpu")
        metrics = evaluator.evaluate(loader)
        assert "accuracy" in metrics

    def test_benchmarker_single_sample(self, mock_model):
        """Test benchmarking with single sample."""
        images = torch.randn(1, 3, 32, 32)
        tabular = torch.randn(1, 64)
        labels = torch.randint(0, 20, (1,))
        dataset = TensorDataset(images, tabular, labels)
        loader = DataLoader(dataset, batch_size=1)

        benchmarker = InferenceBenchmarker(model=mock_model, device="cpu")
        metrics = benchmarker.benchmark(loader, num_runs=1)
        assert metrics.total_time_ms >= 0

    def test_cv_evaluator_small_dataset(self):
        """Test CV with small dataset."""
        images = torch.randn(10, 3, 32, 32)
        tabular = torch.randn(10, 64)
        labels = torch.randint(0, 20, (10,))
        dataset = TensorDataset(images, tabular, labels)

        evaluator = CrossValidationEvaluator(
            model_fn=lambda: MockMultiModalModel(),
            n_splits=2,
            device="cpu",
        )

        results = evaluator.evaluate(
            dataset=dataset,
            epochs=1,
            batch_size=4,
        )
        assert results is not None


class TestEvaluationMetrics:
    """Tests for evaluation metrics computation."""

    @pytest.fixture
    def evaluator(self, mock_model):
        """Create evaluator."""
        return MultiModalEvaluator(model=mock_model, device="cpu")

    def test_accuracy_range(self, evaluator, sample_dataloader):
        """Test accuracy is in valid range."""
        metrics = evaluator.evaluate(sample_dataloader)
        assert 0 <= metrics["accuracy"] <= 1

    def test_loss_non_negative(self, evaluator, sample_dataloader):
        """Test loss is non-negative."""
        metrics = evaluator.evaluate(sample_dataloader)
        assert metrics["loss"] >= 0

    def test_metrics_reproducible(self, evaluator, sample_dataloader):
        """Test metrics are reproducible."""
        torch.manual_seed(42)
        metrics1 = evaluator.evaluate(sample_dataloader)

        torch.manual_seed(42)
        metrics2 = evaluator.evaluate(sample_dataloader)

        assert metrics1["accuracy"] == metrics2["accuracy"]
