"""Tests for model evaluation functionality."""

import numpy as np
import pandas as pd
import pytest

from f1_predict.models.baseline import RuleBasedPredictor
from f1_predict.models.evaluation import ModelEvaluator
from f1_predict.models.logistic import LogisticRacePredictor


@pytest.fixture
def sample_predictions():
    """Create sample prediction data."""
    return {
        "y_true": np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
        "y_pred": np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
        "y_proba": np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.1]),
    }


@pytest.fixture
def sample_features():
    """Create sample feature data."""
    return pd.DataFrame(
        {
            "driver_id": [f"driver{i}" for i in range(30)],
            "qualifying_position": list(range(1, 31)),
            "driver_form_score": [90.0 - i * 2 for i in range(30)],
            "team_reliability_score": [85.0 - i * 1.5 for i in range(30)],
            "circuit_performance_score": [88.0 - i * 2 for i in range(30)],
        }
    )


@pytest.fixture
def sample_race_results():
    """Create sample race results."""
    return pd.DataFrame({"position": list(range(1, 31))})


class TestModelEvaluator:
    """Tests for ModelEvaluator."""

    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator()
        assert evaluator.n_splits == 5
        assert evaluator.random_state == 42

    def test_initialization_custom_splits(self):
        """Test initialization with custom splits."""
        evaluator = ModelEvaluator(n_splits=10)
        assert evaluator.n_splits == 10

    def test_evaluate_basic(self, sample_predictions):
        """Test basic evaluation functionality."""
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(
            sample_predictions["y_true"],
            sample_predictions["y_pred"],
            sample_predictions["y_proba"],
        )

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert "support" in metrics
        assert "positive_ratio" in metrics

        # Check metric values are in valid ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

    def test_evaluate_without_probabilities(self, sample_predictions):
        """Test evaluation without probability scores."""
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(
            sample_predictions["y_true"], sample_predictions["y_pred"]
        )

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" not in metrics

    def test_evaluate_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        evaluator = ModelEvaluator()
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1])

        metrics = evaluator.evaluate(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_cross_validate_logistic(self, sample_features, sample_race_results):
        """Test cross-validation with logistic model."""
        evaluator = ModelEvaluator(n_splits=5)
        model = LogisticRacePredictor(target="podium")

        results = evaluator.cross_validate(model, sample_features, sample_race_results)

        assert "mean_metrics" in results
        assert "std_metrics" in results
        assert "fold_metrics" in results
        assert "n_splits" in results
        assert "total_samples" in results

        # Check mean metrics
        assert "accuracy" in results["mean_metrics"]
        assert "precision" in results["mean_metrics"]
        assert "recall" in results["mean_metrics"]
        assert "f1_score" in results["mean_metrics"]

        # Check fold metrics
        assert len(results["fold_metrics"]) == 5

    def test_cross_validate_rule_based(self, sample_features, sample_race_results):
        """Test cross-validation with rule-based model."""
        evaluator = ModelEvaluator(n_splits=5)
        model = RuleBasedPredictor()

        results = evaluator.cross_validate(model, sample_features, sample_race_results)

        assert "mean_metrics" in results
        assert "std_metrics" in results
        assert len(results["fold_metrics"]) == 5

    def test_cross_validate_insufficient_samples(self):
        """Test cross-validation with too few samples."""
        evaluator = ModelEvaluator(n_splits=5)
        model = LogisticRacePredictor()

        small_features = pd.DataFrame(
            {
                "driver_id": ["d1", "d2", "d3"],
                "qualifying_position": [1, 2, 3],
                "driver_form_score": [80, 70, 60],
                "team_reliability_score": [85, 75, 65],
                "circuit_performance_score": [80, 70, 60],
            }
        )
        small_results = pd.DataFrame({"position": [1, 2, 3]})

        with pytest.raises(ValueError, match="Not enough samples"):
            evaluator.cross_validate(model, small_features, small_results)

    def test_evaluate_confidence_calibration(self):
        """Test confidence calibration evaluation."""
        evaluator = ModelEvaluator()

        # Create well-calibrated predictions
        y_true = np.array([1] * 50 + [0] * 50)
        y_proba = np.concatenate([np.linspace(0.5, 1.0, 50), np.linspace(0.0, 0.5, 50)])

        results = evaluator.evaluate_confidence_calibration(y_true, y_proba, n_bins=5)

        assert "expected_calibration_error" in results
        assert "n_bins" in results
        assert "bin_metrics" in results

        # ECE should be relatively low for well-calibrated predictions
        assert results["expected_calibration_error"] < 0.3

    def test_evaluate_confidence_calibration_perfect(self):
        """Test calibration with perfectly calibrated predictions."""
        evaluator = ModelEvaluator()

        # Perfect calibration: 70% predictions at 0.7 probability
        y_true = np.array([1] * 7 + [0] * 3)
        y_proba = np.array([0.7] * 10)

        results = evaluator.evaluate_confidence_calibration(y_true, y_proba, n_bins=10)

        # Should have low ECE for perfect calibration
        assert results["expected_calibration_error"] < 0.1

    def test_compare_models(self, sample_features, sample_race_results):
        """Test model comparison functionality."""
        evaluator = ModelEvaluator(n_splits=3)  # Use fewer splits for speed

        models = {
            "rule_based": RuleBasedPredictor(),
            "logistic_podium": LogisticRacePredictor(target="podium"),
            "logistic_points": LogisticRacePredictor(target="points"),
        }

        comparison_df = evaluator.compare_models(
            models, sample_features, sample_race_results
        )

        assert len(comparison_df) == 3
        assert "model" in comparison_df.columns
        assert "accuracy_mean" in comparison_df.columns
        assert "f1_mean" in comparison_df.columns

        # Check sorted by F1 score
        f1_scores = comparison_df["f1_mean"].values
        assert all(f1_scores[i] >= f1_scores[i + 1] for i in range(len(f1_scores) - 1))

    def test_compare_models_empty_dict(self, sample_features, sample_race_results):
        """Test model comparison with no models."""
        evaluator = ModelEvaluator()
        comparison_df = evaluator.compare_models({}, sample_features, sample_race_results)

        assert comparison_df.empty

    def test_evaluate_with_pandas_series(self):
        """Test evaluation with pandas Series inputs."""
        evaluator = ModelEvaluator()

        y_true = pd.Series([1, 1, 0, 0, 1])
        y_pred = pd.Series([1, 0, 0, 0, 1])
        y_proba = pd.Series([0.9, 0.6, 0.3, 0.2, 0.8])

        metrics = evaluator.evaluate(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert metrics["support"] == 5.0