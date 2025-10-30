"""Tests for MLflow integration."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from f1_predict.optimization.mlflow_integration import MLflowTracker


@pytest.fixture
def sample_params():
    """Sample hyperparameters."""
    return {
        "learning_rate": 0.01,
        "max_depth": 5,
        "n_estimators": 100,
    }


@pytest.fixture
def sample_metrics():
    """Sample metrics."""
    return {
        "rmse": 0.5,
        "mae": 0.3,
        "r2": 0.85,
    }


class TestMLflowTrackerInit:
    """Test MLflowTracker initialization."""

    @patch("f1_predict.optimization.mlflow_integration.HAS_MLFLOW", True)
    @patch("f1_predict.optimization.mlflow_integration.mlflow")
    def test_init_with_mlflow(self, mock_mlflow):
        """Test initialization when MLflow is available."""
        tracker = MLflowTracker(
            experiment_name="test_experiment",
            tracking_uri="./test_mlruns",
        )

        assert tracker.experiment_name == "test_experiment"
        assert tracker.tracking_uri == "./test_mlruns"
        mock_mlflow.set_tracking_uri.assert_called_once_with("./test_mlruns")
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")

    @patch("f1_predict.optimization.mlflow_integration.HAS_MLFLOW", False)
    def test_init_without_mlflow(self):
        """Test initialization when MLflow is not available."""
        tracker = MLflowTracker(experiment_name="test_experiment")

        assert tracker.experiment_name == "test_experiment"
        # Should not raise exception even without MLflow

    @patch("f1_predict.optimization.mlflow_integration.HAS_MLFLOW", True)
    @patch("f1_predict.optimization.mlflow_integration.mlflow")
    def test_init_default_tracking_uri(self, mock_mlflow):
        """Test initialization uses default tracking URI."""
        tracker = MLflowTracker(experiment_name="test_experiment")

        assert tracker.tracking_uri == "./mlruns"
        mock_mlflow.set_tracking_uri.assert_called_once_with("./mlruns")


class TestLogTrial:
    """Test log_trial method."""

    @patch("f1_predict.optimization.mlflow_integration.HAS_MLFLOW", True)
    @patch("f1_predict.optimization.mlflow_integration.mlflow")
    def test_log_trial_success(self, mock_mlflow, sample_params, sample_metrics):
        """Test successful trial logging."""
        tracker = MLflowTracker(experiment_name="test_experiment")

        # Mock context manager
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        tracker.log_trial(
            trial_number=1,
            params=sample_params,
            metrics=sample_metrics,
        )

        mock_mlflow.start_run.assert_called_once_with(run_name="trial_1")
        mock_mlflow.log_params.assert_called_once_with(sample_params)
        mock_mlflow.log_metrics.assert_called_once_with(sample_metrics)

    @patch("f1_predict.optimization.mlflow_integration.HAS_MLFLOW", False)
    def test_log_trial_without_mlflow(self, sample_params, sample_metrics):
        """Test log_trial doesn't raise when MLflow unavailable."""
        tracker = MLflowTracker(experiment_name="test_experiment")

        # Should not raise exception
        tracker.log_trial(
            trial_number=1,
            params=sample_params,
            metrics=sample_metrics,
        )

    @patch("f1_predict.optimization.mlflow_integration.HAS_MLFLOW", True)
    @patch("f1_predict.optimization.mlflow_integration.mlflow")
    def test_log_trial_exception_handling(self, mock_mlflow, sample_params, sample_metrics):
        """Test log_trial handles exceptions gracefully."""
        tracker = MLflowTracker(experiment_name="test_experiment")

        # Simulate MLflow error
        mock_mlflow.start_run.side_effect = Exception("MLflow error")

        # Should not raise exception
        tracker.log_trial(
            trial_number=1,
            params=sample_params,
            metrics=sample_metrics,
        )


class TestLogBestTrial:
    """Test log_best_trial method."""

    @patch("f1_predict.optimization.mlflow_integration.HAS_MLFLOW", True)
    @patch("f1_predict.optimization.mlflow_integration.mlflow")
    def test_log_best_trial_success(self, mock_mlflow, sample_params, sample_metrics):
        """Test successful best trial logging."""
        tracker = MLflowTracker(experiment_name="test_experiment")

        # Mock context manager
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        tracker.log_best_trial(
            best_params=sample_params,
            best_metrics=sample_metrics,
        )

        mock_mlflow.start_run.assert_called_once_with(run_name="best_trial")
        mock_mlflow.log_params.assert_called_once_with(sample_params)
        mock_mlflow.log_metrics.assert_called_once_with(sample_metrics)
        mock_mlflow.log_dict.assert_called_once_with(sample_params, "best_params.json")

    @patch("f1_predict.optimization.mlflow_integration.HAS_MLFLOW", True)
    @patch("f1_predict.optimization.mlflow_integration.mlflow")
    def test_log_best_trial_with_model(self, mock_mlflow, sample_params, sample_metrics):
        """Test best trial logging with model."""
        tracker = MLflowTracker(experiment_name="test_experiment")

        # Mock context manager
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        # Mock model
        mock_model = MagicMock()

        tracker.log_best_trial(
            best_params=sample_params,
            best_metrics=sample_metrics,
            best_model=mock_model,
        )

        mock_mlflow.sklearn.log_model.assert_called_once_with(mock_model, "model")

    @patch("f1_predict.optimization.mlflow_integration.HAS_MLFLOW", True)
    @patch("f1_predict.optimization.mlflow_integration.mlflow")
    def test_log_best_trial_model_logging_fails(self, mock_mlflow, sample_params, sample_metrics):
        """Test best trial when model logging fails."""
        tracker = MLflowTracker(experiment_name="test_experiment")

        # Mock context manager
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        # Model logging fails
        mock_mlflow.sklearn.log_model.side_effect = Exception("Model logging error")
        mock_model = MagicMock()

        # Should not raise exception
        tracker.log_best_trial(
            best_params=sample_params,
            best_metrics=sample_metrics,
            best_model=mock_model,
        )

    @patch("f1_predict.optimization.mlflow_integration.HAS_MLFLOW", False)
    def test_log_best_trial_without_mlflow(self, sample_params, sample_metrics):
        """Test log_best_trial doesn't raise when MLflow unavailable."""
        tracker = MLflowTracker(experiment_name="test_experiment")

        # Should not raise exception
        tracker.log_best_trial(
            best_params=sample_params,
            best_metrics=sample_metrics,
        )


class TestSaveParamsLocally:
    """Test save_params_locally method."""

    def test_save_params_creates_file(self, tmp_path, sample_params):
        """Test save_params_locally creates JSON file."""
        tracker = MLflowTracker(experiment_name="test_experiment")
        filepath = tmp_path / "best_params.json"

        tracker.save_params_locally(sample_params, filepath)

        assert filepath.exists()
        with open(filepath) as f:
            loaded_params = json.load(f)
        assert loaded_params == sample_params

    def test_save_params_creates_directory(self, tmp_path, sample_params):
        """Test save_params_locally creates parent directory."""
        tracker = MLflowTracker(experiment_name="test_experiment")
        filepath = tmp_path / "subdir" / "best_params.json"

        tracker.save_params_locally(sample_params, filepath)

        assert filepath.exists()
        assert filepath.parent.exists()

    def test_save_params_overwrites_existing(self, tmp_path, sample_params):
        """Test save_params_locally overwrites existing file."""
        tracker = MLflowTracker(experiment_name="test_experiment")
        filepath = tmp_path / "best_params.json"

        # Create initial file
        with open(filepath, "w") as f:
            json.dump({"old": "data"}, f)

        # Save new params
        tracker.save_params_locally(sample_params, filepath)

        with open(filepath) as f:
            loaded_params = json.load(f)
        assert loaded_params == sample_params

    def test_save_params_handles_errors(self, sample_params):
        """Test save_params_locally handles errors gracefully."""
        tracker = MLflowTracker(experiment_name="test_experiment")

        # Invalid path
        filepath = Path("/invalid/path/best_params.json")

        # Should not raise exception
        tracker.save_params_locally(sample_params, filepath)
