"""MLflow integration for experiment tracking."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Track hyperparameter optimization experiments in MLflow."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
    ) -> None:
        """Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server. Defaults to "./mlruns"
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "./mlruns"

        if HAS_MLFLOW:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment_name)
        else:
            logger.warning("MLflow not available, logging disabled")

    def log_trial(
        self,
        trial_number: int,
        params: dict[str, Any],
        metrics: dict[str, float],
    ) -> None:
        """Log a single trial to MLflow.

        Args:
            trial_number: Trial number for run naming
            params: Hyperparameters used in this trial
            metrics: Evaluation metrics from this trial
        """
        if not HAS_MLFLOW:
            return

        try:
            with mlflow.start_run(run_name=f"trial_{trial_number}"):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
        except Exception as e:
            logger.warning(f"Failed to log trial to MLflow: {e}")

    def log_best_trial(
        self,
        best_params: dict[str, Any],
        best_metrics: dict[str, float],
        best_model: Optional[Any] = None,
    ) -> None:
        """Log best trial results to MLflow.

        Args:
            best_params: Best hyperparameters found
            best_metrics: Best evaluation metrics
            best_model: Best model object (optional)
        """
        if not HAS_MLFLOW:
            return

        try:
            with mlflow.start_run(run_name="best_trial"):
                mlflow.log_params(best_params)
                mlflow.log_metrics(best_metrics)
                mlflow.log_dict(best_params, "best_params.json")

                if best_model is not None:
                    try:
                        mlflow.sklearn.log_model(best_model, "model")
                    except Exception as e:
                        logger.debug(f"Could not log model: {e}")

        except Exception as e:
            logger.warning(f"Failed to log best trial to MLflow: {e}")

    def save_params_locally(
        self,
        best_params: dict[str, Any],
        filepath: Path,
    ) -> None:
        """Save best parameters to local JSON file.

        Args:
            best_params: Best hyperparameters to save
            filepath: Path to save JSON file
        """
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(best_params, f, indent=2)
            logger.info(f"Saved best params to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save params locally: {e}")
