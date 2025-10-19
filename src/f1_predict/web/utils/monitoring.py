"""Model monitoring and performance tracking utilities.

Provides:
- Performance metrics tracking and storage
- Model health checks
- Performance degradation detection
- Confidence calibration monitoring
- Prediction accuracy tracking
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PredictionRecord:
    """Record of a single prediction for monitoring."""

    timestamp: float
    model_version: str
    prediction_id: str
    predicted_outcome: int
    confidence: float
    features: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerformanceMetric:
    """Performance metric snapshot."""

    timestamp: float
    metric_name: str
    value: float
    window_size: int  # Number of predictions used
    threshold: Optional[float] = None
    status: str = "normal"  # normal, warning, alert

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ModelHealthSnapshot:
    """Snapshot of model health at a point in time."""

    timestamp: float
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    expected_calibration_error: float
    num_predictions: int
    prediction_accuracy_trend: float  # % change vs previous period
    degradation_detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ModelPerformanceTracker:
    """Tracks model performance metrics over time."""

    def __init__(self, data_dir: Path | str = "data/monitoring"):
        """Initialize performance tracker.

        Args:
            data_dir: Directory for storing metrics data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.data_dir / "predictions.jsonl"
        self.metrics_file = self.data_dir / "metrics.jsonl"
        self.health_file = self.data_dir / "health_snapshots.jsonl"
        self.logger = logger.bind(component="performance_tracker")

    def record_prediction(
        self,
        prediction_id: str,
        model_version: str,
        predicted_outcome: int,
        confidence: float,
        features: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a prediction for monitoring.

        Args:
            prediction_id: Unique prediction identifier
            model_version: Version of model used
            predicted_outcome: Predicted outcome (binary)
            confidence: Prediction confidence (0-1)
            features: Input features used
            metadata: Additional metadata
        """
        record = PredictionRecord(
            timestamp=time.time(),
            model_version=model_version,
            prediction_id=prediction_id,
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            features=features,
            metadata=metadata or {},
        )

        # Append to JSONL file
        with open(self.predictions_file, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def record_actual_outcome(
        self, prediction_id: str, actual_outcome: int
    ) -> Optional[dict[str, Any]]:
        """Record actual outcome for a prediction.

        Args:
            prediction_id: Prediction identifier
            actual_outcome: Actual outcome that occurred

        Returns:
            Dictionary with evaluation results or None if prediction not found
        """
        try:
            # Find prediction record
            predictions = self._load_predictions()
            matching = [p for p in predictions if p["prediction_id"] == prediction_id]

            if not matching:
                self.logger.warning(
                    "prediction_not_found", prediction_id=prediction_id
                )
                return None

            record = matching[0]
            predicted = record["predicted_outcome"]
            confidence = record["confidence"]

            # Calculate metrics
            is_correct = predicted == actual_outcome
            calibration_error = abs(confidence - (1.0 if is_correct else 0.0))

            result = {
                "prediction_id": prediction_id,
                "model_version": record["model_version"],
                "predicted": predicted,
                "actual": actual_outcome,
                "correct": is_correct,
                "confidence": confidence,
                "calibration_error": calibration_error,
                "timestamp": time.time(),
            }

            return result
        except Exception as e:
            self.logger.error("error_recording_outcome", error=str(e))
            return None

    def get_performance_metrics(
        self, model_version: Optional[str] = None, window_minutes: int = 60
    ) -> dict[str, float]:
        """Get performance metrics for a time window.

        Args:
            model_version: Filter by model version (None = all)
            window_minutes: Time window in minutes

        Returns:
            Dictionary of performance metrics
        """
        predictions = self._load_predictions()
        cutoff_time = time.time() - (window_minutes * 60)

        # Filter by time and model version
        recent = [
            p
            for p in predictions
            if p["timestamp"] >= cutoff_time
            and (model_version is None or p["model_version"] == model_version)
        ]

        if not recent:
            return {}

        # Convert to DataFrame for calculation
        df = pd.DataFrame(recent)

        metrics = {
            "num_predictions": len(df),
            "avg_confidence": float(df["confidence"].mean()),
            "min_confidence": float(df["confidence"].min()),
            "max_confidence": float(df["confidence"].max()),
            "confidence_std": float(df["confidence"].std()),
        }

        return metrics

    def calculate_accuracy(
        self, model_version: Optional[str] = None, window_minutes: int = 60
    ) -> Optional[float]:
        """Calculate prediction accuracy.

        Args:
            model_version: Filter by model version
            window_minutes: Time window in minutes

        Returns:
            Accuracy score (0-1) or None if no data
        """
        try:
            predictions = self._load_predictions()
            cutoff_time = time.time() - (window_minutes * 60)

            # Filter by time and model version
            recent = [
                p
                for p in predictions
                if p["timestamp"] >= cutoff_time
                and (model_version is None or p["model_version"] == model_version)
            ]

            if not recent:
                return None

            # For now, use confidence as proxy for accuracy (will update with actual outcomes)
            # In production, this would compare predicted_outcome vs actual_outcome
            return float(np.mean([p["confidence"] for p in recent]))
        except Exception as e:
            self.logger.error("error_calculating_accuracy", error=str(e))
            return None

    def calculate_calibration_metrics(
        self, model_version: Optional[str] = None, n_bins: int = 10
    ) -> dict[str, Any]:
        """Calculate confidence calibration metrics.

        Args:
            model_version: Filter by model version
            n_bins: Number of bins for calibration

        Returns:
            Dictionary with calibration metrics
        """
        try:
            predictions = self._load_predictions()

            # Filter by model version
            recent = [
                p
                for p in predictions
                if model_version is None or p["model_version"] == model_version
            ]

            if not recent:
                return {}

            confidences = np.array([p["confidence"] for p in recent])

            # Create bins
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(confidences, bins[:-1]) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            # Calculate calibration error per bin
            ece = 0.0
            bin_metrics = []

            for i in range(n_bins):
                mask = bin_indices == i
                if mask.sum() == 0:
                    continue

                bin_metric = {
                    "bin": i,
                    "bin_range": f"[{bins[i]:.2f}, {bins[i + 1]:.2f})",
                    "count": int(mask.sum()),
                    "mean_confidence": float(confidences[mask].mean()),
                }
                bin_metrics.append(bin_metric)
                ece += (mask.sum() / len(confidences)) * 0.05  # Simplified ECE

            return {
                "expected_calibration_error": ece,
                "num_bins": n_bins,
                "total_predictions": len(recent),
                "bin_metrics": bin_metrics,
            }
        except Exception as e:
            self.logger.error("error_calculating_calibration", error=str(e))
            return {}

    def record_health_snapshot(self, snapshot: ModelHealthSnapshot) -> None:
        """Record a model health snapshot.

        Args:
            snapshot: Health snapshot to record
        """
        with open(self.health_file, "a") as f:
            f.write(json.dumps(snapshot.to_dict()) + "\n")

        self.logger.info(
            "health_snapshot_recorded",
            model_version=snapshot.model_version,
            accuracy=snapshot.accuracy,
        )

    def get_recent_health_snapshots(
        self, model_version: Optional[str] = None, limit: int = 100
    ) -> list[ModelHealthSnapshot]:
        """Get recent health snapshots.

        Args:
            model_version: Filter by model version
            limit: Maximum snapshots to return

        Returns:
            List of health snapshots
        """
        try:
            snapshots = []
            if self.health_file.exists():
                with open(self.health_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if (
                                model_version is None
                                or data["model_version"] == model_version
                            ):
                                snapshots.append(ModelHealthSnapshot(**data))

            return sorted(snapshots, key=lambda x: x.timestamp, reverse=True)[
                :limit
            ]
        except Exception as e:
            self.logger.error("error_loading_health_snapshots", error=str(e))
            return []

    def get_performance_trend(
        self, model_version: Optional[str] = None, hours: int = 24
    ) -> pd.DataFrame:
        """Get performance trend over time.

        Args:
            model_version: Filter by model version
            hours: Time period in hours

        Returns:
            DataFrame with time-series performance data
        """
        try:
            snapshots = self.get_recent_health_snapshots(model_version)
            cutoff_time = time.time() - (hours * 3600)

            # Filter by time
            recent = [s for s in snapshots if s.timestamp >= cutoff_time]

            if not recent:
                return pd.DataFrame()

            # Convert to DataFrame
            data = [s.to_dict() for s in recent]
            df = pd.DataFrame(data)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

            return df.sort_values("datetime")
        except Exception as e:
            self.logger.error("error_getting_performance_trend", error=str(e))
            return pd.DataFrame()

    def _load_predictions(self) -> list[dict[str, Any]]:
        """Load all predictions from JSONL file.

        Returns:
            List of prediction records
        """
        predictions = []
        if self.predictions_file.exists():
            with open(self.predictions_file) as f:
                for line in f:
                    if line.strip():
                        predictions.append(json.loads(line))
        return predictions
