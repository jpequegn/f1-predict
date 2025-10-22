"""Database-backed monitoring tracker implementation.

Replaces file-based JSON storage with SQLAlchemy ORM for improved performance
and reliability. Maintains backward compatibility with existing interfaces.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog

from f1_predict.web.utils.database import DatabaseManager
from f1_predict.web.utils.database_repositories import (
    HealthSnapshotRepository,
    PredictionRepository,
)

logger = structlog.get_logger(__name__)


class ModelPerformanceTrackerDB:
    """Database-backed model performance tracker.

    Drop-in replacement for file-based ModelPerformanceTracker using
    SQLAlchemy ORM and database repositories.
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        """Initialize tracker.

        Args:
            data_dir: Kept for backward compatibility but not used.
                     Database configuration comes from environment variables.
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data/monitoring")
        self.logger = logger.bind(component="performance_tracker_db")

        if not DatabaseManager.is_enabled():
            raise RuntimeError(
                "Database backend not enabled. Set MONITORING_DB_ENABLED=true"
            )

    def record_prediction(
        self,
        prediction_id: str,
        model_version: str,
        predicted_outcome: int,
        confidence: float,
        features: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Record a prediction to database.

        Args:
            prediction_id: Unique prediction identifier
            model_version: Model version string
            predicted_outcome: Predicted class (0 or 1)
            confidence: Prediction confidence (0.0 to 1.0)
            features: Input features as dictionary
            metadata: Optional metadata

        Returns:
            Recorded prediction data
        """
        try:
            with DatabaseManager.session_scope() as session:
                repo = PredictionRepository(session)
                pred = repo.create(
                    timestamp=datetime.now(timezone.utc),
                    model_version=model_version,
                    prediction_id=prediction_id,
                    predicted_outcome=predicted_outcome,
                    confidence=confidence,
                    features=features,
                    extra_metadata=metadata,
                )

                result = pred.to_dict()
                self.logger.info(
                    "prediction_recorded",
                    prediction_id=prediction_id,
                    model_version=model_version,
                )
                return result
        except Exception as e:
            self.logger.error("prediction_record_failed", error=str(e))
            raise

    def record_actual_outcome(
        self, prediction_id: str, actual_outcome: int
    ) -> Optional[dict[str, Any]]:
        """Record actual outcome for a prediction.

        Args:
            prediction_id: Unique prediction identifier
            actual_outcome: Actual outcome (0 or 1)

        Returns:
            Updated prediction data or None if not found
        """
        try:
            with DatabaseManager.session_scope() as session:
                repo = PredictionRepository(session)
                pred = repo.get_by_prediction_id(prediction_id)

                if not pred:
                    self.logger.warning(
                        "prediction_not_found", prediction_id=prediction_id
                    )
                    return None

                # Update actual outcome
                pred.actual_outcome = actual_outcome
                pred.actual_outcome_timestamp = datetime.now(timezone.utc)
                session.flush()

                # Calculate metrics
                is_correct = pred.predicted_outcome == actual_outcome
                calibration_error = abs(
                    pred.confidence - (1.0 if is_correct else 0.0)
                )

                result = {
                    "prediction_id": prediction_id,
                    "model_version": pred.model_version,
                    "predicted": pred.predicted_outcome,
                    "actual": actual_outcome,
                    "correct": is_correct,
                    "confidence": pred.confidence,
                    "calibration_error": calibration_error,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                self.logger.info(
                    "actual_outcome_recorded",
                    prediction_id=prediction_id,
                    is_correct=is_correct,
                )
                return result
        except Exception as e:
            self.logger.error("outcome_record_failed", error=str(e))
            raise

    def record_health_snapshot(
        self,
        model_version: str,
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        roc_auc: Optional[float] = None,
        ece: float = 0.0,
        num_predictions: int = 0,
        degradation_detected: bool = False,
    ) -> dict[str, Any]:
        """Record a model health snapshot.

        Args:
            model_version: Model version string
            accuracy: Accuracy metric (0.0 to 1.0)
            precision: Precision metric (0.0 to 1.0)
            recall: Recall metric (0.0 to 1.0)
            f1_score: F1 score (0.0 to 1.0)
            roc_auc: ROC AUC metric (optional)
            ece: Expected calibration error
            num_predictions: Number of predictions in window
            degradation_detected: Whether degradation was detected

        Returns:
            Recorded snapshot data
        """
        try:
            with DatabaseManager.session_scope() as session:
                repo = HealthSnapshotRepository(session)
                snapshot = repo.create(
                    timestamp=datetime.now(timezone.utc),
                    model_version=model_version,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    roc_auc=roc_auc,
                    expected_calibration_error=ece,
                    num_predictions=num_predictions,
                    degradation_detected=degradation_detected,
                )

                result = snapshot.to_dict()
                self.logger.info(
                    "health_snapshot_recorded",
                    model_version=model_version,
                    accuracy=accuracy,
                    degradation=degradation_detected,
                )
                return result
        except Exception as e:
            self.logger.error("health_snapshot_failed", error=str(e))
            raise

    def get_performance_metrics(
        self, model_version: Optional[str] = None, window_minutes: int = 60
    ) -> dict[str, Any]:
        """Get performance metrics for time window.

        Args:
            model_version: Optional model version filter
            window_minutes: Time window in minutes

        Returns:
            Dictionary with performance metrics
        """
        try:
            hours = window_minutes / 60

            with DatabaseManager.session_scope() as session:
                repo = PredictionRepository(session)
                predictions = repo.get_recent(
                    model_version=model_version or "v1.0",
                    hours=int(hours) + 1,
                    limit=10000,
                )

                if not predictions:
                    return {}

                # Convert to dicts for DataFrame
                data = [p.to_dict() for p in predictions]
                df = pd.DataFrame(data)

                metrics = {
                    "num_predictions": len(df),
                    "avg_confidence": float(df["confidence"].mean()),
                    "min_confidence": float(df["confidence"].min()),
                    "max_confidence": float(df["confidence"].max()),
                    "confidence_std": float(df["confidence"].std()),
                    "window_minutes": window_minutes,
                }

                self.logger.info(
                    "metrics_calculated",
                    model_version=model_version,
                    num_predictions=len(df),
                )
                return metrics
        except Exception as e:
            self.logger.error("metrics_calculation_failed", error=str(e))
            return {}

    def calculate_accuracy(
        self, model_version: Optional[str] = None, window_minutes: int = 60
    ) -> Optional[float]:
        """Calculate TRUE accuracy from predictions with actual outcomes.

        CRITICAL FIX: Previous implementation used confidence as accuracy proxy.
        This correctly compares predicted vs actual outcomes.

        Args:
            model_version: Optional model version filter
            window_minutes: Time window in minutes

        Returns:
            Accuracy as float (0.0 to 1.0) or None if no data
        """
        try:
            hours = window_minutes / 60

            with DatabaseManager.session_scope() as session:
                repo = PredictionRepository(session)
                predictions = repo.get_recent(
                    model_version=model_version or "v1.0",
                    hours=int(hours) + 1,
                    limit=10000,
                )

                # Filter to only predictions with actual outcomes
                with_outcomes = [
                    p for p in predictions if p.actual_outcome is not None
                ]

                if not with_outcomes:
                    self.logger.warning(
                        "no_predictions_with_outcomes",
                        model_version=model_version,
                    )
                    return None

                # Calculate TRUE accuracy
                correct = sum(
                    1
                    for p in with_outcomes
                    if p.predicted_outcome == p.actual_outcome
                )

                accuracy = correct / len(with_outcomes)

                self.logger.info(
                    "accuracy_calculated",
                    model_version=model_version,
                    accuracy=accuracy,
                    sample_size=len(with_outcomes),
                )
                return accuracy
        except Exception as e:
            self.logger.error("accuracy_calculation_failed", error=str(e))
            return None

    def calculate_calibration_metrics(
        self, model_version: Optional[str] = None, window_minutes: int = 60
    ) -> dict[str, float]:
        """Calculate calibration metrics (ECE, MCE).

        Args:
            model_version: Optional model version filter
            window_minutes: Time window in minutes

        Returns:
            Dictionary with calibration metrics
        """
        try:
            hours = window_minutes / 60

            with DatabaseManager.session_scope() as session:
                repo = PredictionRepository(session)
                predictions = repo.get_recent(
                    model_version=model_version or "v1.0",
                    hours=int(hours) + 1,
                    limit=10000,
                )

                # Filter to predictions with outcomes
                with_outcomes = [
                    p for p in predictions if p.actual_outcome is not None
                ]

                if not with_outcomes:
                    return {"ece": 0.0, "mce": 0.0}

                # Calculate Expected Calibration Error (ECE)
                confidences = np.array([p.confidence for p in with_outcomes])
                accuracies = np.array(
                    [
                        float(p.predicted_outcome == p.actual_outcome)
                        for p in with_outcomes
                    ]
                )

                # Bin-based ECE
                n_bins = 10
                ece = 0.0
                mce = 0.0

                for i in range(n_bins):
                    bin_lower = i / n_bins
                    bin_upper = (i + 1) / n_bins
                    in_bin = (confidences >= bin_lower) & (
                        confidences < bin_upper
                    )

                    if in_bin.sum() > 0:
                        bin_acc = accuracies[in_bin].mean()
                        bin_conf = confidences[in_bin].mean()
                        bin_ece = abs(bin_acc - bin_conf) * in_bin.sum() / len(
                            confidences
                        )
                        bin_mce = abs(bin_acc - bin_conf)

                        ece += bin_ece
                        mce = max(mce, bin_mce)

                result = {"ece": float(ece), "mce": float(mce)}

                self.logger.info(
                    "calibration_calculated",
                    model_version=model_version,
                    ece=ece,
                )
                return result
        except Exception as e:
            self.logger.error("calibration_calculation_failed", error=str(e))
            return {"ece": 0.0, "mce": 0.0}

    def get_health_snapshots(
        self,
        model_version: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get recent health snapshots.

        Args:
            model_version: Optional model version filter
            limit: Maximum results to return

        Returns:
            List of health snapshot dictionaries
        """
        try:
            with DatabaseManager.session_scope() as session:
                repo = HealthSnapshotRepository(session)
                snapshots = repo.get_latest_by_model(
                    model_version=model_version or "v1.0", limit=limit
                )

                result = [s.to_dict() for s in snapshots]

                self.logger.info(
                    "health_snapshots_retrieved",
                    model_version=model_version,
                    count=len(result),
                )
                return result
        except Exception as e:
            self.logger.error("health_snapshots_retrieval_failed", error=str(e))
            return []

    def get_recent_predictions(
        self,
        model_version: Optional[str] = None,
        hours: int = 24,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get recent predictions.

        Args:
            model_version: Optional model version filter
            hours: Number of hours to look back
            limit: Maximum results

        Returns:
            List of prediction dictionaries
        """
        try:
            with DatabaseManager.session_scope() as session:
                repo = PredictionRepository(session)
                predictions = repo.get_recent(
                    model_version=model_version or "v1.0",
                    hours=hours,
                    limit=limit,
                )

                result = [p.to_dict() for p in predictions]

                self.logger.info(
                    "recent_predictions_retrieved",
                    model_version=model_version,
                    count=len(result),
                    hours=hours,
                )
                return result
        except Exception as e:
            self.logger.error("recent_predictions_retrieval_failed", error=str(e))
            return []
