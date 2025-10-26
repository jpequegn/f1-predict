"""Anomaly detection hooks for data pipeline integration."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import structlog

from f1_predict.data.univariate_detector import UnivariateDetector

logger = structlog.get_logger(__name__)


@dataclass
class AnomalyMetadata:
    """Metadata for anomaly detection results."""

    anomaly_flag: bool = False
    anomaly_score: float = 0.0
    anomaly_method: str = ""
    anomaly_confidence: float = 0.0
    features_involved: list[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anomaly_flag": self.anomaly_flag,
            "anomaly_score": self.anomaly_score,
            "anomaly_method": self.anomaly_method,
            "anomaly_confidence": self.anomaly_confidence,
            "features_involved": self.features_involved,
            "explanation": self.explanation,
        }


class AnomalyDetectionHooks:
    """Pluggable hooks for anomaly detection in data pipeline."""

    def __init__(self):
        """Initialize anomaly detection hooks."""
        self.logger = logger.bind(component="anomaly_hooks")
        # Initialize univariate detector for fast checks during collection
        self.univariate_detector = UnivariateDetector()
        # Multivariate analyzer will be initialized in later tasks
        self.multivariate_analyzer = None
        self.registry = None

    def on_data_collected(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Hook: Run fast checks on collected data.

        Args:
            data: List of collected records

        Returns:
            Data with anomaly flags added
        """
        try:
            if not data:
                return data

            # Convert to DataFrame for univariate detection
            df = pd.DataFrame(data)

            # Run univariate anomaly detection
            df_with_anomalies = self.univariate_detector.detect(df)

            # Convert back to list of dicts
            result = df_with_anomalies.to_dict("records")

            # Add structured anomaly metadata to each record
            for record in result:
                # Create AnomalyMetadata from detection results
                record["_anomaly"] = AnomalyMetadata(
                    anomaly_flag=record.get("anomaly_flag", False),
                    anomaly_score=record.get("anomaly_score", 0.0),
                    anomaly_method=record.get("anomaly_method", ""),
                    anomaly_confidence=record.get("anomaly_confidence", 0.0),
                ).to_dict()

            self.logger.debug(
                "on_data_collected_complete",
                data_count=len(result),
                anomalies_detected=sum(
                    1 for r in result if r.get("anomaly_flag", False)
                ),
            )

            return result
        except Exception as e:
            self.logger.error(
                "error_in_data_collection_hook",
                error=str(e),
                data_count=len(data),
                exc_info=True,
            )
            return data

    def on_data_stored(self, data: list[dict[str, Any]], season: int) -> dict[str, Any]:
        """Hook: Run sophisticated analysis post-storage.

        Args:
            data: Stored data records
            season: F1 season

        Returns:
            Anomaly report
        """
        try:
            return {
                "anomalies": [],
                "summary": {"total": 0, "critical": 0},
            }
        except Exception as e:
            self.logger.error(
                "error_in_data_storage_hook",
                error=str(e),
                data_count=len(data),
                season=season,
                exc_info=True,
            )
            return {"anomalies": [], "summary": {}}
