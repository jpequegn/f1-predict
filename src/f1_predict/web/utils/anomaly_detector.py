"""High-level anomaly detection system with adaptive thresholds.

Manages:
- Multiple anomaly detection methods
- Historical baseline tracking
- Adaptive threshold adjustment
- Anomaly explanations and severity assessment
"""

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog

from f1_predict.models.anomaly_detection import (
    AnomalyDetector,
    AnomalyScore,
    AnomalyType,
    ARIMAForecastDetector,
    HistoricalBaseline,
    IQRDetector,
    IsolationForestDetector,
    NDArray,
    ZScoreDetector,
)

logger = structlog.get_logger(__name__)


class ThresholdStrategy(Enum):
    """Strategies for adaptive threshold adjustment."""

    STATIC = "static"  # Fixed threshold
    ADAPTIVE_LINEAR = "adaptive_linear"  # Linear adjustment based on history
    ADAPTIVE_PERCENTILE = "adaptive_percentile"  # Based on percentiles
    ADAPTIVE_DYNAMIC = "adaptive_dynamic"  # Dynamic based on recent data


@dataclass
class DetectorConfig:
    """Configuration for a specific detector."""

    method: str  # "isolation_forest", "zscore", "iqr"
    enabled: bool = True
    weight: float = 1.0  # Weight in ensemble (0.0-1.0)
    threshold_strategy: str = ThresholdStrategy.ADAPTIVE_PERCENTILE.value
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "enabled": self.enabled,
            "weight": self.weight,
            "threshold_strategy": self.threshold_strategy,
            "parameters": self.parameters,
        }


@dataclass
class AnomalyEvent:
    """Recorded anomaly event."""

    timestamp: float
    anomaly_score: AnomalyScore
    detectors_triggered: list[str]
    ensemble_score: float
    acknowledged: bool = False
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "anomaly_score": self.anomaly_score.to_dict(),
            "detectors_triggered": self.detectors_triggered,
            "ensemble_score": round(self.ensemble_score, 4),
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at,
            "acknowledged_by": self.acknowledged_by,
            "notes": self.notes,
        }


class AnomalyDetectionSystem:
    """Manages multiple anomaly detectors with adaptive thresholds."""

    def __init__(
        self,
        data_dir: Path | str = "data/monitoring",
        window_size: int = 100,
    ):
        """Initialize anomaly detection system.

        Args:
            data_dir: Directory for storing baselines and history
            window_size: Window size for rolling statistics
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size

        self.baselines_file = self.data_dir / "anomaly_baselines.json"
        self.events_file = self.data_dir / "anomaly_events.jsonl"
        self.logger = logger.bind(component="anomaly_detection_system")

        # Initialize detectors
        self.detectors: dict[str, AnomalyDetector] = {}
        self.detector_configs: dict[str, DetectorConfig] = {}
        self.historical_baselines: dict[str, HistoricalBaseline] = {}
        self.anomaly_events: list[AnomalyEvent] = []

        self._load_baselines()
        self._load_events()

    def register_detector(self, config: DetectorConfig) -> None:
        """Register an anomaly detector.

        Args:
            config: Detector configuration
        """
        try:
            if not config.enabled:
                self.logger.info("detector_disabled", method=config.method)
                return

            if config.method == "isolation_forest":
                params = config.parameters or {}
                self.detectors[config.method] = IsolationForestDetector(
                    contamination=params.get("contamination", 0.1),
                    random_state=params.get("random_state", 42),
                )
            elif config.method == "zscore":
                params = config.parameters or {}
                self.detectors[config.method] = ZScoreDetector(
                    threshold_sigma=params.get("threshold_sigma", 3.0)
                )
            elif config.method == "iqr":
                params = config.parameters or {}
                self.detectors[config.method] = IQRDetector(
                    iqr_multiplier=params.get("iqr_multiplier", 1.5)
                )
            elif config.method == "arima":
                params = config.parameters or {}
                order = tuple(params.get("order", [1, 1, 1]))
                self.detectors[config.method] = ARIMAForecastDetector(
                    order=order,
                    deviation_threshold=params.get("deviation_threshold", 2.0),
                )
            else:
                self.logger.warning("unknown_detector_method", method=config.method)
                return

            self.detector_configs[config.method] = config
            self.logger.info("detector_registered", method=config.method)

        except Exception as e:
            self.logger.error("detector_registration_failed", method=config.method, error=str(e))

    def fit_detectors(self, data: pd.DataFrame | NDArray, feature_name: Optional[str] = None) -> None:
        """Fit all registered detectors to baseline data.

        Args:
            data: Historical data for baseline
            feature_name: Name of feature being monitored
        """
        try:
            if isinstance(data, pd.DataFrame):
                if feature_name is None and len(data.columns) > 0:
                    feature_name = str(data.columns[0])
                data_array: NDArray = data.values
            else:
                data_array = np.asarray(data)

            for method, detector in self.detectors.items():
                detector.fit(data_array)
                self.logger.info("detector_fitted", method=method)

            # Calculate baseline statistics
            if isinstance(data, pd.DataFrame):
                values = data.iloc[:, 0].values if len(data.columns) > 0 else data_array[:, 0]
            else:
                values = data_array if data_array.ndim == 1 else data_array[:, 0]

            baseline = HistoricalBaseline(
                feature_name=feature_name or "unknown",
                mean=float(np.mean(values)),
                std=float(np.std(values)),
                min_val=float(np.min(values)),
                max_val=float(np.max(values)),
                percentile_25=float(np.percentile(values, 25)),
                percentile_50=float(np.percentile(values, 50)),
                percentile_75=float(np.percentile(values, 75)),
                percentile_95=float(np.percentile(values, 95)),
                percentile_99=float(np.percentile(values, 99)),
                n_samples=len(values),
                created_at=time.time(),
            )

            self.historical_baselines[feature_name or "unknown"] = baseline
            self._save_baselines()

            self.logger.info(
                "anomaly_detectors_fitted",
                feature=feature_name,
                n_samples=len(values),
            )

        except Exception as e:
            self.logger.error("detector_fitting_failed", error=str(e))
            raise

    def detect(
        self,
        data: pd.DataFrame | NDArray,
        feature_name: Optional[str] = None,
        timestamp: Optional[float] = None,
        ensemble: bool = True,
    ) -> Optional[AnomalyEvent]:
        """Detect anomalies using registered detectors.

        Args:
            data: Data to check for anomalies
            feature_name: Name of feature being checked
            timestamp: Observation timestamp
            ensemble: Whether to use ensemble voting

        Returns:
            AnomalyEvent if anomaly detected, None otherwise
        """
        if not self.detectors:
            self.logger.warning("no_detectors_registered")
            return None

        if timestamp is None:
            timestamp = time.time()

        try:
            detectors_triggered = []
            scores = []
            anomaly_explanation_parts = []

            # Run all detectors
            for method, detector in self.detectors.items():
                config = self.detector_configs.get(method)
                if not config or not config.enabled:
                    continue

                try:
                    score = detector.detect(data, timestamp)

                    if score.is_anomaly:
                        detectors_triggered.append(method)

                    # Weighted score
                    weighted_score = score.score * config.weight
                    scores.append(weighted_score)

                    anomaly_explanation_parts.append(
                        f"{method}: {score.explanation} (weight: {config.weight})"
                    )

                    self.logger.debug(
                        "detector_result",
                        method=method,
                        is_anomaly=score.is_anomaly,
                        score=score.score,
                    )

                except Exception as e:
                    self.logger.error(
                        "detector_execution_failed",
                        method=method,
                        error=str(e),
                    )
                    continue

            if not scores:
                return None

            # Calculate ensemble score
            ensemble_score = np.mean(scores) if scores else 0.0

            # Determine if ensemble detects anomaly
            is_ensemble_anomaly = ensemble_score > 0.5 if ensemble else len(detectors_triggered) > 0

            if not is_ensemble_anomaly and not detectors_triggered:
                return None

            # Create anomaly score with ensemble results
            anomaly_score = AnomalyScore(
                timestamp=timestamp,
                is_anomaly=is_ensemble_anomaly,
                anomaly_type=AnomalyType.MULTIVARIATE.value,
                score=ensemble_score,
                threshold=0.5,
                confidence=min(1.0, ensemble_score),
                features_involved=[feature_name or "multiple"],
                explanation="; ".join(anomaly_explanation_parts),
                severity=self._assess_severity(ensemble_score, detectors_triggered),
            )

            # Create event
            event = AnomalyEvent(
                timestamp=timestamp,
                anomaly_score=anomaly_score,
                detectors_triggered=detectors_triggered,
                ensemble_score=ensemble_score,
            )

            # Store event
            self.anomaly_events.append(event)
            self._save_events()

            self.logger.info(
                "anomaly_detected",
                score=ensemble_score,
                detectors=len(detectors_triggered),
                severity=anomaly_score.severity,
            )

            return event

        except Exception as e:
            self.logger.error("anomaly_detection_failed", error=str(e))
            return None

    def get_baseline(self, feature_name: Optional[str] = None) -> Optional[HistoricalBaseline]:
        """Get baseline statistics for a feature.

        Args:
            feature_name: Feature name

        Returns:
            HistoricalBaseline or None
        """
        if feature_name is None:
            feature_name = list(self.historical_baselines.keys())[0] if self.historical_baselines else None

        if feature_name is None:
            return None
        return self.historical_baselines.get(feature_name)

    def update_baseline(self, data: pd.DataFrame | NDArray, feature_name: Optional[str] = None) -> None:
        """Update baseline statistics from new data.

        Args:
            data: New data to incorporate
            feature_name: Feature name
        """
        try:
            if isinstance(data, pd.DataFrame):
                if feature_name is None:
                    feature_name = str(data.columns[0]) if len(data.columns) > 0 else "unknown"
                values: NDArray = data.iloc[:, 0].values if len(data.columns) > 0 else data.values[:, 0]
            else:
                values = np.asarray(data).flatten()
                if feature_name is None:
                    feature_name = "unknown"

            # Calculate new baseline
            baseline = HistoricalBaseline(
                feature_name=feature_name,
                mean=float(np.mean(values)),
                std=float(np.std(values)),
                min_val=float(np.min(values)),
                max_val=float(np.max(values)),
                percentile_25=float(np.percentile(values, 25)),
                percentile_50=float(np.percentile(values, 50)),
                percentile_75=float(np.percentile(values, 75)),
                percentile_95=float(np.percentile(values, 95)),
                percentile_99=float(np.percentile(values, 99)),
                n_samples=len(values),
                created_at=time.time(),
            )

            self.historical_baselines[feature_name] = baseline
            self._save_baselines()

            self.logger.info("baseline_updated", feature=feature_name)

        except Exception as e:
            self.logger.error("baseline_update_failed", error=str(e))

    def get_anomaly_events(
        self, limit: int = 100, severity: Optional[str] = None
    ) -> list[AnomalyEvent]:
        """Get recent anomaly events.

        Args:
            limit: Maximum events to return
            severity: Filter by severity

        Returns:
            List of anomaly events
        """
        events = self.anomaly_events
        if severity:
            events = [e for e in events if e.anomaly_score.severity == severity]
        return sorted(events, key=lambda x: x.timestamp, reverse=True)[:limit]

    def acknowledge_anomaly(
        self, timestamp: float, acknowledged_by: Optional[str] = None, notes: str = ""
    ) -> bool:
        """Acknowledge an anomaly event.

        Args:
            timestamp: Event timestamp
            acknowledged_by: User who acknowledged
            notes: Additional notes

        Returns:
            True if successful
        """
        for event in self.anomaly_events:
            if abs(event.timestamp - timestamp) < 1.0:  # Match within 1 second
                event.acknowledged = True
                event.acknowledged_at = time.time()
                event.acknowledged_by = acknowledged_by
                event.notes = notes
                self._save_events()
                self.logger.info("anomaly_acknowledged", timestamp=timestamp, by=acknowledged_by)
                return True
        return False

    def get_statistics(self) -> dict[str, Any]:
        """Get anomaly detection statistics.

        Returns:
            Dictionary with statistics
        """
        total_anomalies = len(self.anomaly_events)
        unacknowledged = len([e for e in self.anomaly_events if not e.acknowledged])

        by_severity = {
            "critical": len([e for e in self.anomaly_events if e.anomaly_score.severity == "critical"]),
            "warning": len([e for e in self.anomaly_events if e.anomaly_score.severity == "warning"]),
            "info": len([e for e in self.anomaly_events if e.anomaly_score.severity == "info"]),
        }

        by_detector = {}
        for event in self.anomaly_events:
            for detector in event.detectors_triggered:
                if detector not in by_detector:
                    by_detector[detector] = 0
                by_detector[detector] += 1

        return {
            "total_anomalies": total_anomalies,
            "unacknowledged": unacknowledged,
            "by_severity": by_severity,
            "by_detector": by_detector,
            "registered_detectors": list(self.detector_configs.keys()),
            "baselines": list(self.historical_baselines.keys()),
        }

    def _assess_severity(self, score: float, detectors_triggered: list[str]) -> str:
        """Assess anomaly severity based on score and detectors.

        Args:
            score: Ensemble anomaly score
            detectors_triggered: List of detectors that triggered

        Returns:
            Severity level: "info", "warning", "critical"
        """
        if score > 0.8 or len(detectors_triggered) >= 2:
            return "critical"
        if score > 0.6 or len(detectors_triggered) == 1:
            return "warning"
        return "info"

    def _save_baselines(self) -> None:
        """Save baselines to file."""
        try:
            baselines_dict = {
                name: baseline.to_dict()
                for name, baseline in self.historical_baselines.items()
            }
            with open(self.baselines_file, "w") as f:
                json.dump(baselines_dict, f, indent=2)
        except Exception as e:
            self.logger.error("baseline_save_failed", error=str(e))

    def _load_baselines(self) -> None:
        """Load baselines from file."""
        try:
            if self.baselines_file.exists():
                with open(self.baselines_file) as f:
                    baselines_dict = json.load(f)
                    for name, baseline_data in baselines_dict.items():
                        self.historical_baselines[name] = HistoricalBaseline(**baseline_data)
                self.logger.info("baselines_loaded", count=len(self.historical_baselines))
        except Exception as e:
            self.logger.error("baseline_load_failed", error=str(e))

    def _save_events(self) -> None:
        """Save anomaly events to file."""
        try:
            with open(self.events_file, "a") as f:
                # Append only the last event
                if self.anomaly_events:
                    last_event = self.anomaly_events[-1]
                    f.write(json.dumps(last_event.to_dict()) + "\n")
        except Exception as e:
            self.logger.error("event_save_failed", error=str(e))

    def _load_events(self) -> None:
        """Load anomaly events from file."""
        try:
            if self.events_file.exists():
                with open(self.events_file) as f:
                    for line in f:
                        if line.strip():
                            event_data = json.loads(line)
                            # Reconstruct AnomalyEvent from dict
                            score_data = event_data.pop("anomaly_score")
                            anomaly_score = AnomalyScore(**score_data)
                            event = AnomalyEvent(
                                anomaly_score=anomaly_score,
                                **event_data,
                            )
                            self.anomaly_events.append(event)
                self.logger.info("events_loaded", count=len(self.anomaly_events))
        except Exception as e:
            self.logger.error("event_load_failed", error=str(e))
