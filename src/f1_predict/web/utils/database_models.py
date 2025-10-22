"""SQLAlchemy ORM models for monitoring system database schema.

Defines all entities for monitoring, alerts, drift detection,
and performance tracking.
"""


from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class Prediction(Base):
    """Model for individual predictions."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    model_version = Column(String(50), nullable=False, index=True)
    prediction_id = Column(String(100), unique=True, nullable=False, index=True)
    predicted_outcome = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    actual_outcome = Column(Integer, nullable=True)  # Recorded later
    actual_outcome_timestamp = Column(DateTime(timezone=True), nullable=True)
    features = Column(JSON, nullable=False)
    extra_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_predictions_timestamp_model", "timestamp", "model_version"),
        Index("idx_predictions_confidence", "confidence"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model_version": self.model_version,
            "prediction_id": self.prediction_id,
            "predicted_outcome": self.predicted_outcome,
            "confidence": self.confidence,
            "actual_outcome": self.actual_outcome,
            "actual_outcome_timestamp": (
                self.actual_outcome_timestamp.isoformat()
                if self.actual_outcome_timestamp
                else None
            ),
            "features": self.features,
            "extra_metadata": self.extra_metadata,
        }


class HealthSnapshot(Base):
    """Model for periodic health snapshots."""

    __tablename__ = "health_snapshots"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    model_version = Column(String(50), nullable=False, index=True)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    expected_calibration_error = Column(Float)
    num_predictions = Column(Integer)
    prediction_accuracy_trend = Column(Float)
    degradation_detected = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_health_timestamp_model", "timestamp", "model_version"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model_version": self.model_version,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "expected_calibration_error": self.expected_calibration_error,
            "num_predictions": self.num_predictions,
            "prediction_accuracy_trend": self.prediction_accuracy_trend,
            "degradation_detected": self.degradation_detected,
        }


class Alert(Base):
    """Model for alerts."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    alert_id = Column(String(100), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    title = Column(String(255))
    message = Column(Text)
    metric_name = Column(String(100))
    metric_value = Column(Float)
    threshold = Column(Float)
    component = Column(String(50))
    model_version = Column(String(50))
    acknowledged = Column(Boolean, default=False, index=True)
    acknowledged_at = Column(DateTime(timezone=True))
    acknowledged_by = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_alerts_timestamp_severity", "timestamp", "severity"),
        Index("idx_alerts_acknowledged", "acknowledged"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "component": self.component,
            "model_version": self.model_version,
            "acknowledged": self.acknowledged,
            "acknowledged_at": (
                self.acknowledged_at.isoformat() if self.acknowledged_at else None
            ),
            "acknowledged_by": self.acknowledged_by,
        }


class AlertRule(Base):
    """Model for alert rules."""

    __tablename__ = "alert_rules"

    id = Column(Integer, primary_key=True)
    rule_id = Column(String(100), unique=True, nullable=False, index=True)
    metric_name = Column(String(100), nullable=False)
    metric_type = Column(String(50))
    threshold = Column(Float)
    comparison = Column(String(5))
    severity = Column(String(20))
    component = Column(String(50))
    enabled = Column(Boolean, default=True, index=True)
    cooldown_minutes = Column(Integer, default=60)
    channels = Column(JSON, default=[])
    conditions = Column(JSON)
    last_triggered = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (UniqueConstraint("rule_id", name="uq_rule_id"),)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "metric_name": self.metric_name,
            "metric_type": self.metric_type,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "severity": self.severity,
            "component": self.component,
            "enabled": self.enabled,
            "cooldown_minutes": self.cooldown_minutes,
            "channels": self.channels,
            "conditions": self.conditions,
            "last_triggered": (
                self.last_triggered.isoformat() if self.last_triggered else None
            ),
        }


class DriftResult(Base):
    """Model for drift detection results."""

    __tablename__ = "drift_results"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    feature_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50))
    drift_detected = Column(Boolean, index=True)
    test_statistic = Column(Float)
    p_value = Column(Float)
    threshold = Column(Float)
    drift_type = Column(String(50))
    baseline_stats = Column(JSON)
    current_stats = Column(JSON)
    severity = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_drift_timestamp_feature", "timestamp", "feature_name"),
        Index("idx_drift_detected", "drift_detected"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "feature_name": self.feature_name,
            "model_version": self.model_version,
            "drift_detected": self.drift_detected,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "drift_type": self.drift_type,
            "baseline_stats": self.baseline_stats,
            "current_stats": self.current_stats,
            "severity": self.severity,
        }


class FeatureImportance(Base):
    """Model for feature importance history."""

    __tablename__ = "feature_importance"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    model_version = Column(String(50), index=True)
    feature_name = Column(String(100), nullable=False, index=True)
    importance_score = Column(Float)
    shap_value = Column(Float)
    percentage = Column(Float)
    rank = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_importance_timestamp_feature", "timestamp", "feature_name"),
        Index("idx_importance_model_feature", "model_version", "feature_name"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model_version": self.model_version,
            "feature_name": self.feature_name,
            "importance_score": self.importance_score,
            "shap_value": self.shap_value,
            "percentage": self.percentage,
            "rank": self.rank,
        }


class DegradationAnalysis(Base):
    """Model for performance degradation analysis."""

    __tablename__ = "degradation_analysis"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    model_version = Column(String(50), index=True)
    metric_name = Column(String(100))
    baseline_value = Column(Float)
    current_value = Column(Float)
    degradation_percent = Column(Float)
    failure_cohort_size = Column(Integer)
    top_features = Column(JSON)
    error_patterns = Column(JSON)
    recommended_actions = Column(JSON, default=[])
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("idx_degradation_timestamp_model", "timestamp", "model_version"),)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model_version": self.model_version,
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "degradation_percent": self.degradation_percent,
            "failure_cohort_size": self.failure_cohort_size,
            "top_features": self.top_features,
            "error_patterns": self.error_patterns,
            "recommended_actions": self.recommended_actions,
        }


class MetricSnapshot(Base):
    """Model for performance metric snapshots."""

    __tablename__ = "metric_snapshots"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    model_version = Column(String(50), nullable=False, index=True)
    metric_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    window_size = Column(Integer)
    threshold = Column(Float)
    status = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_metric_timestamp_name", "timestamp", "metric_name"),
        Index("idx_metric_model_name", "model_version", "metric_name"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model_version": self.model_version,
            "metric_name": self.metric_name,
            "value": self.value,
            "window_size": self.window_size,
            "threshold": self.threshold,
            "status": self.status,
        }
