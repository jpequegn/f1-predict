"""Repository pattern implementation for database access layer.

Provides clean abstraction for CRUD operations and queries on monitoring entities.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session
import structlog

from f1_predict.web.utils.database_models import (
    Alert,
    AlertRule,
    DegradationAnalysis,
    DriftResult,
    FeatureImportance,
    HealthSnapshot,
    MetricSnapshot,
    Prediction,
)

logger = structlog.get_logger(__name__)


class BaseRepository:
    """Base repository with common CRUD operations."""

    def __init__(self, session: Session, model_class: type) -> None:
        """Initialize repository.

        Args:
            session: SQLAlchemy Session
            model_class: ORM model class
        """
        self.session = session
        self.model_class = model_class

    def create(self, **kwargs: Any) -> Any:
        """Create and persist a new entity.

        Args:
            **kwargs: Entity attributes

        Returns:
            Created entity instance
        """
        instance = self.model_class(**kwargs)
        self.session.add(instance)
        self.session.flush()
        logger.debug("entity_created", model=self.model_class.__name__, id=instance.id)
        return instance

    def update(self, entity_id: int, **kwargs: Any) -> Any:
        """Update an existing entity.

        Args:
            entity_id: Entity primary key
            **kwargs: Attributes to update

        Returns:
            Updated entity instance
        """
        instance = self.session.query(self.model_class).filter_by(id=entity_id).first()
        if instance:
            for key, value in kwargs.items():
                setattr(instance, key, value)
            self.session.flush()
            logger.debug(
                "entity_updated", model=self.model_class.__name__, id=entity_id
            )
        return instance

    def delete(self, entity_id: int) -> bool:
        """Delete an entity.

        Args:
            entity_id: Entity primary key

        Returns:
            True if deleted, False if not found
        """
        instance = self.session.query(self.model_class).filter_by(id=entity_id).first()
        if instance:
            self.session.delete(instance)
            self.session.flush()
            logger.debug("entity_deleted", model=self.model_class.__name__, id=entity_id)
            return True
        return False

    def get_by_id(self, entity_id: int) -> Any | None:
        """Get entity by ID.

        Args:
            entity_id: Entity primary key

        Returns:
            Entity instance or None
        """
        return self.session.query(self.model_class).filter_by(id=entity_id).first()

    def get_all(self, limit: int | None = None) -> list[Any]:
        """Get all entities.

        Args:
            limit: Maximum number of results

        Returns:
            List of entity instances
        """
        query = self.session.query(self.model_class)
        if limit:
            query = query.limit(limit)
        return query.all()


class PredictionRepository(BaseRepository):
    """Repository for prediction records."""

    def __init__(self, session: Session) -> None:
        """Initialize prediction repository."""
        super().__init__(session, Prediction)

    def get_by_prediction_id(self, prediction_id: str) -> Prediction | None:
        """Get prediction by prediction_id.

        Args:
            prediction_id: Unique prediction identifier

        Returns:
            Prediction instance or None
        """
        return (
            self.session.query(Prediction)
            .filter(Prediction.prediction_id == prediction_id)
            .first()
        )

    def get_recent(
        self,
        model_version: str,
        hours: int = 24,
        limit: int = 1000,
    ) -> list[Prediction]:
        """Get recent predictions for a model.

        Args:
            model_version: Model version string
            hours: Number of hours to look back
            limit: Maximum results

        Returns:
            List of Prediction instances
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return (
            self.session.query(Prediction)
            .filter(
                and_(
                    Prediction.model_version == model_version,
                    Prediction.timestamp >= cutoff_time,
                )
            )
            .order_by(desc(Prediction.timestamp))
            .limit(limit)
            .all()
        )

    def get_time_range(
        self,
        model_version: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Prediction]:
        """Get predictions within time range.

        Args:
            model_version: Model version string
            start_time: Start time (inclusive)
            end_time: End time (inclusive)

        Returns:
            List of Prediction instances
        """
        return (
            self.session.query(Prediction)
            .filter(
                and_(
                    Prediction.model_version == model_version,
                    Prediction.timestamp >= start_time,
                    Prediction.timestamp <= end_time,
                )
            )
            .order_by(desc(Prediction.timestamp))
            .all()
        )

    def get_by_confidence_range(
        self,
        model_version: str,
        min_confidence: float,
        max_confidence: float,
        limit: int = 100,
    ) -> list[Prediction]:
        """Get predictions by confidence range.

        Args:
            model_version: Model version string
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            limit: Maximum results

        Returns:
            List of Prediction instances
        """
        return (
            self.session.query(Prediction)
            .filter(
                and_(
                    Prediction.model_version == model_version,
                    Prediction.confidence >= min_confidence,
                    Prediction.confidence <= max_confidence,
                )
            )
            .order_by(desc(Prediction.confidence))
            .limit(limit)
            .all()
        )

    def batch_create(self, predictions: list[dict[str, Any]]) -> int:
        """Batch insert multiple predictions.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Number of created records
        """
        instances = [Prediction(**pred) for pred in predictions]
        self.session.add_all(instances)
        self.session.flush()
        logger.info("batch_predictions_created", count=len(instances))
        return len(instances)


class HealthSnapshotRepository(BaseRepository):
    """Repository for health snapshots."""

    def __init__(self, session: Session) -> None:
        """Initialize health snapshot repository."""
        super().__init__(session, HealthSnapshot)

    def get_latest_by_model(
        self, model_version: str, limit: int = 100
    ) -> list[HealthSnapshot]:
        """Get latest health snapshots for model.

        Args:
            model_version: Model version string
            limit: Maximum results

        Returns:
            List of HealthSnapshot instances sorted by timestamp descending
        """
        return (
            self.session.query(HealthSnapshot)
            .filter(HealthSnapshot.model_version == model_version)
            .order_by(desc(HealthSnapshot.timestamp))
            .limit(limit)
            .all()
        )

    def get_degradation_flags(
        self, model_version: str, hours: int = 24
    ) -> list[HealthSnapshot]:
        """Get snapshots with degradation flags.

        Args:
            model_version: Model version string
            hours: Number of hours to look back

        Returns:
            List of HealthSnapshot instances with degradation detected
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return (
            self.session.query(HealthSnapshot)
            .filter(
                and_(
                    HealthSnapshot.model_version == model_version,
                    HealthSnapshot.timestamp >= cutoff_time,
                    HealthSnapshot.degradation_detected.is_(True),
                )
            )
            .order_by(desc(HealthSnapshot.timestamp))
            .all()
        )


class AlertRepository(BaseRepository):
    """Repository for alert records."""

    def __init__(self, session: Session) -> None:
        """Initialize alert repository."""
        super().__init__(session, Alert)

    def get_by_alert_id(self, alert_id: str) -> Alert | None:
        """Get alert by alert_id.

        Args:
            alert_id: Unique alert identifier

        Returns:
            Alert instance or None
        """
        return self.session.query(Alert).filter(Alert.alert_id == alert_id).first()

    def get_recent(
        self,
        limit: int = 100,
        severity: str | None = None,
        acknowledged: bool | None = None,
    ) -> list[Alert]:
        """Get recent alerts with optional filtering.

        Args:
            limit: Maximum results
            severity: Filter by severity level
            acknowledged: Filter by acknowledgement status

        Returns:
            List of Alert instances sorted by timestamp descending
        """
        query = self.session.query(Alert)

        if severity:
            query = query.filter(Alert.severity == severity)
        if acknowledged is not None:
            query = query.filter(Alert.acknowledged.is_(acknowledged))

        return query.order_by(desc(Alert.timestamp)).limit(limit).all()

    def get_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        severity: str | None = None,
    ) -> list[Alert]:
        """Get alerts within time range.

        Args:
            start_time: Start time (inclusive)
            end_time: End time (inclusive)
            severity: Optional severity filter

        Returns:
            List of Alert instances
        """
        query = self.session.query(Alert).filter(
            and_(
                Alert.timestamp >= start_time,
                Alert.timestamp <= end_time,
            )
        )

        if severity:
            query = query.filter(Alert.severity == severity)

        return query.order_by(desc(Alert.timestamp)).all()

    def get_unacknowledged(self, limit: int = 50) -> list[Alert]:
        """Get unacknowledged alerts.

        Args:
            limit: Maximum results

        Returns:
            List of unacknowledged Alert instances
        """
        return (
            self.session.query(Alert)
            .filter(Alert.acknowledged.is_(False))
            .order_by(desc(Alert.timestamp))
            .limit(limit)
            .all()
        )

    def acknowledge(
        self, alert_id: str, acknowledged_by: str | None = None
    ) -> Alert | None:
        """Acknowledge an alert.

        Args:
            alert_id: Unique alert identifier
            acknowledged_by: User who acknowledged

        Returns:
            Updated Alert instance or None if not found
        """
        alert = self.get_by_alert_id(alert_id)
        if alert:
            alert.acknowledged = True
            alert.acknowledged_at = datetime.now(timezone.utc)
            if acknowledged_by:
                alert.acknowledged_by = acknowledged_by
            self.session.flush()
            logger.info("alert_acknowledged", alert_id=alert_id, user=acknowledged_by)
            return alert
        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get alert statistics.

        Returns:
            Dictionary with alert counts by severity and status
        """
        total = self.session.query(Alert).count()
        by_severity = {}
        for severity in ["critical", "warning", "info"]:
            count = (
                self.session.query(Alert)
                .filter(Alert.severity == severity)
                .count()
            )
            by_severity[severity] = count

        acknowledged = (
            self.session.query(Alert).filter(Alert.acknowledged.is_(True)).count()
        )
        unacknowledged = total - acknowledged

        return {
            "total": total,
            "by_severity": by_severity,
            "acknowledged": acknowledged,
            "unacknowledged": unacknowledged,
        }


class AlertRuleRepository(BaseRepository):
    """Repository for alert rules."""

    def __init__(self, session: Session) -> None:
        """Initialize alert rule repository."""
        super().__init__(session, AlertRule)

    def get_by_rule_id(self, rule_id: str) -> AlertRule | None:
        """Get rule by rule_id.

        Args:
            rule_id: Unique rule identifier

        Returns:
            AlertRule instance or None
        """
        return (
            self.session.query(AlertRule).filter(AlertRule.rule_id == rule_id).first()
        )

    def get_enabled(self) -> list[AlertRule]:
        """Get all enabled rules.

        Returns:
            List of enabled AlertRule instances
        """
        return self.session.query(AlertRule).filter(AlertRule.enabled.is_(True)).all()

    def get_by_metric(self, metric_name: str) -> list[AlertRule]:
        """Get rules for a specific metric.

        Args:
            metric_name: Metric name

        Returns:
            List of AlertRule instances
        """
        return (
            self.session.query(AlertRule)
            .filter(
                and_(
                    AlertRule.metric_name == metric_name,
                    AlertRule.enabled.is_(True),
                )
            )
            .all()
        )

    def update_last_triggered(self, rule_id: str) -> None:
        """Update last_triggered timestamp for rule.

        Args:
            rule_id: Unique rule identifier
        """
        rule = self.get_by_rule_id(rule_id)
        if rule:
            rule.last_triggered = datetime.now(timezone.utc)
            self.session.flush()


class DriftResultRepository(BaseRepository):
    """Repository for drift detection results."""

    def __init__(self, session: Session) -> None:
        """Initialize drift result repository."""
        super().__init__(session, DriftResult)

    def get_recent_by_feature(
        self, feature_name: str, hours: int = 24, limit: int = 100
    ) -> list[DriftResult]:
        """Get recent drift results for feature.

        Args:
            feature_name: Feature name
            hours: Number of hours to look back
            limit: Maximum results

        Returns:
            List of DriftResult instances
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return (
            self.session.query(DriftResult)
            .filter(
                and_(
                    DriftResult.feature_name == feature_name,
                    DriftResult.timestamp >= cutoff_time,
                )
            )
            .order_by(desc(DriftResult.timestamp))
            .limit(limit)
            .all()
        )

    def get_detected_drifts(
        self, hours: int = 24
    ) -> list[DriftResult]:
        """Get all detected drifts in time period.

        Args:
            hours: Number of hours to look back

        Returns:
            List of DriftResult instances where drift was detected
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return (
            self.session.query(DriftResult)
            .filter(
                and_(
                    DriftResult.timestamp >= cutoff_time,
                    DriftResult.drift_detected.is_(True),
                )
            )
            .order_by(desc(DriftResult.timestamp))
            .all()
        )


class FeatureImportanceRepository(BaseRepository):
    """Repository for feature importance history."""

    def __init__(self, session: Session) -> None:
        """Initialize feature importance repository."""
        super().__init__(session, FeatureImportance)

    def get_trend_for_feature(
        self,
        feature_name: str,
        model_version: str | None = None,
        limit: int = 100,
    ) -> list[FeatureImportance]:
        """Get importance trend for feature.

        Args:
            feature_name: Feature name
            model_version: Optional model version filter
            limit: Maximum results

        Returns:
            List of FeatureImportance instances sorted by timestamp
        """
        query = self.session.query(FeatureImportance).filter(
            FeatureImportance.feature_name == feature_name
        )

        if model_version:
            query = query.filter(FeatureImportance.model_version == model_version)

        return (
            query.order_by(desc(FeatureImportance.timestamp)).limit(limit).all()
        )

    def get_latest_for_model(
        self, model_version: str
    ) -> list[FeatureImportance]:
        """Get latest feature importances for model.

        Args:
            model_version: Model version string

        Returns:
            List of most recent FeatureImportance per feature
        """
        # Subquery to get latest timestamp per feature
        subquery = (
            self.session.query(
                FeatureImportance.feature_name,
                func.max(FeatureImportance.timestamp).label("max_timestamp"),
            )
            .filter(FeatureImportance.model_version == model_version)
            .group_by(FeatureImportance.feature_name)
            .subquery()
        )

        return (
            self.session.query(FeatureImportance)
            .join(
                subquery,
                and_(
                    FeatureImportance.feature_name == subquery.c.feature_name,
                    FeatureImportance.timestamp == subquery.c.max_timestamp,
                    FeatureImportance.model_version == model_version,
                ),
            )
            .all()
        )


class DegradationAnalysisRepository(BaseRepository):
    """Repository for degradation analysis records."""

    def __init__(self, session: Session) -> None:
        """Initialize degradation analysis repository."""
        super().__init__(session, DegradationAnalysis)

    def get_recent_by_model(
        self, model_version: str, limit: int = 50
    ) -> list[DegradationAnalysis]:
        """Get recent degradation analyses for model.

        Args:
            model_version: Model version string
            limit: Maximum results

        Returns:
            List of DegradationAnalysis instances
        """
        return (
            self.session.query(DegradationAnalysis)
            .filter(DegradationAnalysis.model_version == model_version)
            .order_by(desc(DegradationAnalysis.timestamp))
            .limit(limit)
            .all()
        )


class MetricSnapshotRepository(BaseRepository):
    """Repository for metric snapshots."""

    def __init__(self, session: Session) -> None:
        """Initialize metric snapshot repository."""
        super().__init__(session, MetricSnapshot)

    def get_metric_history(
        self,
        model_version: str,
        metric_name: str,
        hours: int = 24,
    ) -> list[MetricSnapshot]:
        """Get metric history for time period.

        Args:
            model_version: Model version string
            metric_name: Metric name
            hours: Number of hours to look back

        Returns:
            List of MetricSnapshot instances
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return (
            self.session.query(MetricSnapshot)
            .filter(
                and_(
                    MetricSnapshot.model_version == model_version,
                    MetricSnapshot.metric_name == metric_name,
                    MetricSnapshot.timestamp >= cutoff_time,
                )
            )
            .order_by(desc(MetricSnapshot.timestamp))
            .all()
        )
