"""Database-backed alerting system implementation.

Replaces file-based JSON storage with SQLAlchemy ORM for improved
persistence, querying, and alert management.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
import uuid

import structlog

from f1_predict.web.utils.alert_channels import EmailAlertChannel, SlackAlertChannel
from f1_predict.web.utils.alert_config import (
    AlertChannelConfig,
    load_alert_config_from_env,
)
from f1_predict.web.utils.database import DatabaseManager
from f1_predict.web.utils.database_repositories import (
    AlertRepository,
    AlertRuleRepository,
)

logger = structlog.get_logger(__name__)


class AlertingSystemDB:
    """Database-backed alerting system.

    Drop-in replacement for file-based AlertingSystem using SQLAlchemy ORM.
    """

    def __init__(
        self,
        data_dir: Path | str = "data/monitoring",
        channel_config: Optional[AlertChannelConfig] = None,
    ) -> None:
        """Initialize database-backed alerting system.

        Args:
            data_dir: Kept for backward compatibility, not used with database backend
            channel_config: Alert channel configuration (loads from env if None)
        """
        self.data_dir = Path(data_dir)
        self.logger = logger.bind(component="alerting_system_db")

        if not DatabaseManager.is_enabled():
            raise RuntimeError(
                "Database backend not enabled. Set MONITORING_DB_ENABLED=true"
            )

        self.alert_callbacks: dict[str, list[Callable]] = {}

        # Initialize alert channels
        if channel_config is None:
            channel_config = load_alert_config_from_env()

        self.channel_config = channel_config
        self.email_channel: Optional[EmailAlertChannel] = None
        self.slack_channel: Optional[SlackAlertChannel] = None

        self._initialize_channels()

    def _initialize_channels(self) -> None:
        """Initialize alert delivery channels based on configuration."""
        try:
            if self.channel_config.email_enabled:
                self.email_channel = EmailAlertChannel(self.channel_config.email_config)
                self.logger.info("email_channel_initialized")
        except Exception as e:
            self.logger.warning("email_channel_initialization_failed", error=str(e))

        try:
            if self.channel_config.slack_enabled:
                self.slack_channel = SlackAlertChannel(self.channel_config.slack_config)
                self.logger.info("slack_channel_initialized")
        except Exception as e:
            self.logger.warning("slack_channel_initialization_failed", error=str(e))

    def create_alert(
        self,
        severity: str,
        title: str,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
        component: str,
        model_version: str,
    ) -> dict[str, Any]:
        """Create an alert and store in database.

        Args:
            severity: Alert severity (info, warning, critical)
            title: Alert title
            message: Alert message
            metric_name: Name of monitored metric
            metric_value: Current metric value
            threshold: Threshold value
            component: Component being monitored (performance, drift, degradation)
            model_version: Model version

        Returns:
            Created alert dictionary
        """
        try:
            alert_id = str(uuid.uuid4())

            with DatabaseManager.session_scope() as session:
                repo = AlertRepository(session)
                alert = repo.create(
                    timestamp=datetime.now(timezone.utc),
                    alert_id=alert_id,
                    severity=severity,
                    title=title,
                    message=message,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    threshold=threshold,
                    component=component,
                    model_version=model_version,
                    acknowledged=False,
                )

                result = alert.to_dict()

                self.logger.info(
                    "alert_created",
                    alert_id=alert_id,
                    severity=severity,
                    component=component,
                )

                # Execute callbacks
                if component in self.alert_callbacks:
                    for callback in self.alert_callbacks[component]:
                        try:
                            callback(result)
                        except Exception as e:
                            self.logger.error(
                                "alert_callback_failed",
                                alert_id=alert_id,
                                error=str(e),
                            )

                return result

        except Exception as e:
            self.logger.error("alert_creation_failed", error=str(e))
            raise

    def acknowledge_alert(
        self, alert_id: str, acknowledged_by: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Acknowledge an alert.

        Args:
            alert_id: Alert identifier
            acknowledged_by: User or system that acknowledged the alert

        Returns:
            Updated alert dictionary or None if not found
        """
        try:
            with DatabaseManager.session_scope() as session:
                repo = AlertRepository(session)
                alert = repo.acknowledge(alert_id, acknowledged_by=acknowledged_by)

                if alert:
                    result = alert.to_dict()
                    self.logger.info(
                        "alert_acknowledged",
                        alert_id=alert_id,
                        acknowledged_by=acknowledged_by,
                    )
                    return result
                self.logger.warning("alert_not_found_for_acknowledgment", alert_id=alert_id)
                return None

        except Exception as e:
            self.logger.error("alert_acknowledgment_failed", error=str(e))
            return None

    def get_alerts(
        self,
        limit: int = 100,
        severity: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get recent alerts from database.

        Args:
            limit: Maximum number of alerts to return
            severity: Filter by severity (None = all)

        Returns:
            List of alert dictionaries
        """
        try:
            with DatabaseManager.session_scope() as session:
                repo = AlertRepository(session)
                alerts = repo.get_recent(
                    limit=limit,
                    severity=severity,
                    acknowledged=None,
                )

                result = [a.to_dict() for a in alerts]

                self.logger.info(
                    "recent_alerts_retrieved",
                    count=len(result),
                    severity=severity,
                )
                return result

        except Exception as e:
            self.logger.error("recent_alerts_retrieval_failed", error=str(e))
            return []

    def get_unacknowledged_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get unacknowledged alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of unacknowledged alert dictionaries
        """
        try:
            with DatabaseManager.session_scope() as session:
                repo = AlertRepository(session)
                alerts = repo.get_unacknowledged(limit=limit)

                result = [a.to_dict() for a in alerts]

                self.logger.info("unacknowledged_alerts_retrieved", count=len(result))
                return result

        except Exception as e:
            self.logger.error("unacknowledged_alerts_retrieval_failed", error=str(e))
            return []

    def add_rule(
        self,
        rule_id: str,
        metric_name: str,
        metric_type: str,
        threshold: float,
        comparison: str,
        severity: str,
        component: str,
        enabled: bool = True,
        cooldown_minutes: int = 60,
        channels: Optional[list[str]] = None,
        conditions: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Add an alert rule to database.

        Args:
            rule_id: Unique rule identifier
            metric_name: Name of metric to monitor
            metric_type: Type of metric (threshold, degradation, drift)
            threshold: Threshold value
            comparison: Comparison operator (<, >, ==, !=)
            severity: Alert severity
            component: Component being monitored
            enabled: Whether rule is enabled
            cooldown_minutes: Minimum time between alerts
            channels: Delivery channels
            conditions: Additional conditions

        Returns:
            Created rule dictionary
        """
        try:
            with DatabaseManager.session_scope() as session:
                repo = AlertRuleRepository(session)
                rule = repo.create(
                    rule_id=rule_id,
                    metric_name=metric_name,
                    metric_type=metric_type,
                    threshold=threshold,
                    comparison=comparison,
                    severity=severity,
                    component=component,
                    enabled=enabled,
                    cooldown_minutes=cooldown_minutes,
                    channels=channels or ["console"],
                    conditions=conditions or {},
                )

                result = rule.to_dict()

                self.logger.info(
                    "alert_rule_added",
                    rule_id=rule_id,
                    metric_name=metric_name,
                    component=component,
                )
                return result

        except Exception as e:
            self.logger.error("alert_rule_creation_failed", error=str(e))
            raise

    def check_alert_rules(
        self,
        metric_name: str,
        metric_value: float,
        component: str,
        model_version: str,
    ) -> list[dict[str, Any]]:
        """Check alert rules and create alerts if thresholds are exceeded.

        Args:
            metric_name: Name of metric
            metric_value: Current metric value
            component: Component being monitored
            model_version: Model version

        Returns:
            List of created alerts
        """
        created_alerts = []

        try:
            with DatabaseManager.session_scope() as session:
                rule_repo = AlertRuleRepository(session)
                enabled_rules = rule_repo.get_enabled()

                # Filter rules for this metric
                matching_rules = [
                    r
                    for r in enabled_rules
                    if r.metric_name == metric_name and r.component == component
                ]

                now = datetime.now(timezone.utc)

                for rule in matching_rules:
                    # Check cooldown
                    if rule.last_triggered:
                        cooldown_seconds = rule.cooldown_minutes * 60
                        seconds_since_trigger = (now - rule.last_triggered).total_seconds()
                        if seconds_since_trigger < cooldown_seconds:
                            continue

                    # Check threshold
                    threshold_exceeded = False
                    if rule.comparison == "<":
                        threshold_exceeded = metric_value < rule.threshold
                    elif rule.comparison == ">":
                        threshold_exceeded = metric_value > rule.threshold
                    elif rule.comparison == "==":
                        threshold_exceeded = metric_value == rule.threshold
                    elif rule.comparison == "!=":
                        threshold_exceeded = metric_value != rule.threshold

                    if threshold_exceeded:
                        # Create alert
                        alert = self.create_alert(
                            severity=rule.severity,
                            title=f"{metric_name} Alert",
                            message=f"{metric_name} is {metric_value} (threshold: {rule.threshold})",
                            metric_name=metric_name,
                            metric_value=metric_value,
                            threshold=rule.threshold,
                            component=component,
                            model_version=model_version,
                        )
                        created_alerts.append(alert)

                        # Update last triggered timestamp atomically
                        rule_repo.update_last_triggered(rule.rule_id)

                        # Emit alert
                        self.emit_alert(alert, rule.channels)

        except Exception as e:
            self.logger.error("alert_rules_check_failed", error=str(e))

        return created_alerts

    def emit_alert(
        self, alert: dict[str, Any], channels: Optional[list[str]] = None
    ) -> bool:
        """Emit alert through configured channels.

        Args:
            alert: Alert dictionary
            channels: Channels to emit through (defaults to configured channels)

        Returns:
            True if at least one channel succeeded
        """
        if channels is None:
            channels = ["console"]

        success = False

        for channel in channels:
            try:
                if channel == "email" and self.email_channel:
                    self.email_channel.send_alert(alert)
                    success = True
                elif channel == "slack" and self.slack_channel:
                    self.slack_channel.send_alert(alert)
                    success = True
                elif channel == "console":
                    self.logger.info("alert_emitted", alert_id=alert.get("alert_id"))
                    success = True
            except Exception as e:
                self.logger.warning(
                    "alert_emission_failed",
                    channel=channel,
                    alert_id=alert.get("alert_id"),
                    error=str(e),
                )

        return success

    def register_callback(
        self, component: str, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """Register callback for alerts on specific component.

        Args:
            component: Component name
            callback: Callback function
        """
        if component not in self.alert_callbacks:
            self.alert_callbacks[component] = []
        self.alert_callbacks[component].append(callback)

        self.logger.info("alert_callback_registered", component=component)

    def get_alert_statistics(self) -> dict[str, Any]:
        """Get alert statistics from database.

        Returns:
            Dictionary with alert statistics
        """
        try:
            with DatabaseManager.session_scope() as session:
                repo = AlertRepository(session)
                stats = repo.get_statistics()

                self.logger.info("alert_statistics_retrieved", total=stats.get("total", 0))
                return stats

        except Exception as e:
            self.logger.error("alert_statistics_retrieval_failed", error=str(e))
            return {"total": 0, "by_severity": {}, "acknowledged_count": 0}
