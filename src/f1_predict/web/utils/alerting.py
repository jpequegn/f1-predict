"""Alerting system for model monitoring.

Provides:
- Alert creation and management
- Threshold-based alerts
- Multiple alert channels (email, console, file)
- Alert history and tracking
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
import time
from typing import Callable, Optional

import structlog

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""

    CONSOLE = "console"
    EMAIL = "email"
    SLACK = "slack"
    FILE = "file"


@dataclass
class Alert:
    """Monitoring alert."""

    timestamp: float
    alert_id: str
    severity: str  # "info", "warning", "critical"
    title: str
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    component: str  # "performance", "drift", "degradation"
    model_version: str
    acknowledged: bool = False
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AlertRule:
    """Rule for generating alerts."""

    rule_id: str
    metric_name: str
    metric_type: str  # "threshold", "degradation", "drift"
    threshold: float
    comparison: str  # "<", ">", "==", "!="
    severity: str
    component: str
    enabled: bool = True
    cooldown_minutes: int = 60  # Minimum time between alerts
    channels: list[str] = field(default_factory=lambda: ["console"])
    conditions: dict = field(default_factory=dict)  # Additional conditions

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class AlertingSystem:
    """Manages alerts and alert rules."""

    def __init__(self, data_dir: Path | str = "data/monitoring"):
        """Initialize alerting system.

        Args:
            data_dir: Directory for alert storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_file = self.data_dir / "alerts.jsonl"
        self.rules_file = self.data_dir / "alert_rules.json"
        self.logger = logger.bind(component="alerting_system")
        self.alert_callbacks: dict[str, list[Callable]] = {}
        self._last_alert_times: dict[str, float] = {}
        self._load_rules()

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
    ) -> Alert:
        """Create an alert.

        Args:
            severity: Alert severity
            title: Alert title
            message: Alert message
            metric_name: Name of monitored metric
            metric_value: Current metric value
            threshold: Threshold value
            component: Component being monitored
            model_version: Model version

        Returns:
            Created alert
        """
        alert_id = f"alert_{int(time.time() * 1000)}"

        alert = Alert(
            timestamp=time.time(),
            alert_id=alert_id,
            severity=severity,
            title=title,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            component=component,
            model_version=model_version,
        )

        # Store alert
        with open(self.alerts_file, "a") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")

        # Log and trigger callbacks
        self.logger.info(
            "alert_created",
            alert_id=alert_id,
            severity=severity,
            title=title,
            component=component,
        )

        self._trigger_callbacks(alert)

        return alert

    def evaluate_rule(
        self,
        rule: AlertRule,
        metric_value: float,
        model_version: str,
    ) -> Optional[Alert]:
        """Evaluate alert rule and create alert if triggered.

        Args:
            rule: Alert rule to evaluate
            metric_value: Current metric value
            model_version: Model version

        Returns:
            Alert if triggered, None otherwise
        """
        if not rule.enabled:
            return None

        # Check cooldown
        rule_key = rule.rule_id
        if rule_key in self._last_alert_times:
            cooldown_seconds = rule.cooldown_minutes * 60
            if time.time() - self._last_alert_times[rule_key] < cooldown_seconds:
                return None

        # Evaluate comparison
        triggered = False
        if rule.comparison == ">":
            triggered = metric_value > rule.threshold
        elif rule.comparison == "<":
            triggered = metric_value < rule.threshold
        elif rule.comparison == "==":
            triggered = metric_value == rule.threshold
        elif rule.comparison == "!=":
            triggered = metric_value != rule.threshold

        if not triggered:
            return None

        # Check additional conditions
        for condition_key, condition_value in rule.conditions.items():
            if condition_key not in {"min_value", "max_value"}:
                continue
            if condition_key == "min_value" and metric_value < condition_value:
                return None
            if condition_key == "max_value" and metric_value > condition_value:
                return None

        # Create alert
        message = (
            f"{rule.metric_name} is {metric_value:.4f} "
            f"({rule.comparison} {rule.threshold:.4f})"
        )

        alert = self.create_alert(
            severity=rule.severity,
            title=f"{rule.component.upper()}: {rule.metric_name} Alert",
            message=message,
            metric_name=rule.metric_name,
            metric_value=metric_value,
            threshold=rule.threshold,
            component=rule.component,
            model_version=model_version,
        )

        # Update cooldown
        self._last_alert_times[rule_key] = time.time()

        # Deliver to channels
        for channel in rule.channels:
            self._deliver_alert(alert, channel)

        return alert

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: Rule to add
        """
        rules = self._load_rules() or []
        rules.append(rule.to_dict())

        with open(self.rules_file, "w") as f:
            json.dump(rules, f, indent=2)

        self.logger.info("alert_rule_added", rule_id=rule.rule_id)

    def get_alerts(
        self, limit: int = 100, severity: Optional[str] = None
    ) -> list[Alert]:
        """Get recent alerts.

        Args:
            limit: Maximum alerts to return
            severity: Filter by severity (None = all)

        Returns:
            List of alerts
        """
        alerts = []
        if self.alerts_file.exists():
            with open(self.alerts_file) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if severity is None or data["severity"] == severity:
                            alerts.append(Alert(**data))

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:limit]

    def acknowledge_alert(
        self, alert_id: str, acknowledged_by: Optional[str] = None
    ) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: User who acknowledged

        Returns:
            True if successful, False otherwise
        """
        try:
            alerts = self.get_alerts(limit=10000)

            # Find and update alert
            for alert in alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_at = time.time()
                    alert.acknowledged_by = acknowledged_by

                    # Rewrite file
                    with open(self.alerts_file, "w") as f:
                        for a in sorted(alerts, key=lambda x: x.timestamp):
                            f.write(json.dumps(a.to_dict()) + "\n")

                    self.logger.info(
                        "alert_acknowledged", alert_id=alert_id, by=acknowledged_by
                    )
                    return True

            return False
        except Exception as e:
            self.logger.error("error_acknowledging_alert", error=str(e))
            return False

    def get_alert_statistics(self) -> dict:
        """Get alert statistics.

        Returns:
            Dictionary with alert statistics
        """
        alerts = self.get_alerts(limit=10000)

        stats = {
            "total_alerts": len(alerts),
            "unacknowledged": len([a for a in alerts if not a.acknowledged]),
            "by_severity": {
                "critical": len([a for a in alerts if a.severity == "critical"]),
                "warning": len([a for a in alerts if a.severity == "warning"]),
                "info": len([a for a in alerts if a.severity == "info"]),
            },
            "by_component": {},
        }

        for alert in alerts:
            if alert.component not in stats["by_component"]:
                stats["by_component"][alert.component] = 0
            stats["by_component"][alert.component] += 1

        return stats

    def register_callback(self, channel: str, callback: Callable[[Alert], None]) -> None:
        """Register callback for alert delivery.

        Args:
            channel: Alert channel name
            callback: Callback function to call with alert
        """
        if channel not in self.alert_callbacks:
            self.alert_callbacks[channel] = []
        self.alert_callbacks[channel].append(callback)

    def _trigger_callbacks(self, alert: Alert) -> None:
        """Trigger registered callbacks for alert.

        Args:
            alert: Alert to trigger callbacks for
        """
        # Log alert
        log_level = "error" if alert.severity == "critical" else "warning"
        getattr(self.logger, log_level)(
            "alert_triggered",
            alert_id=alert.alert_id,
            severity=alert.severity,
            title=alert.title,
        )

        # Trigger callbacks
        for channel_callbacks in self.alert_callbacks.values():
            for callback in channel_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error("error_in_alert_callback", error=str(e))

    def _deliver_alert(self, alert: Alert, channel: str) -> None:
        """Deliver alert to specified channel.

        Args:
            alert: Alert to deliver
            channel: Channel name
        """
        try:
            if channel == "console":
                print(
                    f"[{alert.severity.upper()}] {alert.title}: {alert.message}"
                )
            elif channel == "file":
                # Write to alert log file
                alert_log = self.data_dir / "alert_log.txt"
                with open(alert_log, "a") as f:
                    f.write(
                        f"[{alert.timestamp}] [{alert.severity}] {alert.title}\n"
                    )
                    f.write(f"  {alert.message}\n\n")
            # Email and Slack would require additional configuration
        except Exception as e:
            self.logger.error("error_delivering_alert", channel=channel, error=str(e))

    def _load_rules(self) -> Optional[list[dict]]:
        """Load alert rules from file.

        Returns:
            List of rule dictionaries or None
        """
        try:
            if self.rules_file.exists():
                with open(self.rules_file) as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error("error_loading_rules", error=str(e))
        return None
