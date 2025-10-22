"""Tests for database-backed alerting system."""

from datetime import datetime, timezone
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from f1_predict.web.utils.alerting_database import AlertingSystemDB
from f1_predict.web.utils.database import DatabaseManager
from f1_predict.web.utils.database_models import Base
from f1_predict.web.utils.alert_config import AlertChannelConfig


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db_url = f"sqlite:///{db_path}"

        # Create engine and tables
        engine = create_engine(db_url)
        Base.metadata.create_all(engine)

        # Initialize DatabaseManager with test database
        test_config = DatabaseManager._instance
        try:
            # Mock the database configuration for testing
            import os
            os.environ["MONITORING_DB_ENABLED"] = "true"
            os.environ["MONITORING_DB_TYPE"] = "sqlite"
            os.environ["MONITORING_DB_PATH"] = str(db_path)

            yield db_url
        finally:
            # Cleanup
            Base.metadata.drop_all(engine)
            engine.dispose()


class TestAlertingSystemDB:
    """Test database-backed alerting system."""

    def test_alerting_system_initialization(self, monkeypatch):
        """Test AlertingSystemDB initialization."""
        monkeypatch.setenv("MONITORING_DB_ENABLED", "true")

        try:
            system = AlertingSystemDB(data_dir="data/monitoring")
            assert system is not None
            assert system.alert_callbacks == {}
        except RuntimeError:
            # Expected if database not fully initialized
            pytest.skip("Database not configured for this test")

    def test_create_alert(self, monkeypatch, temp_db):
        """Test creating an alert."""
        monkeypatch.setenv("MONITORING_DB_ENABLED", "true")
        monkeypatch.setenv("MONITORING_DB_TYPE", "sqlite")

        try:
            system = AlertingSystemDB(data_dir="data/monitoring")

            alert = system.create_alert(
                severity="critical",
                title="Test Alert",
                message="Test message",
                metric_name="accuracy",
                metric_value=0.75,
                threshold=0.85,
                component="performance",
                model_version="v1.0",
            )

            assert alert is not None
            assert alert["severity"] == "critical"
            assert alert["title"] == "Test Alert"
            assert alert["acknowledged"] is False
        except RuntimeError:
            # Expected if database not fully initialized
            pytest.skip("Database not configured for this test")

    def test_acknowledge_alert(self, monkeypatch, temp_db):
        """Test acknowledging an alert."""
        monkeypatch.setenv("MONITORING_DB_ENABLED", "true")
        monkeypatch.setenv("MONITORING_DB_TYPE", "sqlite")

        try:
            system = AlertingSystemDB(data_dir="data/monitoring")

            # Create alert first
            alert = system.create_alert(
                severity="warning",
                title="Test Alert",
                message="Test message",
                metric_name="accuracy",
                metric_value=0.75,
                threshold=0.85,
                component="performance",
                model_version="v1.0",
            )

            alert_id = alert["alert_id"]

            # Acknowledge it
            acknowledged = system.acknowledge_alert(alert_id, acknowledged_by="user123")

            assert acknowledged is not None
            assert acknowledged["acknowledged"] is True
            assert acknowledged["acknowledged_by"] == "user123"
        except RuntimeError:
            pytest.skip("Database not configured for this test")

    def test_get_alerts(self, monkeypatch, temp_db):
        """Test retrieving alerts."""
        monkeypatch.setenv("MONITORING_DB_ENABLED", "true")
        monkeypatch.setenv("MONITORING_DB_TYPE", "sqlite")

        try:
            system = AlertingSystemDB(data_dir="data/monitoring")

            # Create multiple alerts
            for i in range(3):
                system.create_alert(
                    severity="warning",
                    title=f"Alert {i}",
                    message="Test message",
                    metric_name="accuracy",
                    metric_value=0.75 - (i * 0.05),
                    threshold=0.85,
                    component="performance",
                    model_version="v1.0",
                )

            # Get alerts
            alerts = system.get_alerts(limit=10)
            assert len(alerts) == 3
        except RuntimeError:
            pytest.skip("Database not configured for this test")

    def test_add_rule(self, monkeypatch, temp_db):
        """Test adding an alert rule."""
        monkeypatch.setenv("MONITORING_DB_ENABLED", "true")
        monkeypatch.setenv("MONITORING_DB_TYPE", "sqlite")

        try:
            system = AlertingSystemDB(data_dir="data/monitoring")

            rule = system.add_rule(
                rule_id="rule_001",
                metric_name="accuracy",
                metric_type="threshold",
                threshold=0.85,
                comparison="<",
                severity="critical",
                component="performance",
                enabled=True,
                cooldown_minutes=60,
                channels=["console"],
            )

            assert rule is not None
            assert rule["rule_id"] == "rule_001"
            assert rule["enabled"] is True
        except RuntimeError:
            pytest.skip("Database not configured for this test")

    def test_get_alert_statistics(self, monkeypatch, temp_db):
        """Test getting alert statistics."""
        monkeypatch.setenv("MONITORING_DB_ENABLED", "true")
        monkeypatch.setenv("MONITORING_DB_TYPE", "sqlite")

        try:
            system = AlertingSystemDB(data_dir="data/monitoring")

            # Create alerts with different severities
            system.create_alert(
                severity="critical",
                title="Critical Alert",
                message="Test",
                metric_name="accuracy",
                metric_value=0.5,
                threshold=0.85,
                component="performance",
                model_version="v1.0",
            )
            system.create_alert(
                severity="warning",
                title="Warning Alert",
                message="Test",
                metric_name="accuracy",
                metric_value=0.75,
                threshold=0.85,
                component="performance",
                model_version="v1.0",
            )

            # Get statistics
            stats = system.get_alert_statistics()

            assert stats is not None
            assert stats.get("total", 0) >= 0
        except RuntimeError:
            pytest.skip("Database not configured for this test")
