"""Tests for database configuration and repository functionality."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from f1_predict.web.utils.database import DatabaseConfig
from f1_predict.web.utils.database_models import (
    Base,
)
from f1_predict.web.utils.database_repositories import (
    AlertRepository,
    AlertRuleRepository,
    DriftResultRepository,
    FeatureImportanceRepository,
    HealthSnapshotRepository,
    PredictionRepository,
)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db_url = f"sqlite:///{db_path}"

        # Create engine and tables
        engine = create_engine(db_url)
        Base.metadata.create_all(engine)

        # Create session factory
        session_local = sessionmaker(bind=engine)  # noqa: N806
        yield session_local

        # Cleanup
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def session(temp_db):
    """Create a database session for testing."""
    session_local = temp_db
    db_session = session_local()
    yield db_session
    db_session.close()


class TestDatabaseConfig:
    """Test DatabaseConfig class."""

    def test_sqlite_config_default(self):
        """Test default SQLite configuration."""
        config = DatabaseConfig()
        assert config.db_type == "sqlite"
        assert "sqlite:///" in config.url

    def test_sqlite_config_with_custom_path(self, monkeypatch):
        """Test SQLite configuration with custom path."""
        monkeypatch.setenv("MONITORING_DB_PATH", "/custom/path/db.sqlite")
        config = DatabaseConfig()
        assert config.db_type == "sqlite"
        assert "/custom/path/db.sqlite" in config.url

    def test_postgresql_config(self, monkeypatch):
        """Test PostgreSQL configuration."""
        monkeypatch.setenv("MONITORING_DB_TYPE", "postgresql")
        monkeypatch.setenv("MONITORING_DB_HOST", "localhost")
        monkeypatch.setenv("MONITORING_DB_PORT", "5432")
        monkeypatch.setenv("MONITORING_DB_USER", "testuser")
        monkeypatch.setenv("MONITORING_DB_PASSWORD", "testpass")
        monkeypatch.setenv("MONITORING_DB_NAME", "testdb")

        config = DatabaseConfig()
        assert config.db_type == "postgresql"
        assert "testuser" in config.url
        assert "testpass" in config.url
        assert "localhost:5432" in config.url


class TestPredictionRepository:
    """Test PredictionRepository."""

    def test_create_prediction(self, session):
        """Test creating a prediction."""
        repo = PredictionRepository(session)

        pred = repo.create(
            timestamp=datetime.now(timezone.utc),
            model_version="v1.0",
            prediction_id="pred_001",
            predicted_outcome=1,
            confidence=0.85,
            features={"feature_1": 10, "feature_2": 20},
        )

        assert pred.id is not None
        assert pred.prediction_id == "pred_001"
        assert pred.confidence == 0.85

    def test_get_by_prediction_id(self, session):
        """Test retrieving prediction by ID."""
        repo = PredictionRepository(session)

        repo.create(
            timestamp=datetime.now(timezone.utc),
            model_version="v1.0",
            prediction_id="pred_001",
            predicted_outcome=1,
            confidence=0.85,
            features={"feature_1": 10},
        )

        found = repo.get_by_prediction_id("pred_001")
        assert found is not None
        assert found.prediction_id == "pred_001"

    def test_batch_create(self, session):
        """Test batch creating predictions."""
        repo = PredictionRepository(session)

        predictions = [
            {
                "timestamp": datetime.now(timezone.utc),
                "model_version": "v1.0",
                "prediction_id": f"pred_{i:03d}",
                "predicted_outcome": i % 2,
                "confidence": 0.5 + (i * 0.01),
                "features": {"feature_1": i},
            }
            for i in range(100)
        ]

        count = repo.batch_create(predictions)
        assert count == 100

        # Verify all were created
        all_preds = repo.get_all()
        assert len(all_preds) == 100

    def test_get_recent(self, session):
        """Test getting recent predictions."""
        repo = PredictionRepository(session)
        now = datetime.now(timezone.utc)

        # Create predictions at different times
        for i in range(10):
            repo.create(
                timestamp=now - timedelta(hours=i),
                model_version="v1.0",
                prediction_id=f"pred_{i:03d}",
                predicted_outcome=i % 2,
                confidence=0.5 + (i * 0.01),
                features={"feature_1": i},
            )

        # Get last 24 hours
        recent = repo.get_recent("v1.0", hours=24, limit=100)
        assert len(recent) == 10  # All within 24 hours

        # Get last 5 hours
        recent = repo.get_recent("v1.0", hours=5, limit=100)
        assert len(recent) == 5  # Only 5 within 5 hours

    def test_get_by_confidence_range(self, session):
        """Test filtering by confidence range."""
        repo = PredictionRepository(session)

        # Create predictions with different confidences
        for i in range(10):
            repo.create(
                timestamp=datetime.now(timezone.utc),
                model_version="v1.0",
                prediction_id=f"pred_{i:03d}",
                predicted_outcome=i % 2,
                confidence=0.5 + (i * 0.05),  # 0.5 to 0.95
                features={"feature_1": i},
            )

        # Get predictions with confidence 0.65 to 0.85
        results = repo.get_by_confidence_range("v1.0", 0.65, 0.85)
        assert all(0.65 <= p.confidence <= 0.85 for p in results)


class TestAlertRepository:
    """Test AlertRepository."""

    def test_create_alert(self, session):
        """Test creating an alert."""
        repo = AlertRepository(session)

        alert = repo.create(
            timestamp=datetime.now(timezone.utc),
            alert_id="alert_001",
            severity="critical",
            title="Model Degradation",
            message="Accuracy dropped below threshold",
            metric_name="accuracy",
            metric_value=0.75,
            threshold=0.85,
            component="performance",
            model_version="v1.0",
        )

        assert alert.alert_id == "alert_001"
        assert alert.severity == "critical"

    def test_acknowledge_alert(self, session):
        """Test acknowledging an alert."""
        repo = AlertRepository(session)

        repo.create(
            timestamp=datetime.now(timezone.utc),
            alert_id="alert_001",
            severity="critical",
            title="Test",
            message="Test alert",
            metric_name="accuracy",
            metric_value=0.75,
            threshold=0.85,
            component="performance",
            model_version="v1.0",
        )

        acknowledged = repo.acknowledge("alert_001", acknowledged_by="user123")
        assert acknowledged is not None
        assert acknowledged.acknowledged  # Should be True
        assert acknowledged.acknowledged_by == "user123"

    def test_get_unacknowledged(self, session):
        """Test getting unacknowledged alerts."""
        repo = AlertRepository(session)

        # Create mix of acknowledged and unacknowledged alerts
        for i in range(10):
            repo.create(
                timestamp=datetime.now(timezone.utc),
                alert_id=f"alert_{i:03d}",
                severity="warning",
                title=f"Alert {i}",
                message="Test",
                metric_name="accuracy",
                metric_value=0.75,
                threshold=0.85,
                component="performance",
                model_version="v1.0",
                acknowledged=(i % 2 == 0),  # Even are acknowledged
            )

        unack = repo.get_unacknowledged()
        assert len(unack) == 5  # 5 unacknowledged
        assert all(not ua.acknowledged for ua in unack)

    def test_get_statistics(self, session):
        """Test getting alert statistics."""
        repo = AlertRepository(session)

        # Create alerts with different severities
        severities = ["critical", "critical", "warning", "warning", "warning", "info"]
        for i, sev in enumerate(severities):
            repo.create(
                timestamp=datetime.now(timezone.utc),
                alert_id=f"alert_{i:03d}",
                severity=sev,
                title="Test",
                message="Test",
                metric_name="accuracy",
                metric_value=0.75,
                threshold=0.85,
                component="performance",
                model_version="v1.0",
            )

        stats = repo.get_statistics()
        assert stats["total"] == 6
        assert stats["by_severity"]["critical"] == 2
        assert stats["by_severity"]["warning"] == 3
        assert stats["by_severity"]["info"] == 1


class TestHealthSnapshotRepository:
    """Test HealthSnapshotRepository."""

    def test_create_snapshot(self, session):
        """Test creating health snapshot."""
        repo = HealthSnapshotRepository(session)

        snapshot = repo.create(
            timestamp=datetime.now(timezone.utc),
            model_version="v1.0",
            accuracy=0.85,
            precision=0.87,
            recall=0.83,
            f1_score=0.85,
            num_predictions=1000,
        )

        assert snapshot.accuracy == 0.85
        assert snapshot.model_version == "v1.0"

    def test_get_latest_by_model(self, session):
        """Test getting latest snapshots."""
        repo = HealthSnapshotRepository(session)
        now = datetime.now(timezone.utc)

        # Create multiple snapshots
        for i in range(5):
            repo.create(
                timestamp=now - timedelta(hours=i),
                model_version="v1.0",
                accuracy=0.80 + (i * 0.02),
                precision=0.82,
                recall=0.78,
                f1_score=0.80,
                num_predictions=1000 + (i * 100),
            )

        latest = repo.get_latest_by_model("v1.0", limit=10)
        assert len(latest) == 5
        # Should be sorted by timestamp descending
        assert latest[0].timestamp > latest[-1].timestamp


class TestAlertRuleRepository:
    """Test AlertRuleRepository."""

    def test_create_rule(self, session):
        """Test creating alert rule."""
        repo = AlertRuleRepository(session)

        rule = repo.create(
            rule_id="rule_001",
            metric_name="accuracy",
            metric_type="threshold",
            threshold=0.85,
            comparison="<",
            severity="critical",
            component="performance",
            enabled=True,
            cooldown_minutes=60,
            channels=["email", "slack"],
        )

        assert rule.rule_id == "rule_001"
        assert rule.enabled  # Should be True

    def test_get_enabled_rules(self, session):
        """Test getting enabled rules."""
        repo = AlertRuleRepository(session)

        # Create mix of enabled and disabled rules
        for i in range(5):
            repo.create(
                rule_id=f"rule_{i:03d}",
                metric_name="accuracy",
                metric_type="threshold",
                threshold=0.85,
                comparison="<",
                severity="critical",
                component="performance",
                enabled=(i % 2 == 0),
                cooldown_minutes=60,
                channels=["email"],
            )

        enabled = repo.get_enabled()
        assert len(enabled) == 3  # 3 enabled (0, 2, 4)
        assert all(r.enabled for r in enabled)


class TestDriftResultRepository:
    """Test DriftResultRepository."""

    def test_create_drift_result(self, session):
        """Test creating drift result."""
        repo = DriftResultRepository(session)

        drift = repo.create(
            timestamp=datetime.now(timezone.utc),
            feature_name="feature_1",
            model_version="v1.0",
            drift_detected=True,
            test_statistic=0.42,
            p_value=0.001,
            threshold=0.05,
            drift_type="shift",
            baseline_stats={"mean": 100, "std": 15},
            current_stats={"mean": 105, "std": 15},
            severity="high",
        )

        assert drift.drift_detected  # Should be True
        assert drift.drift_type == "shift"

    def test_get_detected_drifts(self, session):
        """Test getting detected drifts."""
        repo = DriftResultRepository(session)
        now = datetime.now(timezone.utc)

        # Create mix of detected and not detected
        for i in range(10):
            repo.create(
                timestamp=now - timedelta(hours=i),
                feature_name=f"feature_{i % 3}",
                model_version="v1.0",
                drift_detected=(i % 2 == 0),
                test_statistic=0.5 - (i * 0.02),
                p_value=0.01 if (i % 2 == 0) else 0.5,
                threshold=0.05,
                drift_type="shift" if (i % 2 == 0) else "none",
                baseline_stats={"mean": 100},
                current_stats={"mean": 105},
            )

        detected = repo.get_detected_drifts(hours=24)
        assert len(detected) == 5  # 5 detected
        assert all(d.drift_detected for d in detected)


class TestFeatureImportanceRepository:
    """Test FeatureImportanceRepository."""

    def test_create_importance(self, session):
        """Test creating feature importance record."""
        repo = FeatureImportanceRepository(session)

        fi = repo.create(
            timestamp=datetime.now(timezone.utc),
            model_version="v1.0",
            feature_name="feature_1",
            importance_score=0.35,
            shap_value=0.35,
            percentage=25.0,
            rank=1,
        )

        assert fi.feature_name == "feature_1"
        assert fi.rank == 1

    def test_get_trend_for_feature(self, session):
        """Test getting importance trend."""
        repo = FeatureImportanceRepository(session)
        now = datetime.now(timezone.utc)

        # Create importance records over time
        for i in range(10):
            repo.create(
                timestamp=now - timedelta(hours=i),
                model_version="v1.0",
                feature_name="feature_1",
                importance_score=0.30 + (i * 0.01),
                shap_value=0.30 + (i * 0.01),
                percentage=25.0 + (i * 0.5),
                rank=1,
            )

        trend = repo.get_trend_for_feature("feature_1", "v1.0")
        assert len(trend) == 10
        # Should be sorted by timestamp descending
        assert trend[0].timestamp > trend[-1].timestamp
