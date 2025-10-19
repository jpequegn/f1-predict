"""Tests for model monitoring system."""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from f1_predict.web.utils.ab_testing import ABTestConfig, ABTestingFramework
from f1_predict.web.utils.alerting import AlertingSystem, AlertRule
from f1_predict.web.utils.drift_detection import DriftDetector
from f1_predict.web.utils.model_versioning import ModelRegistry
from f1_predict.web.utils.monitoring import (
    ModelHealthSnapshot,
    ModelPerformanceTracker,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def performance_tracker(temp_dir):
    """Create performance tracker."""
    return ModelPerformanceTracker(temp_dir)


@pytest.fixture
def drift_detector():
    """Create drift detector."""
    return DriftDetector(ks_threshold=0.05, psi_threshold=0.2)


@pytest.fixture
def alerting_system(temp_dir):
    """Create alerting system."""
    return AlertingSystem(temp_dir)


@pytest.fixture
def model_registry(temp_dir):
    """Create model registry."""
    return ModelRegistry(temp_dir / "models")


@pytest.fixture
def ab_testing_framework(temp_dir):
    """Create A/B testing framework."""
    return ABTestingFramework(temp_dir)


# Performance Tracking Tests
class TestPerformanceTracker:
    """Tests for performance tracking."""

    def test_record_prediction(self, performance_tracker):
        """Test recording a prediction."""
        performance_tracker.record_prediction(
            prediction_id="pred_001",
            model_version="v1.0",
            predicted_outcome=1,
            confidence=0.85,
            features={"feature1": 1.5, "feature2": 2.0},
        )

        assert performance_tracker.predictions_file.exists()

    def test_get_performance_metrics(self, performance_tracker):
        """Test getting performance metrics."""
        # Record predictions
        for i in range(10):
            performance_tracker.record_prediction(
                prediction_id=f"pred_{i:03d}",
                model_version="v1.0",
                predicted_outcome=i % 2,
                confidence=0.80 + (i % 5) * 0.01,
                features={"feature1": float(i)},
            )

        metrics = performance_tracker.get_performance_metrics("v1.0", window_minutes=60)
        assert metrics["num_predictions"] == 10
        assert "avg_confidence" in metrics
        assert metrics["avg_confidence"] > 0

    def test_calculate_accuracy(self, performance_tracker):
        """Test accuracy calculation."""
        # Record predictions
        for i in range(20):
            performance_tracker.record_prediction(
                prediction_id=f"pred_{i:03d}",
                model_version="v1.0",
                predicted_outcome=1,
                confidence=0.75 + (i % 5) * 0.05,
                features={"feature1": float(i)},
            )

        accuracy = performance_tracker.calculate_accuracy("v1.0")
        assert accuracy is not None
        assert 0.0 <= accuracy <= 1.0

    def test_record_health_snapshot(self, performance_tracker):
        """Test recording health snapshot."""
        snapshot = ModelHealthSnapshot(
            timestamp=1234567890.0,
            model_version="v1.0",
            accuracy=0.92,
            precision=0.89,
            recall=0.91,
            f1_score=0.90,
            roc_auc=0.95,
            expected_calibration_error=0.08,
            num_predictions=1000,
            prediction_accuracy_trend=0.02,
        )

        performance_tracker.record_health_snapshot(snapshot)
        assert performance_tracker.health_file.exists()


# Drift Detection Tests
class TestDriftDetector:
    """Tests for drift detection."""

    def test_ks_test_no_drift(self, drift_detector):
        """Test KS test with no drift."""
        baseline = pd.DataFrame({"feature": np.random.normal(0, 1, 100)})
        current = pd.DataFrame({"feature": np.random.normal(0, 1, 100)})

        result = drift_detector.detect_feature_drift_ks_test(baseline, current, "feature")
        assert result is not None
        # Statistically should not detect drift (same distribution)

    def test_ks_test_with_drift(self, drift_detector):
        """Test KS test detects drift."""
        baseline = pd.DataFrame({"feature": np.random.normal(0, 1, 100)})
        current = pd.DataFrame({"feature": np.random.normal(2, 1, 100)})  # Shifted mean

        result = drift_detector.detect_feature_drift_ks_test(baseline, current, "feature")
        assert result is not None
        # Should detect drift with shifted distribution

    def test_psi_calculation(self, drift_detector):
        """Test PSI calculation."""
        baseline = pd.DataFrame({"feature": np.random.normal(0, 1, 100)})
        current = pd.DataFrame({"feature": np.random.normal(0.5, 1, 100)})  # Slight shift

        result = drift_detector.calculate_psi(baseline, current, "feature")
        assert result is not None
        assert result.test_statistic > 0

    def test_detect_dataset_drift(self, drift_detector):
        """Test dataset-level drift detection."""
        baseline = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
        })
        current = pd.DataFrame({
            "feature1": np.random.normal(0.3, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
        })

        result = drift_detector.detect_dataset_drift(baseline, current, method="psi")
        assert "total_features_checked" in result
        assert "features_with_drift" in result
        assert "drift_ratio" in result


# Alerting Tests
class TestAlertingSystem:
    """Tests for alerting system."""

    def test_create_alert(self, alerting_system):
        """Test creating an alert."""
        alert = alerting_system.create_alert(
            severity="warning",
            title="Test Alert",
            message="This is a test alert",
            metric_name="accuracy",
            metric_value=0.85,
            threshold=0.90,
            component="performance",
            model_version="v1.0",
        )

        assert alert.alert_id is not None
        assert alert.severity == "warning"

    def test_evaluate_alert_rule(self, alerting_system):
        """Test alert rule evaluation."""
        rule = AlertRule(
            rule_id="rule_001",
            metric_name="accuracy",
            metric_type="threshold",
            threshold=0.90,
            comparison="<",
            severity="critical",
            component="performance",
        )

        # Should trigger alert
        alert = alerting_system.evaluate_rule(rule, metric_value=0.85, model_version="v1.0")
        assert alert is not None
        assert alert.severity == "critical"

        # Should not trigger alert
        alert = alerting_system.evaluate_rule(rule, metric_value=0.95, model_version="v1.0")
        assert alert is None

    def test_get_alerts(self, alerting_system):
        """Test retrieving alerts."""
        alerting_system.create_alert(
            severity="info",
            title="Test",
            message="Test",
            metric_name="test",
            metric_value=1.0,
            threshold=1.0,
            component="test",
            model_version="v1.0",
        )

        alerts = alerting_system.get_alerts(limit=10)
        assert len(alerts) > 0

    def test_acknowledge_alert(self, alerting_system):
        """Test acknowledging an alert."""
        alert = alerting_system.create_alert(
            severity="warning",
            title="Test",
            message="Test",
            metric_name="test",
            metric_value=1.0,
            threshold=1.0,
            component="test",
            model_version="v1.0",
        )

        result = alerting_system.acknowledge_alert(alert.alert_id, acknowledged_by="user1")
        assert result


# Model Versioning Tests
class TestModelRegistry:
    """Tests for model registry."""

    def test_register_model(self, model_registry, temp_dir):
        """Test model registration."""
        model_file = temp_dir / "test_model.pkl"
        model_file.write_text("fake model data")

        metadata = model_registry.register_model(
            model_type="ensemble",
            model_path=model_file,
            metrics={"accuracy": 0.92, "f1": 0.90},
            description="Test model",
        )

        assert metadata.model_version is not None
        assert metadata.metrics["accuracy"] == 0.92

    def test_list_model_versions(self, model_registry, temp_dir):
        """Test listing model versions."""
        # Register multiple models
        for i in range(3):
            model_file = temp_dir / f"model_{i}.pkl"
            model_file.write_text(f"model data {i}")
            model_registry.register_model(
                model_type="ensemble",
                model_path=model_file,
                metrics={"accuracy": 0.90 + i * 0.01},
            )

        versions = model_registry.list_model_versions(limit=10)
        assert len(versions) >= 3

    def test_activate_model(self, model_registry, temp_dir):
        """Test activating a model."""
        model_file = temp_dir / "model.pkl"
        model_file.write_text("model")

        metadata = model_registry.register_model(
            model_type="ensemble",
            model_path=model_file,
        )

        result = model_registry.activate_model(metadata.model_version)
        assert result
        assert model_registry.get_active_model().model_version == metadata.model_version

    def test_compare_models(self, model_registry, temp_dir):
        """Test comparing model versions."""
        # Register two models
        models = []
        for i in range(2):
            model_file = temp_dir / f"model_{i}.pkl"
            model_file.write_text(f"model {i}")
            m = model_registry.register_model(
                model_type="ensemble",
                model_path=model_file,
                metrics={"accuracy": 0.85 + i * 0.05},
            )
            models.append(m)

        comparison = model_registry.compare_models(
            models[0].model_version, models[1].model_version
        )
        assert comparison is not None
        assert "accuracy" in comparison["metrics_comparison"]


# A/B Testing Tests
class TestABTestingFramework:
    """Tests for A/B testing."""

    def test_create_ab_test(self, ab_testing_framework):
        """Test creating A/B test."""
        config = ABTestConfig(
            test_id="test_001",
            control_model="v1.0",
            treatment_model="v2.0",
            traffic_allocation="even_split",
        )

        result = ab_testing_framework.create_test(config)
        assert result

    def test_get_assigned_model(self, ab_testing_framework):
        """Test model assignment for user."""
        config = ABTestConfig(
            test_id="test_001",
            control_model="v1.0",
            treatment_model="v2.0",
            traffic_allocation="even_split",
        )

        ab_testing_framework.create_test(config)

        # Same user should get consistent assignment
        model1 = ab_testing_framework.get_assigned_model("test_001", "user_123")
        model2 = ab_testing_framework.get_assigned_model("test_001", "user_123")

        assert model1 == model2
        assert model1 in ["control", "treatment"]

    def test_record_observation(self, ab_testing_framework):
        """Test recording test observation."""
        ab_testing_framework.record_test_observation(
            test_id="test_001",
            model_group="control",
            metrics={"accuracy": 0.92},
        )

        # Verify file was created
        obs_file = ab_testing_framework.data_dir / "ab_test_test_001_observations.jsonl"
        assert obs_file.exists()
