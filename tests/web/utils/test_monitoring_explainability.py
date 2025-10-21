"""Tests for monitoring explainability system."""

import time

import numpy as np
import pandas as pd
import pytest

from f1_predict.web.utils.alert_enricher import ExplanabilityAlertEnricher
from f1_predict.web.utils.alerting import Alert
from f1_predict.web.utils.monitoring_explainability import (
    DriftExplanation,
    FeatureImportance,
    PerformanceDegradationAnalysis,
    ShapExplainabilityMonitor,
)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    baseline = pd.DataFrame({
        "feature_1": np.random.normal(100, 15, 100),
        "feature_2": np.random.normal(50, 10, 100),
        "feature_3": np.random.normal(75, 20, 100),
    })

    current = pd.DataFrame({
        "feature_1": np.random.normal(105, 15, 50),  # Shifted
        "feature_2": np.random.normal(50, 10, 50),
        "feature_3": np.random.normal(75, 20, 50),
    })

    return baseline, current


@pytest.fixture
def monitor(tmp_path):
    """Create temporary monitor instance."""
    return ShapExplainabilityMonitor(data_dir=tmp_path)


class TestFeatureImportance:
    """Test FeatureImportance dataclass."""

    def test_feature_importance_creation(self):
        """Test creating FeatureImportance."""
        fi = FeatureImportance(
            feature_name="accuracy",
            importance_score=0.35,
            shap_value=0.35,
            percentage=25.0,
            rank=1,
        )

        assert fi.feature_name == "accuracy"
        assert fi.importance_score == 0.35
        assert fi.rank == 1

    def test_feature_importance_to_dict(self):
        """Test converting to dictionary."""
        fi = FeatureImportance(
            feature_name="test",
            importance_score=0.5,
            shap_value=0.5,
            percentage=50.0,
            rank=1,
        )

        result = fi.to_dict()
        assert result["feature_name"] == "test"
        assert result["importance_score"] == 0.5


class TestDriftExplanation:
    """Test DriftExplanation dataclass."""

    def test_drift_explanation_creation(self):
        """Test creating DriftExplanation."""
        explanation = DriftExplanation(
            feature_name="accuracy",
            drift_type="shift",
            baseline_mean=0.85,
            current_mean=0.75,
            baseline_std=0.05,
            current_std=0.08,
            shap_contribution=0.12,
            contributing_features=["feature_1", "feature_2"],
            confidence=0.92,
            recommendation="Retrain the model",
        )

        assert explanation.feature_name == "accuracy"
        assert explanation.drift_type == "shift"
        assert explanation.confidence == 0.92

    def test_drift_explanation_to_dict(self):
        """Test converting drift explanation to dict."""
        explanation = DriftExplanation(
            feature_name="test",
            drift_type="scale",
            baseline_mean=10.0,
            current_mean=10.5,
            baseline_std=2.0,
            current_std=3.0,
            shap_contribution=0.15,
            contributing_features=["feat1"],
            confidence=0.85,
            recommendation="Monitor",
        )

        result = explanation.to_dict()
        assert result["feature_name"] == "test"
        assert result["drift_type"] == "scale"


class TestPerformanceDegradationAnalysis:
    """Test PerformanceDegradationAnalysis dataclass."""

    def test_degradation_analysis_creation(self):
        """Test creating analysis."""
        features = [
            FeatureImportance("f1", 0.5, 0.5, 50.0, 1),
            FeatureImportance("f2", 0.3, 0.3, 30.0, 2),
        ]

        analysis = PerformanceDegradationAnalysis(
            timestamp=time.time(),
            metric_name="accuracy",
            baseline_value=0.9,
            current_value=0.75,
            degradation_percent=16.67,
            top_contributing_features=features,
            error_patterns={"n_errors": 50},
            failure_cohort_size=50,
            recommended_actions=["Retrain", "Check data"],
        )

        assert analysis.metric_name == "accuracy"
        assert analysis.degradation_percent == pytest.approx(16.67)
        assert len(analysis.top_contributing_features) == 2

    def test_degradation_analysis_to_dict(self):
        """Test converting to dict."""
        features = [FeatureImportance("f1", 0.5, 0.5, 50.0, 1)]

        analysis = PerformanceDegradationAnalysis(
            timestamp=time.time(),
            metric_name="precision",
            baseline_value=0.85,
            current_value=0.70,
            degradation_percent=17.65,
            top_contributing_features=features,
            error_patterns={},
            failure_cohort_size=10,
            recommended_actions=["Action1"],
        )

        result = analysis.to_dict()
        assert result["metric_name"] == "precision"
        assert isinstance(result["top_contributing_features"], list)
        assert len(result["top_contributing_features"]) == 1


class TestShapExplainabilityMonitor:
    """Test ShapExplainabilityMonitor."""

    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.data_dir.exists()
        assert monitor.shap_explainer is None

    def test_explain_drift(self, monitor, sample_data):
        """Test drift explanation."""
        baseline, current = sample_data

        shap_values = np.random.randn(50, 3)

        explanation = monitor.explain_drift(
            feature_name="feature_1",
            baseline_data=baseline,
            current_data=current,
            predictions=np.random.randn(50),
            shap_values=shap_values,
        )

        assert explanation.feature_name == "feature_1"
        assert explanation.drift_type in ["shift", "scale", "distribution"]
        assert 0.0 <= explanation.confidence <= 1.0

    def test_explain_drift_classification(self, monitor):
        """Test drift type classification."""
        baseline = pd.DataFrame({"feature": [100] * 100})
        current = pd.DataFrame({"feature": [105] * 50})

        explanation = monitor.explain_drift(
            feature_name="feature",
            baseline_data=baseline,
            current_data=current,
            predictions=np.zeros(50),
            shap_values=np.zeros((50, 1)),
        )

        assert explanation.drift_type in ["shift", "scale", "distribution"]

    def test_analyze_performance_degradation(self, monitor, sample_data):
        """Test performance degradation analysis."""
        baseline, current = sample_data

        predictions = current.copy()
        shap_values = np.random.randn(50, 3)
        errors = np.array([True] * 10 + [False] * 40)

        analysis = monitor.analyze_performance_degradation(
            metric_name="accuracy",
            baseline_value=0.90,
            current_value=0.75,
            predictions=predictions,
            shap_values=shap_values,
            errors=errors,
        )

        assert analysis.metric_name == "accuracy"
        assert analysis.degradation_percent > 0
        assert analysis.failure_cohort_size == 10
        assert len(analysis.recommended_actions) > 0

    def test_track_feature_importance(self, monitor):
        """Test feature importance tracking."""
        importances = {
            "feature_1": 0.5,
            "feature_2": 0.3,
            "feature_3": 0.2,
        }

        monitor.track_feature_importance(importances, "v1.0")

        # Verify history was updated
        assert "feature_1" in monitor.feature_importance_history

    def test_get_feature_importance_trend(self, monitor):
        """Test getting feature importance trend."""
        importances = {
            "feature_1": 0.5,
            "feature_2": 0.3,
        }

        monitor.track_feature_importance(importances, "v1.0")
        trend = monitor.get_feature_importance_trend("feature_1")

        assert isinstance(trend, list)

    def test_get_degradation_analyses(self, monitor, sample_data):
        """Test retrieving degradation analyses."""
        baseline, current = sample_data

        # Create first analysis
        monitor.analyze_performance_degradation(
            metric_name="accuracy",
            baseline_value=0.90,
            current_value=0.75,
            predictions=current,
            shap_values=np.random.randn(50, 3),
            errors=None,
        )

        # Get all analyses
        analyses = monitor.get_degradation_analyses()
        assert len(analyses) >= 1

    def test_persistence_baselines(self, monitor):
        """Test baseline file persistence."""
        importances = {"f1": 0.5, "f2": 0.3}
        monitor.track_feature_importance(importances, "v1.0")

        # Verify file exists
        assert monitor.feature_importance_file.exists()

    def test_empty_shap_values_handling(self, monitor):
        """Test handling of empty SHAP values."""
        baseline = pd.DataFrame({"f": [1] * 10})
        current = pd.DataFrame({"f": [1] * 5})

        explanation = monitor.explain_drift(
            feature_name="f",
            baseline_data=baseline,
            current_data=current,
            predictions=np.zeros(5),
            shap_values=np.array([]),
        )

        assert explanation.shap_contribution == 0.0

    def test_drift_recommendation_generation(self, monitor):
        """Test recommendation generation for drift."""
        baseline = pd.DataFrame({"f": [100] * 100})
        current = pd.DataFrame({"f": [110] * 50})

        explanation = monitor.explain_drift(
            feature_name="f",
            baseline_data=baseline,
            current_data=current,
            predictions=np.zeros(50),
            shap_values=np.ones((50, 1)),
        )

        assert len(explanation.recommendation) > 0
        assert isinstance(explanation.recommendation, str)


class TestExplanabilityAlertEnricher:
    """Test ExplanabilityAlertEnricher."""

    @pytest.fixture
    def enricher(self):
        """Create enricher instance."""
        return ExplanabilityAlertEnricher()

    @pytest.fixture
    def sample_alert(self):
        """Create sample alert."""
        return Alert(
            timestamp=time.time(),
            alert_id="test_1",
            severity="warning",
            title="Test Alert",
            message="This is a test alert",
            metric_name="accuracy",
            metric_value=0.75,
            threshold=0.85,
            component="performance",
            model_version="v1.0",
        )

    def test_enrich_alert_basic(self, enricher, sample_alert):
        """Test basic alert enrichment."""
        enriched = enricher.enrich_alert(sample_alert)

        assert "alert" in enriched
        assert "explanation" in enriched
        assert enriched["alert"]["alert_id"] == "test_1"

    def test_enrich_alert_with_drift(self, enricher, sample_alert):
        """Test enrichment with drift explanation."""
        drift = DriftExplanation(
            feature_name="accuracy",
            drift_type="shift",
            baseline_mean=0.90,
            current_mean=0.75,
            baseline_std=0.05,
            current_std=0.06,
            shap_contribution=0.15,
            contributing_features=["f1"],
            confidence=0.95,
            recommendation="Retrain",
        )

        enriched = enricher.enrich_alert(sample_alert, drift_explanation=drift)

        assert "drift" in enriched["explanation"]
        assert enriched["explanation"]["drift"]["feature"] == "accuracy"

    def test_format_email_with_explanation(self, enricher, sample_alert):
        """Test email formatting with explanation."""
        drift = DriftExplanation(
            feature_name="accuracy",
            drift_type="shift",
            baseline_mean=0.90,
            current_mean=0.75,
            baseline_std=0.05,
            current_std=0.06,
            shap_contribution=0.15,
            contributing_features=["f1"],
            confidence=0.95,
            recommendation="Retrain",
        )

        html = enricher.format_email_with_explanation(
            sample_alert,
            drift_explanation=drift,
        )

        assert "<html>" in html
        assert sample_alert.title in html
        assert "Drift Explanation" in html

    def test_format_slack_with_explanation(self, enricher, sample_alert):
        """Test Slack formatting with explanation."""
        drift = DriftExplanation(
            feature_name="accuracy",
            drift_type="shift",
            baseline_mean=0.90,
            current_mean=0.75,
            baseline_std=0.05,
            current_std=0.06,
            shap_contribution=0.15,
            contributing_features=["f1"],
            confidence=0.95,
            recommendation="Retrain",
        )

        blocks = enricher.format_slack_with_explanation(
            sample_alert,
            drift_explanation=drift,
        )

        assert isinstance(blocks, list)
        assert len(blocks) > 0
        assert any("Drift Explanation" in str(b) for b in blocks)

    def test_format_with_degradation(self, enricher, sample_alert):
        """Test formatting with degradation analysis."""
        analysis = PerformanceDegradationAnalysis(
            timestamp=time.time(),
            metric_name="accuracy",
            baseline_value=0.90,
            current_value=0.75,
            degradation_percent=16.67,
            top_contributing_features=[
                FeatureImportance("f1", 0.5, 0.5, 50.0, 1),
            ],
            error_patterns={"n_errors": 25},
            failure_cohort_size=25,
            recommended_actions=["Retrain", "Check data quality"],
        )

        html = enricher.format_email_with_explanation(
            sample_alert,
            degradation_analysis=analysis,
        )

        assert "Performance Analysis" in html
        assert "16.67" in html

    def test_slack_with_degradation(self, enricher, sample_alert):
        """Test Slack with degradation analysis."""
        analysis = PerformanceDegradationAnalysis(
            timestamp=time.time(),
            metric_name="accuracy",
            baseline_value=0.90,
            current_value=0.75,
            degradation_percent=16.67,
            top_contributing_features=[
                FeatureImportance("f1", 0.5, 0.5, 50.0, 1),
            ],
            error_patterns={},
            failure_cohort_size=25,
            recommended_actions=["Retrain"],
        )

        blocks = enricher.format_slack_with_explanation(
            sample_alert,
            degradation_analysis=analysis,
        )

        assert isinstance(blocks, list)
        assert any("Performance Analysis" in str(b) for b in blocks)
