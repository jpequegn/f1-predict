"""Tests for monitoring explainability dashboard utilities."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from f1_predict.web.utils.monitoring_explainability import (
    DriftExplanation,
    FeatureImportance,
    PerformanceDegradationAnalysis,
    ShapExplainabilityMonitor,
)
from f1_predict.web.utils.monitoring_explainability_dashboard import (
    ExplainabilityChartBuilders,
    ExplainabilityDataLoaders,
    ExplainabilityTableFormatters,
)


class TestExplainabilityChartBuilders:
    """Test chart builder utilities."""

    def test_create_feature_importance_heatmap_with_data(self):
        """Test creating feature importance heatmap with data."""
        importance_history = {
            "feature_1": [
                FeatureImportance("feature_1", 0.8, 0.8, 40.0, 1),
                FeatureImportance("feature_1", 0.75, 0.75, 37.5, 1),
            ],
            "feature_2": [
                FeatureImportance("feature_2", 0.6, 0.6, 30.0, 2),
                FeatureImportance("feature_2", 0.65, 0.65, 32.5, 2),
            ],
        }

        fig = ExplainabilityChartBuilders.create_feature_importance_heatmap(
            importance_history, limit=2
        )

        assert fig is not None
        assert fig.data is not None

    def test_create_feature_importance_heatmap_empty_data(self):
        """Test creating heatmap with empty data."""
        importance_history = {}

        fig = ExplainabilityChartBuilders.create_feature_importance_heatmap(
            importance_history
        )

        assert fig is not None
        # Should return a figure with annotation for empty data

    def test_create_drift_explanation_chart(self):
        """Test creating drift explanation chart."""
        drift = DriftExplanation(
            feature_name="test_feature",
            drift_type="shift",
            baseline_mean=10.0,
            current_mean=12.5,
            baseline_std=2.0,
            current_std=2.5,
            shap_contribution=0.5,
            contributing_features=["f1", "f2"],
            confidence=0.95,
            recommendation="Retrain model",
        )

        fig = ExplainabilityChartBuilders.create_drift_explanation_chart(drift)

        assert fig is not None
        assert len(fig.data) > 0

    def test_create_degradation_analysis_chart(self):
        """Test creating degradation analysis chart."""
        analysis = PerformanceDegradationAnalysis(
            timestamp=1234567890.0,
            metric_name="accuracy",
            baseline_value=0.95,
            current_value=0.88,
            degradation_percent=7.37,
            top_contributing_features=[
                FeatureImportance("f1", 0.6, 0.6, 50.0, 1),
                FeatureImportance("f2", 0.4, 0.4, 33.0, 2),
            ],
            error_patterns={"n_errors": 50},
            failure_cohort_size=50,
            recommended_actions=["Check data quality"],
        )

        fig = ExplainabilityChartBuilders.create_degradation_analysis_chart(analysis)

        assert fig is not None
        assert len(fig.data) > 0

    def test_create_top_features_chart(self):
        """Test creating top features chart."""
        features = [
            FeatureImportance("feature_1", 0.8, 0.8, 40.0, 1),
            FeatureImportance("feature_2", 0.6, 0.6, 30.0, 2),
            FeatureImportance("feature_3", 0.4, 0.4, 20.0, 3),
        ]

        fig = ExplainabilityChartBuilders.create_top_features_chart(features)

        assert fig is not None
        assert len(fig.data) > 0

    def test_create_top_features_chart_empty(self):
        """Test creating top features chart with empty list."""
        features = []

        fig = ExplainabilityChartBuilders.create_top_features_chart(features)

        assert fig is not None
        # Should return a figure with annotation for empty data


class TestExplainabilityTableFormatters:
    """Test table formatting utilities."""

    def test_format_drift_explanation(self):
        """Test formatting drift explanation as DataFrame."""
        drift = DriftExplanation(
            feature_name="test_feature",
            drift_type="shift",
            baseline_mean=10.0,
            current_mean=12.5,
            baseline_std=2.0,
            current_std=2.5,
            shap_contribution=0.5,
            contributing_features=["f1", "f2"],
            confidence=0.95,
            recommendation="Retrain model",
        )

        df = ExplainabilityTableFormatters.format_drift_explanation(drift)

        assert isinstance(df, pd.DataFrame)
        assert "Metric" in df.columns
        assert "Value" in df.columns
        assert len(df) == 9

    def test_format_degradation_analysis(self):
        """Test formatting degradation analysis as DataFrame."""
        analysis = PerformanceDegradationAnalysis(
            timestamp=1234567890.0,
            metric_name="accuracy",
            baseline_value=0.95,
            current_value=0.88,
            degradation_percent=7.37,
            top_contributing_features=[
                FeatureImportance("f1", 0.6, 0.6, 50.0, 1),
                FeatureImportance("f2", 0.4, 0.4, 33.0, 2),
            ],
            error_patterns={"n_errors": 50},
            failure_cohort_size=50,
            recommended_actions=["Check data quality"],
        )

        df = ExplainabilityTableFormatters.format_degradation_analysis(analysis)

        assert isinstance(df, pd.DataFrame)
        assert "Metric" in df.columns
        assert "Value" in df.columns
        assert len(df) == 6

    def test_format_feature_importance_table(self):
        """Test formatting feature importance list as DataFrame."""
        features = [
            FeatureImportance("feature_1", 0.8, 0.8, 40.0, 1),
            FeatureImportance("feature_2", 0.6, 0.6, 30.0, 2),
            FeatureImportance("feature_3", 0.4, 0.4, 20.0, 3),
        ]

        df = ExplainabilityTableFormatters.format_feature_importance_table(features)

        assert isinstance(df, pd.DataFrame)
        assert "Rank" in df.columns
        assert "Feature" in df.columns
        assert "Importance Score" in df.columns
        assert len(df) == 3


class TestExplainabilityDataLoaders:
    """Test data loading utilities."""

    def test_load_recent_drift_explanations(self):
        """Test loading recent drift explanations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = ShapExplainabilityMonitor(data_dir=temp_dir)

            drift = DriftExplanation(
                feature_name="test_feature",
                drift_type="shift",
                baseline_mean=10.0,
                current_mean=12.5,
                baseline_std=2.0,
                current_std=2.5,
                shap_contribution=0.5,
                contributing_features=["f1", "f2"],
                confidence=0.95,
                recommendation="Retrain model",
            )

            # Save explanation
            monitor._save_explanation(drift)

            # Load explanations
            explanations = ExplainabilityDataLoaders.load_recent_drift_explanations(
                monitor, limit=5
            )

            assert len(explanations) == 1
            assert explanations[0].feature_name == "test_feature"

    def test_load_recent_drift_explanations_empty(self):
        """Test loading drift explanations with no data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = ShapExplainabilityMonitor(data_dir=temp_dir)

            explanations = ExplainabilityDataLoaders.load_recent_drift_explanations(
                monitor, limit=5
            )

            assert len(explanations) == 0

    def test_get_feature_importance_trends(self):
        """Test getting feature importance trends."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = ShapExplainabilityMonitor(data_dir=temp_dir)

            # Track some feature importance
            monitor.track_feature_importance(
                {"feature_1": 0.8, "feature_2": 0.6}, model_version="v1"
            )
            monitor.track_feature_importance(
                {"feature_1": 0.75, "feature_2": 0.65}, model_version="v1"
            )

            trends = ExplainabilityDataLoaders.get_feature_importance_trends(monitor)

            assert isinstance(trends, dict)

    def test_get_feature_importance_trends_specific_feature(self):
        """Test getting trends for specific feature."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = ShapExplainabilityMonitor(data_dir=temp_dir)

            monitor.track_feature_importance(
                {"feature_1": 0.8, "feature_2": 0.6}, model_version="v1"
            )

            trends = ExplainabilityDataLoaders.get_feature_importance_trends(
                monitor, feature_name="feature_1"
            )

            assert isinstance(trends, dict)
            if trends:
                assert "feature_1" in trends


class TestIntegration:
    """Integration tests for dashboard components."""

    def test_end_to_end_drift_workflow(self):
        """Test complete drift explanation and visualization workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = ShapExplainabilityMonitor(data_dir=temp_dir)

            # Create baseline data
            baseline = pd.DataFrame({"feature": [10.0, 10.5, 11.0]})
            current = pd.DataFrame({"feature": [12.0, 12.5, 13.0]})

            # Simulate SHAP values
            import numpy as np

            shap_values = np.array([0.4, 0.45, 0.5])

            # Explain drift
            drift = monitor.explain_drift(
                feature_name="feature",
                baseline_data=baseline,
                current_data=current,
                predictions=np.array([1, 1, 1]),
                shap_values=shap_values,
            )

            # Create visualizations
            chart = ExplainabilityChartBuilders.create_drift_explanation_chart(drift)
            table = ExplainabilityTableFormatters.format_drift_explanation(drift)

            assert chart is not None
            assert isinstance(table, pd.DataFrame)

    def test_end_to_end_degradation_workflow(self):
        """Test complete degradation analysis and visualization workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            import numpy as np

            monitor = ShapExplainabilityMonitor(data_dir=temp_dir)

            predictions = pd.DataFrame({"feature_1": [1.0, 2.0, 3.0]})
            shap_values = np.array([[0.4, 0.3], [0.45, 0.35], [0.5, 0.3]])

            # Analyze degradation
            analysis = monitor.analyze_performance_degradation(
                metric_name="accuracy",
                baseline_value=0.95,
                current_value=0.88,
                predictions=predictions,
                shap_values=shap_values,
                errors=np.array([False, False, True]),
            )

            # Create visualizations
            chart = ExplainabilityChartBuilders.create_degradation_analysis_chart(
                analysis
            )
            table = ExplainabilityTableFormatters.format_degradation_analysis(analysis)
            features_chart = ExplainabilityChartBuilders.create_top_features_chart(
                analysis.top_contributing_features
            )

            assert chart is not None
            assert isinstance(table, pd.DataFrame)
            assert features_chart is not None
