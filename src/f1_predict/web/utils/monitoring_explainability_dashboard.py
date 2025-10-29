"""Explainability dashboard utilities for monitoring system integration.

Provides Streamlit components for visualizing SHAP-based explanations, drift analysis,
and performance degradation insights in the monitoring dashboard.
"""

import json
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import structlog

from f1_predict.web.utils.monitoring_explainability import (
    DriftExplanation,
    FeatureImportance,
    PerformanceDegradationAnalysis,
    ShapExplainabilityMonitor,
)

logger = structlog.get_logger(__name__)


class ExplainabilityChartBuilders:
    """Build interactive charts for explainability visualization."""

    @staticmethod
    def create_feature_importance_heatmap(
        importance_history: dict[str, list[FeatureImportance]], limit: int = 10
    ) -> go.Figure:
        """Create feature importance heatmap over time.

        Args:
            importance_history: Historical feature importance data
            limit: Maximum features to display

        Returns:
            Plotly figure with feature importance heatmap
        """
        try:
            # Prepare data for heatmap
            features = list(importance_history.keys())[:limit]
            time_points = []
            data_matrix = []

            if not features:
                return go.Figure().add_annotation(
                    text="No feature importance data available",
                    showarrow=False,
                )

            # Get time points from first feature
            if features and importance_history[features[0]]:
                time_points = [f"Time {i+1}" for i in range(len(importance_history[features[0]]))]

            # Build data matrix
            for feature in features:
                importance_scores = [
                    fi.importance_score for fi in importance_history.get(feature, [])
                ]
                # Pad if necessary
                while len(importance_scores) < len(time_points):
                    importance_scores.append(0.0)
                data_matrix.append(importance_scores[: len(time_points)])

            fig = go.Figure(
                data=go.Heatmap(
                    z=data_matrix,
                    x=time_points,
                    y=features,
                    colorscale="Viridis",
                    colorbar={"title": "Importance"},
                )
            )

            fig.update_layout(
                title="Feature Importance Over Time",
                xaxis_title="Time",
                yaxis_title="Features",
                height=500,
                hovermode="closest",
            )

            return fig

        except Exception as e:
            logger.error("feature_importance_heatmap_failed", error=str(e))
            return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)

    @staticmethod
    def create_drift_explanation_chart(drift_explanation: DriftExplanation) -> go.Figure:
        """Create visualization for drift explanation.

        Args:
            drift_explanation: DriftExplanation object

        Returns:
            Plotly figure showing drift details
        """
        try:
            fig = go.Figure()

            # Baseline vs current mean
            fig.add_trace(
                go.Bar(
                    x=["Baseline", "Current"],
                    y=[drift_explanation.baseline_mean, drift_explanation.current_mean],
                    marker={"color": ["#1F4E8C", "#FFC107"]},
                    text=[
                        f"{drift_explanation.baseline_mean:.4f}",
                        f"{drift_explanation.current_mean:.4f}",
                    ],
                    textposition="auto",
                )
            )

            fig.update_layout(
                title=f"Drift Detection: {drift_explanation.feature_name}",
                yaxis_title="Mean Value",
                height=400,
                showlegend=False,
            )

            return fig

        except Exception as e:
            logger.error("drift_explanation_chart_failed", error=str(e))
            return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)

    @staticmethod
    def create_degradation_analysis_chart(
        analysis: PerformanceDegradationAnalysis,
    ) -> go.Figure:
        """Create visualization for performance degradation analysis.

        Args:
            analysis: PerformanceDegradationAnalysis object

        Returns:
            Plotly figure showing degradation details
        """
        try:
            fig = go.Figure()

            # Baseline vs current metric
            fig.add_trace(
                go.Bar(
                    x=["Baseline", "Current"],
                    y=[analysis.baseline_value, analysis.current_value],
                    marker={"color": ["#28A745", "#DC3545"]},
                    text=[
                        f"{analysis.baseline_value:.4f}",
                        f"{analysis.current_value:.4f}",
                    ],
                    textposition="auto",
                )
            )

            fig.update_layout(
                title=f"Performance Degradation: {analysis.metric_name}",
                yaxis_title="Metric Value",
                height=400,
                showlegend=False,
            )

            return fig

        except Exception as e:
            logger.error("degradation_analysis_chart_failed", error=str(e))
            return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)

    @staticmethod
    def create_top_features_chart(top_features: list[FeatureImportance]) -> go.Figure:
        """Create bar chart for top contributing features.

        Args:
            top_features: List of FeatureImportance objects

        Returns:
            Plotly figure showing top features
        """
        try:
            if not top_features:
                return go.Figure().add_annotation(
                    text="No feature importance data available",
                    showarrow=False,
                )

            df = pd.DataFrame(
                {
                    "Feature": [f.feature_name for f in top_features],
                    "Importance": [f.importance_score for f in top_features],
                    "Percentage": [f.percentage for f in top_features],
                }
            )

            fig = px.bar(
                df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Contributing Features",
                labels={"Importance": "SHAP Importance Score", "Feature": "Feature Name"},
                color="Importance",
                color_continuous_scale="Viridis",
            )

            fig.update_layout(height=400, showlegend=False)

            return fig

        except Exception as e:
            logger.error("top_features_chart_failed", error=str(e))
            return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)


class ExplainabilityTableFormatters:
    """Format data for tabular display in Streamlit."""

    @staticmethod
    def format_drift_explanation(drift: DriftExplanation) -> pd.DataFrame:
        """Format drift explanation as DataFrame for display.

        Args:
            drift: DriftExplanation object

        Returns:
            Formatted DataFrame
        """
        return pd.DataFrame(
            {
                "Metric": [
                    "Feature Name",
                    "Drift Type",
                    "Baseline Mean",
                    "Current Mean",
                    "Baseline Std",
                    "Current Std",
                    "SHAP Contribution",
                    "Confidence",
                    "Recommendation",
                ],
                "Value": [
                    drift.feature_name,
                    drift.drift_type.upper(),
                    f"{drift.baseline_mean:.4f}",
                    f"{drift.current_mean:.4f}",
                    f"{drift.baseline_std:.4f}",
                    f"{drift.current_std:.4f}",
                    f"{drift.shap_contribution:.4f}",
                    f"{drift.confidence:.1%}",
                    drift.recommendation,
                ],
            }
        )

    @staticmethod
    def format_degradation_analysis(
        analysis: PerformanceDegradationAnalysis,
    ) -> pd.DataFrame:
        """Format degradation analysis as DataFrame for display.

        Args:
            analysis: PerformanceDegradationAnalysis object

        Returns:
            Formatted DataFrame
        """
        return pd.DataFrame(
            {
                "Metric": [
                    "Metric Name",
                    "Baseline Value",
                    "Current Value",
                    "Degradation",
                    "Failure Count",
                    "Top Contributing Features",
                ],
                "Value": [
                    analysis.metric_name,
                    f"{analysis.baseline_value:.4f}",
                    f"{analysis.current_value:.4f}",
                    f"{analysis.degradation_percent:.2f}%",
                    str(analysis.failure_cohort_size),
                    ", ".join([f.feature_name for f in analysis.top_contributing_features[:3]]),
                ],
            }
        )

    @staticmethod
    def format_feature_importance_table(
        importance_list: list[FeatureImportance],
    ) -> pd.DataFrame:
        """Format feature importance list as DataFrame for display.

        Args:
            importance_list: List of FeatureImportance objects

        Returns:
            Formatted DataFrame
        """
        return pd.DataFrame(
            {
                "Rank": [fi.rank for fi in importance_list],
                "Feature": [fi.feature_name for fi in importance_list],
                "Importance Score": [f"{fi.importance_score:.4f}" for fi in importance_list],
                "SHAP Value": [f"{fi.shap_value:.4f}" for fi in importance_list],
                "Percentage": [f"{fi.percentage:.2f}%" for fi in importance_list],
            }
        )


class ExplainabilityDataLoaders:
    """Load and prepare explainability data for visualization."""

    @staticmethod
    def load_recent_drift_explanations(
        monitor: ShapExplainabilityMonitor, limit: int = 5
    ) -> list[DriftExplanation]:
        """Load recent drift explanations.

        Args:
            monitor: ShapExplainabilityMonitor instance
            limit: Maximum explanations to load

        Returns:
            List of DriftExplanation objects
        """
        try:
            # Try to load from file
            explanations = []
            if monitor.explanations_file.exists():
                with open(monitor.explanations_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            drift = DriftExplanation(**data)
                            explanations.append(drift)

            return explanations[-limit:]

        except Exception as e:
            logger.error("load_drift_explanations_failed", error=str(e))
            return []

    @staticmethod
    def get_feature_importance_trends(
        monitor: ShapExplainabilityMonitor, feature_name: Optional[str] = None
    ) -> dict[str, Any]:
        """Get feature importance trend data.

        Args:
            monitor: ShapExplainabilityMonitor instance
            feature_name: Optional specific feature to get

        Returns:
            Dictionary with trend data
        """
        try:
            trends = {}

            if feature_name:
                trend_data = monitor.get_feature_importance_trend(feature_name)
                trends[feature_name] = trend_data
            else:
                for feat in monitor.feature_importance_history:
                    trend_data = monitor.get_feature_importance_trend(feat)
                    if trend_data:
                        trends[feat] = trend_data

            return trends

        except Exception as e:
            logger.error("get_feature_importance_trends_failed", error=str(e))
            return {}
