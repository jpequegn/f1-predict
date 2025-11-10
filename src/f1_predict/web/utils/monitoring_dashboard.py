"""Monitoring dashboard utilities and components for Streamlit visualization.

Provides reusable functions for displaying monitoring data, performance metrics,
drift detection results, and alert management in the Streamlit dashboard.
"""

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import structlog

logger = structlog.get_logger(__name__)


class DashboardMetrics:
    """Utility class for dashboard KPI calculations."""

    @staticmethod
    def calculate_performance_delta(
        current: float, previous: float, metric_name: str = "Metric"
    ) -> dict[str, Any]:
        """Calculate delta for metric display.

        Args:
            current: Current metric value
            previous: Previous metric value
            metric_name: Metric name for context

        Returns:
            Dictionary with delta and delta_color
        """
        delta = current - previous
        delta_color = "off" if delta == 0 else ("inverse" if delta < 0 else "normal")

        return {
            "delta": delta,
            "delta_color": delta_color,
            "delta_pct": (delta / previous * 100) if previous != 0 else 0,
        }

    @staticmethod
    def get_health_status_color(accuracy: float) -> str:
        """Get color for health status based on accuracy.

        Args:
            accuracy: Model accuracy (0-1)

        Returns:
            Color string (green/yellow/red)
        """
        if accuracy >= 0.85:
            return "🟢"
        if accuracy >= 0.75:
            return "🟡"
        return "🔴"

    @staticmethod
    def get_alert_severity_color(severity: str) -> str:
        """Get color for alert severity.

        Args:
            severity: Alert severity (INFO, WARNING, CRITICAL)

        Returns:
            Hex color string
        """
        colors = {
            "CRITICAL": "#DC3545",
            "WARNING": "#FFC107",
            "INFO": "#28A745",
        }
        return colors.get(severity, "#A3A9BF")

    @staticmethod
    def get_drift_severity_badge(p_value: float) -> str:
        """Get badge for drift test p-value.

        Args:
            p_value: Statistical p-value

        Returns:
            Severity badge emoji
        """
        if p_value < 0.01:
            return "🔴 HIGH"
        if p_value < 0.05:
            return "🟡 MEDIUM"
        return "🟢 LOW"


class ChartBuilders:
    """Utility class for creating monitoring charts."""

    @staticmethod
    def create_accuracy_trend_chart(
        performance_data: pd.DataFrame, height: int = 400
    ) -> go.Figure:
        """Create accuracy trend line chart.

        Args:
            performance_data: DataFrame with timestamp and accuracy columns
            height: Chart height in pixels

        Returns:
            Plotly figure
        """
        fig = px.line(
            performance_data,
            x="timestamp",
            y="accuracy",
            title="Accuracy Trend Over Time",
            labels={"accuracy": "Accuracy", "timestamp": "Time"},
            template="plotly_dark",
            height=height,
        )

        fig.update_traces(line={"color": "#1F4E8C", "width": 2})
        fig.update_layout(
            hovermode="x unified",
            plot_bgcolor="#1E2130",
            paper_bgcolor="#121317",
            font={"color": "#E0E6F0"},
        )

        return fig

    @staticmethod
    def create_confidence_distribution_chart(
        confidences: list[float], height: int = 400
    ) -> go.Figure:
        """Create confidence distribution histogram.

        Args:
            confidences: List of prediction confidences
            height: Chart height

        Returns:
            Plotly figure
        """
        fig = px.histogram(
            x=confidences,
            nbins=20,
            title="Prediction Confidence Distribution",
            labels={"x": "Confidence", "count": "Frequency"},
            template="plotly_dark",
            height=height,
        )

        fig.update_traces(marker={"color": "#2762B3"})
        fig.update_layout(
            plot_bgcolor="#1E2130",
            paper_bgcolor="#121317",
            font={"color": "#E0E6F0"},
        )

        return fig

    @staticmethod
    def create_model_health_scorecard(health_data: dict[str, float]) -> None:
        """Display model health scorecard with metrics.

        Args:
            health_data: Dictionary with accuracy, precision, recall, f1
        """
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Accuracy",
                f"{health_data.get('accuracy', 0):.3f}",
                delta=f"{health_data.get('accuracy_delta', 0):+.4f}",
            )

        with col2:
            st.metric(
                "Precision",
                f"{health_data.get('precision', 0):.3f}",
                delta=f"{health_data.get('precision_delta', 0):+.4f}",
            )

        with col3:
            st.metric(
                "Recall",
                f"{health_data.get('recall', 0):.3f}",
                delta=f"{health_data.get('recall_delta', 0):+.4f}",
            )

        with col4:
            st.metric(
                "F1 Score",
                f"{health_data.get('f1_score', 0):.3f}",
                delta=f"{health_data.get('f1_delta', 0):+.4f}",
            )

    @staticmethod
    def create_drift_heatmap(
        drift_results: pd.DataFrame, height: int = 400
    ) -> go.Figure:
        """Create drift status heatmap.

        Args:
            drift_results: DataFrame with features and drift metrics
            height: Chart height

        Returns:
            Plotly heatmap
        """
        if drift_results.empty:
            fig = go.Figure()
            fig.add_annotation(text="No drift data available")
            return fig

        # Prepare data for heatmap
        features = drift_results["feature_name"].tolist()
        psi_values = drift_results["psi"].tolist()

        fig = go.Figure(
            data=go.Heatmap(
                z=[psi_values],
                x=features,
                colorscale="RdYlGn_r",
                colorbar={"title": "PSI"},
                hovertemplate="<b>%{x}</b><br>PSI: %{z:.3f}",
            )
        )

        fig.update_layout(
            title="Feature Drift Status (PSI)",
            height=height,
            plot_bgcolor="#1E2130",
            paper_bgcolor="#121317",
            font={"color": "#E0E6F0"},
        )

        return fig

    @staticmethod
    def create_psi_trend_chart(
        psi_history: pd.DataFrame, height: int = 400
    ) -> go.Figure:
        """Create PSI trend line chart.

        Args:
            psi_history: DataFrame with timestamp and psi columns
            height: Chart height

        Returns:
            Plotly figure
        """
        fig = px.line(
            psi_history,
            x="timestamp",
            y="psi",
            color="feature_name",
            title="PSI Trends Over Time",
            labels={"psi": "PSI Value", "timestamp": "Time"},
            template="plotly_dark",
            height=height,
        )

        fig.update_layout(
            hovermode="x unified",
            plot_bgcolor="#1E2130",
            paper_bgcolor="#121317",
            font={"color": "#E0E6F0"},
        )

        return fig

    @staticmethod
    def create_alert_timeline(alerts_df: pd.DataFrame, height: int = 300) -> go.Figure:
        """Create alert timeline visualization.

        Args:
            alerts_df: DataFrame with alerts
            height: Chart height

        Returns:
            Plotly figure
        """
        fig = px.scatter(
            alerts_df,
            x="timestamp",
            y="severity",
            color="severity",
            title="Alert Timeline",
            labels={"timestamp": "Time", "severity": "Severity"},
            template="plotly_dark",
            height=height,
            hover_data=["metric_name", "message"],
        )

        fig.update_layout(
            plot_bgcolor="#1E2130",
            paper_bgcolor="#121317",
            font={"color": "#E0E6F0"},
        )

        return fig

    @staticmethod
    def create_model_comparison_chart(
        model_metrics: pd.DataFrame, height: int = 400
    ) -> go.Figure:
        """Create side-by-side model comparison chart.

        Args:
            model_metrics: DataFrame with model names and metrics
            height: Chart height

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            fig.add_trace(
                go.Bar(
                    x=model_metrics["model_name"],
                    y=model_metrics[metric],
                    name=metric.replace("_", " ").title(),
                    hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:.3f}<extra></extra>",
                )
            )

        fig.update_layout(
            title="Model Performance Comparison",
            height=height,
            barmode="group",
            plot_bgcolor="#1E2130",
            paper_bgcolor="#121317",
            font={"color": "#E0E6F0"},
            xaxis_title="Model",
            yaxis_title="Score",
        )

        return fig


class TableFormatters:
    """Utility class for formatting monitoring tables."""

    @staticmethod
    def format_drift_results_table(drift_results: pd.DataFrame) -> pd.DataFrame:
        """Format drift detection results for display.

        Args:
            drift_results: Raw drift results DataFrame

        Returns:
            Formatted DataFrame
        """
        if drift_results.empty:
            return drift_results

        formatted = drift_results.copy()

        # Format numeric columns
        for col in ["psi", "ks_statistic", "p_value"]:
            if col in formatted.columns:
                formatted[col] = formatted[col].apply(lambda x: f"{x:.4f}")

        # Add severity indicator
        if "p_value" in formatted.columns:
            formatted["severity"] = formatted["p_value"].apply(
                lambda p: ChartBuilders.create_alert_timeline.__doc__
            )

        return formatted

    @staticmethod
    def format_alert_table(alerts_df: pd.DataFrame) -> pd.DataFrame:
        """Format alerts for display.

        Args:
            alerts_df: Raw alerts DataFrame

        Returns:
            Formatted DataFrame
        """
        if alerts_df.empty:
            return alerts_df

        formatted = alerts_df.copy()

        # Format timestamp
        if "timestamp" in formatted.columns:
            formatted["timestamp"] = pd.to_datetime(formatted["timestamp"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        # Add status indicator
        if "acknowledged" in formatted.columns:
            formatted["status"] = formatted["acknowledged"].apply(
                lambda x: "✓ Acknowledged" if x else "⚠️ Pending"
            )

        return formatted

    @staticmethod
    def format_health_snapshot_table(health_df: pd.DataFrame) -> pd.DataFrame:
        """Format health snapshots for display.

        Args:
            health_df: Raw health snapshots DataFrame

        Returns:
            Formatted DataFrame
        """
        if health_df.empty:
            return health_df

        formatted = health_df.copy()

        # Format timestamp
        if "timestamp" in formatted.columns:
            formatted["timestamp"] = pd.to_datetime(formatted["timestamp"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        # Format metrics
        for col in ["accuracy", "precision", "recall", "f1_score"]:
            if col in formatted.columns:
                formatted[col] = formatted[col].apply(lambda x: f"{x:.3f}")

        return formatted


class DataLoaders:
    """Utility class for loading monitoring data for dashboard display."""

    @staticmethod
    def load_performance_trend(
        performance_tracker: Any, hours: int = 24
    ) -> pd.DataFrame:
        """Load performance trend data.

        Args:
            performance_tracker: ModelPerformanceTracker instance
            hours: Hours of history to load

        Returns:
            DataFrame with performance trend
        """
        try:
            trend = performance_tracker.get_performance_trend(window_minutes=hours * 60)
            if not trend:
                return pd.DataFrame()

            return pd.DataFrame(trend)
        except Exception as e:
            logger.error("performance_trend_load_failed", error=str(e))
            return pd.DataFrame()

    @staticmethod
    def load_health_snapshots(
        performance_tracker: Any, limit: int = 100
    ) -> pd.DataFrame:
        """Load recent health snapshots.

        Args:
            performance_tracker: ModelPerformanceTracker instance
            limit: Maximum number of snapshots

        Returns:
            DataFrame with health snapshots
        """
        try:
            snapshots = performance_tracker.get_recent_health_snapshots(limit=limit)
            if not snapshots:
                return pd.DataFrame()

            return pd.DataFrame(snapshots)
        except Exception as e:
            logger.error("health_snapshots_load_failed", error=str(e))
            return pd.DataFrame()

    @staticmethod
    def load_recent_alerts(
        alerting_system: Any, hours: int = 24, limit: int = 50
    ) -> pd.DataFrame:
        """Load recent alerts.

        Args:
            alerting_system: AlertingSystem instance
            hours: Hours of history
            limit: Maximum alerts

        Returns:
            DataFrame with alerts
        """
        try:
            alerts = alerting_system.get_alerts(limit=limit)
            if not alerts:
                return pd.DataFrame()

            return pd.DataFrame(alerts)
        except Exception as e:
            logger.error("alerts_load_failed", error=str(e))
            return pd.DataFrame()

    @staticmethod
    def load_drift_results(drift_detector: Any) -> pd.DataFrame:
        """Load drift detection results.

        Args:
            drift_detector: DriftDetector instance

        Returns:
            DataFrame with drift results
        """
        try:
            # This would load from drift detector
            return pd.DataFrame()
        except Exception as e:
            logger.error("drift_results_load_failed", error=str(e))
            return pd.DataFrame()


def display_dashboard_info_box(title: str, content: str, icon: str = "ℹ️") -> None:
    """Display an info box with icon and content.

    Args:
        title: Box title
        content: Box content
        icon: Icon emoji
    """
    st.info(f"{icon} **{title}**\n\n{content}")


def display_dashboard_warning(title: str, content: str) -> None:
    """Display a warning box.

    Args:
        title: Warning title
        content: Warning content
    """
    st.warning(f"⚠️ **{title}**\n\n{content}")


def display_dashboard_error(title: str, content: str) -> None:
    """Display an error box.

    Args:
        title: Error title
        content: Error content
    """
    st.error(f"❌ **{title}**\n\n{content}")


def display_dashboard_success(title: str, content: str) -> None:
    """Display a success box.

    Args:
        title: Success title
        content: Success content
    """
    st.success(f"✅ **{title}**\n\n{content}")
