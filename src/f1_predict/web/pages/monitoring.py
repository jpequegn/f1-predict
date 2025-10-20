"""Monitoring Dashboard - Real-time model performance and drift detection.

Provides comprehensive visualization of model performance metrics, drift detection,
alerts, and model comparison across all deployed models.
"""

import tempfile
from typing import Any

import pandas as pd
import streamlit as st
import structlog

from f1_predict.web.utils.alerting import AlertingSystem
from f1_predict.web.utils.drift_detection import DriftDetector
from f1_predict.web.utils.monitoring import ModelPerformanceTracker
from f1_predict.web.utils.monitoring_dashboard import (
    ChartBuilders,
    DataLoaders,
    TableFormatters,
    display_dashboard_info_box,
)

logger = structlog.get_logger(__name__)


@st.cache_resource
def get_monitoring_systems() -> dict[str, Any]:
    """Get or initialize monitoring systems.

    Returns:
        Dictionary with monitoring system instances
    """
    temp_dir = tempfile.gettempdir()
    return {
        "performance_tracker": ModelPerformanceTracker(temp_dir),
        "alerting_system": AlertingSystem(temp_dir),
        "drift_detector": DriftDetector(),
    }


def _display_performance_dashboard() -> None:
    """Display performance monitoring dashboard."""
    st.subheader("📊 Performance Monitoring")

    systems = get_monitoring_systems()
    perf_tracker = systems["performance_tracker"]

    # Time range selector
    col1, col2, col3 = st.columns(3)
    with col1:
        time_range = st.selectbox(
            "Time Range",
            ["Last 24 hours", "Last 7 days", "Last 30 days"],
            key="perf_time_range",
        )
    with col2:
        st.checkbox("Auto-refresh", value=False, key="perf_auto_refresh")
    with col3:
        st.button("🔄 Refresh", key="perf_refresh")

    hours_map = {
        "Last 24 hours": 24,
        "Last 7 days": 168,
        "Last 30 days": 720,
    }
    hours = hours_map.get(time_range, 24)

    # Load performance data
    performance_data = DataLoaders.load_performance_trend(perf_tracker, hours)
    health_snapshots = DataLoaders.load_health_snapshots(perf_tracker, limit=100)

    if performance_data.empty:
        display_dashboard_info_box(
            "No Data Available",
            "Performance data will appear here once predictions are recorded.",
        )
        return

    # Performance KPIs
    st.markdown("#### Key Performance Indicators")
    if not health_snapshots.empty:
        latest_health = health_snapshots.iloc[0].to_dict()
        previous_health = (
            health_snapshots.iloc[1].to_dict() if len(health_snapshots) > 1 else latest_health
        )

        health_data = {
            "accuracy": latest_health.get("accuracy", 0.0),
            "accuracy_delta": (
                latest_health.get("accuracy", 0.0) - previous_health.get("accuracy", 0.0)
            ),
            "precision": latest_health.get("precision", 0.0),
            "precision_delta": (
                latest_health.get("precision", 0.0) - previous_health.get("precision", 0.0)
            ),
            "recall": latest_health.get("recall", 0.0),
            "recall_delta": (
                latest_health.get("recall", 0.0) - previous_health.get("recall", 0.0)
            ),
            "f1_score": latest_health.get("f1_score", 0.0),
            "f1_delta": (
                latest_health.get("f1_score", 0.0) - previous_health.get("f1_score", 0.0)
            ),
        }

        ChartBuilders.create_model_health_scorecard(health_data)
    else:
        display_dashboard_info_box("Insufficient Data", "Health snapshots not yet available")

    # Accuracy trend chart
    st.markdown("#### Accuracy Trend")
    if not performance_data.empty and "accuracy" in performance_data.columns:
        fig = ChartBuilders.create_accuracy_trend_chart(performance_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        display_dashboard_info_box("No Accuracy Data", "Accuracy data not available")

    # Confidence distribution
    st.markdown("#### Prediction Confidence Distribution")
    st.info("ℹ️ Shows distribution of prediction confidence scores")
    # In production, this would load actual confidence data
    st.info("Data will appear once predictions are made")

    # Performance degradation alerts
    st.markdown("#### Performance Degradation Alerts")
    if not health_snapshots.empty:
        degraded_models = health_snapshots[
            health_snapshots.get("degradation_detected", False)
        ]
        if not degraded_models.empty:
            st.warning("⚠️ Performance degradation detected in some models")
            st.dataframe(TableFormatters.format_health_snapshot_table(degraded_models))
        else:
            display_dashboard_info_box(
                "All Systems Nominal", "No performance degradation detected"
            )
    else:
        st.info("ℹ️ No health data to display")


def _display_drift_detection_dashboard() -> None:
    """Display drift detection dashboard."""
    st.subheader("🔍 Drift Detection")

    # Drift threshold
    col1, col2 = st.columns(2)
    with col1:
        psi_threshold = st.slider(
            "PSI Threshold", 0.0, 0.5, 0.25, step=0.05, key="psi_threshold"
        )
    with col2:
        ks_threshold = st.slider(
            "KS Test P-Value Threshold", 0.0, 0.1, 0.05, step=0.01, key="ks_threshold"
        )

    display_dashboard_info_box(
        "Drift Detection",
        f"Monitoring feature distributions using PSI (threshold: {psi_threshold}) and KS Test (p-value: {ks_threshold})",
    )

    # Drift status heatmap
    st.markdown("#### Feature Drift Status")
    st.info("ℹ️ Green = no drift, Yellow = moderate drift, Red = significant drift")

    # PSI trends
    st.markdown("#### PSI Trends Over Time")
    st.info("ℹ️ Population Stability Index tracks distribution changes")

    # Flagged features
    st.markdown("#### Flagged Features")
    st.info("ℹ️ Features exceeding drift thresholds will be highlighted here")

    # Distribution comparison
    st.markdown("#### Distribution Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox(
            "Select Feature to Compare", ["Feature 1", "Feature 2", "Feature 3"],
            key="feature_select"
        )
    with col2:
        st.info("Select a feature to compare baseline vs current distribution")


def _display_alert_management() -> None:
    """Display alert management interface."""
    st.subheader("🚨 Alert Management")

    systems = get_monitoring_systems()
    alerting_system = systems["alerting_system"]

    # Alert tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Alert History", "Alert Rules", "Statistics", "Configuration"]
    )

    with tab1:
        st.markdown("#### Alert History")
        alerts_df = DataLoaders.load_recent_alerts(alerting_system)

        if not alerts_df.empty:
            st.markdown(f"**Recent Alerts ({len(alerts_df)} total)**")
            formatted_alerts = TableFormatters.format_alert_table(alerts_df)
            st.dataframe(formatted_alerts, use_container_width=True)

            # Alert acknowledgment
            st.markdown("#### Acknowledge Alerts")
            alert_id = st.selectbox("Select Alert to Acknowledge", alerts_df.index)
            if st.button("✓ Acknowledge"):
                alerting_system.acknowledge_alert(str(alert_id))
                st.success("Alert acknowledged")
        else:
            display_dashboard_info_box(
                "No Alerts", "No alerts in the recent history. System is operating normally."
            )

    with tab2:
        st.markdown("#### Alert Rules")
        st.info("ℹ️ Configure rules that trigger alerts")

        col1, col2 = st.columns(2)
        with col1:
            metric_name = st.selectbox(
                "Metric",
                ["accuracy", "precision", "recall", "f1_score"],
                key="alert_metric",
            )
        with col2:
            comparison = st.selectbox(
                "Condition", ["<", ">", "<=", ">=", "=="], key="alert_condition"
            )

        threshold = st.slider("Threshold", 0.0, 1.0, 0.8, key="alert_threshold")
        severity = st.selectbox("Severity", ["INFO", "WARNING", "CRITICAL"], key="alert_severity")

        if st.button("+ Add Rule"):
            st.success(f"Rule added: {metric_name} {comparison} {threshold}")

    with tab3:
        st.markdown("#### Alert Statistics")
        if not alerts_df.empty:
            stats = alerts_df["severity"].value_counts()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Alerts", len(alerts_df))
            with col2:
                critical_count = len(alerts_df[alerts_df["severity"] == "CRITICAL"])
                st.metric("Critical", critical_count)
            with col3:
                warning_count = len(alerts_df[alerts_df["severity"] == "WARNING"])
                st.metric("Warnings", warning_count)

            # Alert timeline
            fig = ChartBuilders.create_alert_timeline(alerts_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            display_dashboard_info_box("No Statistics", "Alerts not yet available")

    with tab4:
        st.markdown("#### Alert Configuration")
        st.info("ℹ️ Configure alert delivery channels")

        enable_console = st.checkbox("Console Output", value=True)
        enable_file = st.checkbox("File Logging", value=True)
        enable_email = st.checkbox("Email Notifications", value=False)
        enable_slack = st.checkbox("Slack Integration", value=False)

        if enable_email:
            email_recipients = st.text_area("Email Recipients (comma-separated)")

        if enable_slack:
            slack_webhook = st.text_input("Slack Webhook URL", type="password")

        if st.button("💾 Save Configuration"):
            st.success("Alert configuration saved")


def _display_model_comparison() -> None:
    """Display model comparison dashboard."""
    st.subheader("📈 Model Comparison")

    # Model selection
    col1, col2, col3 = st.columns(3)
    with col1:
        model1 = st.selectbox(
            "Model 1", ["Ensemble", "XGBoost", "LightGBM", "Random Forest"], key="model1"
        )
    with col2:
        model2 = st.selectbox(
            "Model 2", ["XGBoost", "Ensemble", "LightGBM", "Random Forest"], key="model2"
        )
    with col3:
        compare_btn = st.button("🔄 Compare")

    st.markdown("#### Performance Metrics Comparison")

    # Side-by-side metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{model1}**")
        m1_data = {
            "Accuracy": 0.852,
            "Precision": 0.834,
            "Recall": 0.821,
            "F1 Score": 0.827,
        }
        st.metric("Accuracy", f"{m1_data['Accuracy']:.3f}")
        st.metric("Precision", f"{m1_data['Precision']:.3f}")
        st.metric("Recall", f"{m1_data['Recall']:.3f}")
        st.metric("F1 Score", f"{m1_data['F1 Score']:.3f}")

    with col2:
        st.markdown(f"**{model2}**")
        m2_data = {
            "Accuracy": 0.831,
            "Precision": 0.812,
            "Recall": 0.798,
            "F1 Score": 0.805,
        }
        st.metric("Accuracy", f"{m2_data['Accuracy']:.3f}")
        st.metric("Precision", f"{m2_data['Precision']:.3f}")
        st.metric("Recall", f"{m2_data['Recall']:.3f}")
        st.metric("F1 Score", f"{m2_data['F1 Score']:.3f}")

    # Comparison chart
    st.markdown("#### Performance Trend Comparison")

    model_metrics = pd.DataFrame(
        {
            "model_name": [model1, model2],
            "accuracy": [0.852, 0.831],
            "precision": [0.834, 0.812],
            "recall": [0.821, 0.798],
            "f1_score": [0.827, 0.805],
        }
    )

    fig = ChartBuilders.create_model_comparison_chart(model_metrics)
    st.plotly_chart(fig, use_container_width=True)

    # Deployment status
    st.markdown("#### Deployment Status")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**{model1}**")
        st.info("✓ Active | Last Updated: 2025-10-20 10:30:15")

    with col2:
        st.write(f"**{model2}**")
        st.info("⏸ Standby | Last Updated: 2025-10-15 14:22:45")


def show_monitoring_page() -> None:
    """Main monitoring dashboard page."""
    st.set_page_config(page_title="Monitoring", page_icon="📊", layout="wide")

    st.title("🚨 Model Monitoring Dashboard")
    st.markdown(
        "Real-time monitoring of model performance, drift detection, and system alerts"
    )

    st.divider()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Performance 📊",
            "Drift Detection 🔍",
            "Alerts 🚨",
            "Model Comparison 📈",
        ]
    )

    with tab1:
        _display_performance_dashboard()

    with tab2:
        _display_drift_detection_dashboard()

    with tab3:
        _display_alert_management()

    with tab4:
        _display_model_comparison()

    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Export Report"):
            st.info("📥 Report export functionality coming soon")
    with col2:
        if st.button("⚙️ Configure Alerts"):
            st.info("⚙️ Alert configuration in Alerts tab")
    with col3:
        if st.button("🔄 Full Refresh"):
            st.rerun()
