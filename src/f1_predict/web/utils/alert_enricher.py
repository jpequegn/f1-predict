"""Alert enrichment with explainability insights.

Adds SHAP-based explanations to alerts for improved actionability.
"""

from typing import Any, Optional

import structlog

from f1_predict.web.utils.alerting import Alert
from f1_predict.web.utils.monitoring_explainability import (
    DriftExplanation,
    PerformanceDegradationAnalysis,
    ShapExplainabilityMonitor,
)

logger = structlog.get_logger(__name__)


class ExplanabilityAlertEnricher:
    """Enriches alerts with SHAP-based explanations."""

    def __init__(self, explainability_monitor: Optional[ShapExplainabilityMonitor] = None):
        """Initialize alert enricher.

        Args:
            explainability_monitor: ShapExplainabilityMonitor instance
        """
        self.explainability_monitor = explainability_monitor
        self.logger = logger.bind(component="alert_enricher")

    def enrich_alert(
        self,
        alert: Alert,
        drift_explanation: Optional[DriftExplanation] = None,
        degradation_analysis: Optional[PerformanceDegradationAnalysis] = None,
    ) -> dict[str, Any]:
        """Enrich alert with explainability insights.

        Args:
            alert: Alert to enrich
            drift_explanation: Optional drift explanation
            degradation_analysis: Optional degradation analysis

        Returns:
            Enriched alert dictionary with explanations
        """
        enriched = {
            "alert": alert.to_dict(),
            "explanation": {},
        }

        # Add drift explanation
        if drift_explanation:
            enriched["explanation"]["drift"] = self._format_drift_explanation(drift_explanation)

        # Add degradation analysis
        if degradation_analysis:
            enriched["explanation"]["degradation"] = self._format_degradation_explanation(degradation_analysis)

        return enriched

    def format_email_with_explanation(
        self,
        alert: Alert,
        drift_explanation: Optional[DriftExplanation] = None,
        degradation_analysis: Optional[PerformanceDegradationAnalysis] = None,
    ) -> str:
        """Format enriched email alert with explanations.

        Args:
            alert: Alert to format
            drift_explanation: Optional drift explanation
            degradation_analysis: Optional degradation analysis

        Returns:
            Formatted HTML email body
        """
        severity_color = {
            "critical": "#DC3545",
            "warning": "#FFC107",
            "info": "#28A745",
        }.get(alert.severity.lower(), "#A3A9BF")

        # Base email content
        html_parts = [
            f"""
        <html>
            <body style="font-family: Arial, sans-serif; background-color: #f5f5f5; padding: 20px;">
                <div style="background-color: white; border-radius: 8px; padding: 20px; max-width: 600px; margin: 0 auto;">
                    <div style="border-left: 4px solid {severity_color}; padding-left: 16px; margin-bottom: 20px;">
                        <h2 style="margin: 0 0 10px 0; color: #121317;">
                            {alert.title}
                        </h2>
                        <p style="margin: 0; color: #666; font-size: 14px;">
                            Severity: <strong style="color: {severity_color};">
                                {alert.severity.upper()}
                            </strong>
                        </p>
                    </div>

                    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 4px; margin-bottom: 20px;">
                        <p style="margin: 10px 0; color: #333;">
                            <strong>Message:</strong> {alert.message}
                        </p>
                        <p style="margin: 10px 0; color: #333;">
                            <strong>Metric:</strong> {alert.metric_name}
                        </p>
                        <p style="margin: 10px 0; color: #333;">
                            <strong>Current Value:</strong> {alert.metric_value:.4f}
                        </p>
                        <p style="margin: 10px 0; color: #333;">
                            <strong>Threshold:</strong> {alert.threshold:.4f}
                        </p>
                    </div>
            """
        ]

        # Add drift explanation
        if drift_explanation:
            html_parts.append(
                self._format_drift_explanation_html(
                    drift_explanation, severity_color
                )
            )

        # Add degradation analysis
        if degradation_analysis:
            html_parts.append(
                self._format_degradation_explanation_html(
                    degradation_analysis, severity_color
                )
            )

        # Footer
        html_parts.append("""
                    <p style="font-size: 12px; color: #999; margin: 20px 0 0 0;">
                        This alert includes explainability insights powered by SHAP.
                    </p>
                </div>
            </body>
        </html>
        """)

        return "".join(html_parts)

    def format_slack_with_explanation(
        self,
        alert: Alert,
        drift_explanation: Optional[DriftExplanation] = None,
        degradation_analysis: Optional[PerformanceDegradationAnalysis] = None,
    ) -> list[dict[str, Any]]:
        """Format enriched Slack alert with explanations.

        Args:
            alert: Alert to format
            drift_explanation: Optional drift explanation
            degradation_analysis: Optional degradation analysis

        Returns:
            List of Slack block kit dictionaries
        """
        severity_emoji = {
            "critical": "🔴",
            "warning": "🟡",
            "info": "🟢",
        }.get(alert.severity.lower(), "ℹ️")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{severity_emoji} {alert.title}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:*\n{alert.severity.upper()}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Component:*\n{alert.component}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Metric:*\n{alert.metric_name}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Current Value:*\n{alert.metric_value:.4f}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Message:*\n{alert.message}",
                },
            },
        ]

        # Add drift explanation
        if drift_explanation:
            blocks.append(
                self._format_drift_explanation_slack(drift_explanation)
            )

        # Add degradation analysis
        if degradation_analysis:
            blocks.append(
                self._format_degradation_explanation_slack(degradation_analysis)
            )

        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"📊 *Explainability:* This alert includes SHAP-based insights | Model: `{alert.model_version}`",
                },
            }
        )

        return blocks

    def _format_drift_explanation(self, explanation: DriftExplanation) -> dict[str, Any]:
        """Format drift explanation."""
        return {
            "type": "drift",
            "feature": explanation.feature_name,
            "drift_type": explanation.drift_type,
            "baseline_mean": round(explanation.baseline_mean, 4),
            "current_mean": round(explanation.current_mean, 4),
            "shap_contribution": round(explanation.shap_contribution, 4),
            "confidence": round(explanation.confidence, 4),
            "contributing_features": explanation.contributing_features,
            "recommendation": explanation.recommendation,
        }

    def _format_degradation_explanation(self, analysis: PerformanceDegradationAnalysis) -> dict[str, Any]:
        """Format degradation analysis."""
        return {
            "type": "degradation",
            "metric": analysis.metric_name,
            "degradation_percent": round(analysis.degradation_percent, 2),
            "top_features": [
                {
                    "feature": f.feature_name,
                    "importance": round(f.importance_score, 4),
                    "percentage": round(f.percentage, 2),
                }
                for f in analysis.top_contributing_features[:3]
            ],
            "failure_count": analysis.failure_cohort_size,
            "recommendations": analysis.recommended_actions,
        }

    def _format_drift_explanation_html(self, explanation: DriftExplanation, color: str) -> str:
        """Format drift explanation as HTML."""
        return f"""
                    <div style="background-color: #e8f4f8; padding: 15px; border-radius: 4px; margin-bottom: 20px; border-left: 4px solid {color};">
                        <h3 style="margin: 0 0 10px 0; color: #121317;">🔍 Drift Explanation</h3>
                        <p style="margin: 5px 0; color: #333;">
                            <strong>Feature:</strong> {explanation.feature_name}
                        </p>
                        <p style="margin: 5px 0; color: #333;">
                            <strong>Drift Type:</strong> {explanation.drift_type.upper()}
                        </p>
                        <p style="margin: 5px 0; color: #333;">
                            <strong>Change:</strong> {explanation.baseline_mean:.4f} → {explanation.current_mean:.4f}
                        </p>
                        <p style="margin: 5px 0; color: #333;">
                            <strong>SHAP Contribution:</strong> {explanation.shap_contribution:.4f}
                        </p>
                        <p style="margin: 5px 0; color: #333;">
                            <strong>Confidence:</strong> {explanation.confidence:.1%}
                        </p>
                        <p style="margin: 10px 0 0 0; color: #333; font-style: italic;">
                            💡 <strong>Recommendation:</strong> {explanation.recommendation}
                        </p>
                    </div>
        """

    def _format_degradation_explanation_html(self, analysis: PerformanceDegradationAnalysis, color: str) -> str:
        """Format degradation analysis as HTML."""
        top_features_html = "".join([
            f"<li>{f.feature_name} ({f.percentage:.1f}%)</li>"
            for f in analysis.top_contributing_features[:3]
        ])

        recommendations_html = "".join([
            f"<li>{rec}</li>" for rec in analysis.recommended_actions
        ])

        return f"""
                    <div style="background-color: #fff3cd; padding: 15px; border-radius: 4px; margin-bottom: 20px; border-left: 4px solid {color};">
                        <h3 style="margin: 0 0 10px 0; color: #121317;">📉 Performance Analysis</h3>
                        <p style="margin: 5px 0; color: #333;">
                            <strong>Metric:</strong> {analysis.metric_name}
                        </p>
                        <p style="margin: 5px 0; color: #333;">
                            <strong>Degradation:</strong> {analysis.degradation_percent:.2f}%
                        </p>
                        <p style="margin: 5px 0; color: #333;">
                            <strong>Error Cases:</strong> {analysis.failure_cohort_size}
                        </p>
                        <p style="margin: 5px 0; color: #333;">
                            <strong>Top Contributing Features:</strong>
                        </p>
                        <ul style="margin: 5px 0; padding-left: 20px; color: #333;">
                            {top_features_html}
                        </ul>
                        <p style="margin: 10px 0 0 0; color: #333;">
                            <strong>Recommendations:</strong>
                        </p>
                        <ul style="margin: 5px 0; padding-left: 20px; color: #333;">
                            {recommendations_html}
                        </ul>
                    </div>
        """

    def _format_drift_explanation_slack(self, explanation: DriftExplanation) -> dict[str, Any]:
        """Format drift explanation as Slack block."""
        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*🔍 Drift Explanation:*\n"
                    f"• Feature: `{explanation.feature_name}`\n"
                    f"• Type: {explanation.drift_type.upper()}\n"
                    f"• Change: {explanation.baseline_mean:.4f} → {explanation.current_mean:.4f}\n"
                    f"• SHAP Contribution: {explanation.shap_contribution:.4f}\n"
                    f"• Confidence: {explanation.confidence:.1%}\n"
                    f"• Recommendation: {explanation.recommendation}"
                ),
            },
        }

    def _format_degradation_explanation_slack(self, analysis: PerformanceDegradationAnalysis) -> dict[str, Any]:
        """Format degradation analysis as Slack block."""
        top_features = ", ".join([f.feature_name for f in analysis.top_contributing_features[:3]])

        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*📉 Performance Analysis:*\n"
                    f"• Metric: `{analysis.metric_name}`\n"
                    f"• Degradation: {analysis.degradation_percent:.2f}%\n"
                    f"• Error Cases: {analysis.failure_cohort_size}\n"
                    f"• Top Features: {top_features}\n"
                    f"• Action: {analysis.recommended_actions[0] if analysis.recommended_actions else 'Monitor'}"
                ),
            },
        }
