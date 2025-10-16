"""SHAP visualization utilities for F1 prediction explanations.

This module provides plotting functions for SHAP explanations with support
for both matplotlib and Plotly backends for interactive dashboards.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
import structlog

logger = structlog.get_logger(__name__)


class SHAPVisualizer:
    """Visualization utilities for SHAP explanations.

    Provides multiple visualization types:
    - Waterfall plots (single prediction)
    - Force plots (single prediction)
    - Summary plots (global importance)
    - Bar plots (feature importance)
    - Dependence plots (feature interactions)
    """

    def __init__(self, dark_mode: bool = True):
        """Initialize SHAP visualizer.

        Args:
            dark_mode: Whether to use dark mode for plots
        """
        self.dark_mode = dark_mode
        self.logger = logger.bind(component="shap_visualizer")

        # Set style based on mode
        if dark_mode:
            self.bg_color = "#121317"
            self.text_color = "#E0E6F0"
            self.grid_color = "#333A56"
            self.positive_color = "#28A745"
            self.negative_color = "#DC3545"
        else:
            self.bg_color = "#FFFFFF"
            self.text_color = "#000000"
            self.grid_color = "#E0E0E0"
            self.positive_color = "#2ECC71"
            self.negative_color = "#E74C3C"

    def waterfall_plot(
        self,
        explanation: dict[str, Any],
        max_display: int = 10,
        title: str = "Prediction Explanation",
    ) -> go.Figure:
        """Create interactive waterfall plot showing feature contributions.

        Args:
            explanation: SHAP explanation dictionary from SHAPExplainer
            max_display: Maximum number of features to display
            title: Plot title

        Returns:
            Plotly Figure object
        """
        self.logger.info("creating_waterfall_plot", max_display=max_display)

        # Extract data
        shap_values = explanation["shap_values"]
        feature_names = explanation["feature_names"]
        feature_values = explanation["feature_values"]
        base_value = explanation["base_value"]

        # Get top features by absolute SHAP value
        indices = np.argsort(np.abs(shap_values))[::-1][:max_display]

        # Build waterfall data
        display_features = [feature_names[i] for i in indices]
        display_shap = [shap_values[i] for i in indices]
        display_values = [feature_values[i] for i in indices]

        # Calculate cumulative sums for waterfall
        cumulative = [base_value]
        for shap_val in display_shap:
            cumulative.append(cumulative[-1] + shap_val)

        # Create labels with feature values
        labels = [
            f"{name}<br>value: {val:.2f}"
            for name, val in zip(display_features, display_values, strict=True)
        ]

        # Create waterfall plot
        fig = go.Figure()

        # Add bars
        for i, (label, shap_val) in enumerate(zip(labels, display_shap, strict=True)):
            color = self.positive_color if shap_val > 0 else self.negative_color
            fig.add_trace(
                go.Bar(
                    name=label,
                    x=[label],
                    y=[abs(shap_val)],
                    base=cumulative[i] if shap_val > 0 else cumulative[i + 1],
                    marker_color=color,
                    text=f"{shap_val:+.3f}",
                    textposition="outside",
                    hovertemplate=f"<b>{label}</b><br>SHAP: {shap_val:+.3f}<extra></extra>",
                )
            )

        # Add base value marker
        fig.add_trace(
            go.Scatter(
                x=[labels[0]],
                y=[base_value],
                mode="markers+text",
                name="Base Value",
                marker=dict(size=10, color=self.text_color),
                text=f"Base: {base_value:.3f}",
                textposition="top center",
                hovertemplate=f"<b>Base Value</b><br>{base_value:.3f}<extra></extra>",
            )
        )

        # Add final prediction marker
        final_value = cumulative[-1]
        fig.add_trace(
            go.Scatter(
                x=[labels[-1]],
                y=[final_value],
                mode="markers+text",
                name="Prediction",
                marker=dict(size=10, color=self.text_color),
                text=f"Prediction: {final_value:.3f}",
                textposition="top center",
                hovertemplate=f"<b>Final Prediction</b><br>{final_value:.3f}<extra></extra>",
            )
        )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Model Output",
            showlegend=False,
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(color=self.text_color),
            height=500,
        )

        return fig

    def force_plot(
        self,
        explanation: dict[str, Any],
        max_display: int = 10,
    ) -> go.Figure:
        """Create force plot showing how features push prediction up or down.

        Args:
            explanation: SHAP explanation dictionary
            max_display: Maximum number of features to display

        Returns:
            Plotly Figure object
        """
        self.logger.info("creating_force_plot", max_display=max_display)

        # Extract data
        shap_values = explanation["shap_values"]
        feature_names = explanation["feature_names"]
        feature_values = explanation["feature_values"]
        base_value = explanation["base_value"]
        final_value = explanation["model_output"]

        # Get top features
        indices = np.argsort(np.abs(shap_values))[::-1][:max_display]

        display_features = [feature_names[i] for i in indices]
        display_shap = [shap_values[i] for i in indices]
        display_values = [feature_values[i] for i in indices]

        # Separate positive and negative contributions
        positive_features = [
            (f, s, v) for f, s, v in zip(display_features, display_shap, display_values, strict=True) if s > 0
        ]
        negative_features = [
            (f, s, v) for f, s, v in zip(display_features, display_shap, display_values, strict=True) if s < 0
        ]

        fig = go.Figure()

        # Add horizontal bar for base to prediction
        fig.add_trace(
            go.Scatter(
                x=[base_value, final_value],
                y=[0.5, 0.5],
                mode="lines",
                line=dict(width=20, color=self.grid_color),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Add markers and annotations for features
        cumulative = base_value
        y_pos = 0.5

        # Positive features (push right)
        for i, (feature, shap_val, feat_val) in enumerate(positive_features):
            start = cumulative
            end = cumulative + shap_val
            cumulative = end

            fig.add_trace(
                go.Scatter(
                    x=[start, end],
                    y=[y_pos + (i + 1) * 0.02, y_pos + (i + 1) * 0.02],
                    mode="lines+markers",
                    line=dict(width=3, color=self.positive_color),
                    marker=dict(size=8),
                    name=feature,
                    hovertemplate=f"<b>{feature}</b><br>Value: {feat_val:.2f}<br>SHAP: +{shap_val:.3f}<extra></extra>",
                )
            )

            # Add text annotation
            fig.add_annotation(
                x=end,
                y=y_pos + (i + 1) * 0.02 + 0.05,
                text=f"{feature}<br>+{shap_val:.3f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=self.positive_color,
                font=dict(size=10, color=self.text_color),
            )

        # Negative features (push left)
        cumulative = base_value
        for i, (feature, shap_val, feat_val) in enumerate(negative_features):
            start = cumulative
            end = cumulative + shap_val
            cumulative = end

            fig.add_trace(
                go.Scatter(
                    x=[start, end],
                    y=[y_pos - (i + 1) * 0.02, y_pos - (i + 1) * 0.02],
                    mode="lines+markers",
                    line=dict(width=3, color=self.negative_color),
                    marker=dict(size=8),
                    name=feature,
                    hovertemplate=f"<b>{feature}</b><br>Value: {feat_val:.2f}<br>SHAP: {shap_val:.3f}<extra></extra>",
                )
            )

            # Add text annotation
            fig.add_annotation(
                x=end,
                y=y_pos - (i + 1) * 0.02 - 0.05,
                text=f"{feature}<br>{shap_val:.3f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=self.negative_color,
                font=dict(size=10, color=self.text_color),
            )

        # Add base value and prediction markers
        fig.add_trace(
            go.Scatter(
                x=[base_value],
                y=[y_pos],
                mode="markers+text",
                marker=dict(size=15, color=self.text_color, symbol="diamond"),
                text=f"Base<br>{base_value:.3f}",
                textposition="bottom center",
                name="Base Value",
                hovertemplate=f"<b>Base Value</b><br>{base_value:.3f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[final_value],
                y=[y_pos],
                mode="markers+text",
                marker=dict(size=15, color=self.text_color, symbol="diamond"),
                text=f"Prediction<br>{final_value:.3f}",
                textposition="bottom center",
                name="Prediction",
                hovertemplate=f"<b>Final Prediction</b><br>{final_value:.3f}<extra></extra>",
            )
        )

        # Update layout
        fig.update_layout(
            title="Force Plot - Feature Contributions",
            xaxis_title="Model Output",
            yaxis=dict(visible=False),
            showlegend=False,
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(color=self.text_color),
            height=600,
            hovermode="closest",
        )

        return fig

    def summary_plot(
        self,
        dataset_explanation: dict[str, Any],
        max_display: int = 15,
    ) -> go.Figure:
        """Create summary plot showing global feature importance.

        Args:
            dataset_explanation: Dataset-level SHAP explanation
            max_display: Maximum number of features to display

        Returns:
            Plotly Figure object
        """
        self.logger.info("creating_summary_plot", max_display=max_display)

        # Extract data
        mean_abs_shap = dataset_explanation["mean_abs_shap"]
        feature_names = dataset_explanation["feature_names"]

        # Sort by importance
        indices = np.argsort(mean_abs_shap)[::-1][:max_display]

        display_features = [feature_names[i] for i in indices]
        display_importance = [mean_abs_shap[i] for i in indices]

        # Create horizontal bar chart
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=display_features[::-1],  # Reverse for top-to-bottom display
                x=display_importance[::-1],
                orientation="h",
                marker=dict(
                    color=display_importance[::-1],
                    colorscale="Blues",
                    showscale=True,
                    colorbar=dict(title="Mean |SHAP|"),
                ),
                text=[f"{val:.3f}" for val in display_importance[::-1]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Global Feature Importance (Mean |SHAP|)",
            xaxis_title="Mean Absolute SHAP Value",
            yaxis_title="Features",
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(color=self.text_color),
            height=max(400, len(display_features) * 30),
        )

        return fig

    def feature_importance_bar(
        self,
        importance_dict: dict[str, float],
        top_k: int = 10,
        title: str = "Feature Importance",
    ) -> go.Figure:
        """Create bar chart for feature importance.

        Args:
            importance_dict: Dictionary mapping feature names to importance scores
            top_k: Number of top features to display
            title: Plot title

        Returns:
            Plotly Figure object
        """
        self.logger.info("creating_importance_bar", top_k=top_k)

        # Sort features by importance
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_k]

        features, importances = zip(*sorted_features, strict=True)

        # Create bar chart
        fig = go.Figure()

        colors = [
            self.positive_color if imp > 0 else self.negative_color
            for imp in importances
        ]

        fig.add_trace(
            go.Bar(
                y=list(features)[::-1],
                x=list(importances)[::-1],
                orientation="h",
                marker=dict(color=colors[::-1]),
                text=[f"{val:.3f}" for val in importances[::-1]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(color=self.text_color),
            height=max(400, len(features) * 30),
        )

        return fig

    def comparison_plot(
        self,
        comparisons: list[dict[str, Any]],
        labels: list[str],
    ) -> go.Figure:
        """Create comparison plot for multiple predictions or models.

        Args:
            comparisons: List of SHAP explanation dictionaries
            labels: List of labels for each comparison

        Returns:
            Plotly Figure object
        """
        self.logger.info("creating_comparison_plot", num_comparisons=len(comparisons))

        # Extract top features from each comparison
        all_features = set()
        for comp in comparisons:
            all_features.update(comp["feature_names"])

        # Create data for grouped bar chart
        feature_data = {feature: [] for feature in all_features}

        for comp in comparisons:
            shap_dict = dict(zip(comp["feature_names"], comp["shap_values"], strict=True))
            for feature in all_features:
                feature_data[feature].append(shap_dict.get(feature, 0))

        # Get top features by average absolute SHAP value
        avg_abs_shap = {
            feature: np.mean(np.abs(values))
            for feature, values in feature_data.items()
        }
        top_features = sorted(avg_abs_shap.items(), key=lambda x: x[1], reverse=True)[:10]
        top_feature_names = [f[0] for f in top_features]

        # Create grouped bar chart
        fig = go.Figure()

        for i, label in enumerate(labels):
            fig.add_trace(
                go.Bar(
                    name=label,
                    y=top_feature_names,
                    x=[feature_data[f][i] for f in top_feature_names],
                    orientation="h",
                    hovertemplate=f"<b>{label}</b><br>%{{y}}<br>SHAP: %{{x:.3f}}<extra></extra>",
                )
            )

        fig.update_layout(
            title="Feature Contribution Comparison",
            xaxis_title="SHAP Value",
            yaxis_title="Features",
            barmode="group",
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(color=self.text_color),
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        return fig
