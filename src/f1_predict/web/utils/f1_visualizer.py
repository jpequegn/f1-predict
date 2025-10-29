"""Central F1Visualizer class for unified visualization management.

Provides a central registry for all F1 race and analytics visualizations
with consistent theming, color schemes, and interactive features.
"""

from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go

from f1_predict.web.utils.analytics_visualization import (
    create_circuit_heatmap,
    create_circuit_sector_performance,
    create_feature_importance_waterfall,
    create_points_distribution_chart,
    create_points_progression_chart,
    create_prediction_confidence_distribution,
    create_qualifying_vs_race_chart,
    create_reliability_chart,
    create_standings_chart,
    create_win_rate_chart,
)
from f1_predict.web.utils.visualization import (
    create_driver_radar_chart,
    create_lap_time_heatmap,
    create_points_trend,
    create_position_changes_chart,
    create_position_distribution,
    create_race_results_comparison,
    create_stats_comparison,
)


class F1VisualizerTheme:
    """Theme configuration for F1 visualizations (Nebula Dark)."""

    # Color palette
    BACKGROUND = "#121317"
    SURFACE = "#1E2130"
    SURFACE_HOVER = "#2A2F45"
    TEXT_PRIMARY = "#E0E6F0"
    TEXT_SECONDARY = "#A3A9BF"
    DIVIDER = "#333A56"

    # Accent colors
    PRIMARY = "#1F4E8C"
    PRIMARY_HOVER = "#2762B3"
    SUCCESS = "#28A745"
    WARNING = "#FFC107"
    DANGER = "#DC3545"

    # Theme name for Plotly
    PLOTLY_TEMPLATE = "plotly_dark"


class F1Visualizer:
    """Central registry for F1 race and analytics visualizations.

    Provides unified access to all visualization functions with consistent
    theming, color schemes, and layout configurations using the Nebula Dark
    design system.

    Example:
        >>> viz = F1Visualizer()
        >>> fig = viz.race_position_comparison(driver1_data, driver2_data, "Max", "Lewis")
        >>> fig.show()
    """

    def __init__(self, theme: str = "nebula_dark") -> None:
        """Initialize visualizer with theme configuration.

        Args:
            theme: Theme name (default: "nebula_dark")
        """
        self.theme = self._load_theme(theme)

    @staticmethod
    def _load_theme(theme_name: str) -> F1VisualizerTheme:
        """Load theme configuration.

        Args:
            theme_name: Name of theme to load

        Returns:
            F1VisualizerTheme instance

        Raises:
            ValueError: If theme not found
        """
        if theme_name != "nebula_dark":
            msg = f"Unknown theme: {theme_name}"
            raise ValueError(msg)
        return F1VisualizerTheme()

    # ===== RACE VISUALIZATIONS =====

    def race_position_comparison(
        self,
        driver1_data: pd.DataFrame,
        driver2_data: pd.DataFrame,
        driver1_name: str,
        driver2_name: str,
    ) -> go.Figure:
        """Create race results comparison chart.

        Args:
            driver1_data: Driver 1 race results
            driver2_data: Driver 2 race results
            driver1_name: Driver 1 name
            driver2_name: Driver 2 name

        Returns:
            Plotly figure
        """
        return create_race_results_comparison(
            driver1_data, driver2_data, driver1_name, driver2_name
        )

    def points_trend(
        self,
        driver1_data: pd.DataFrame,
        driver2_data: pd.DataFrame,
        driver1_name: str,
        driver2_name: str,
    ) -> go.Figure:
        """Create cumulative points trend chart.

        Args:
            driver1_data: Driver 1 race results
            driver2_data: Driver 2 race results
            driver1_name: Driver 1 name
            driver2_name: Driver 2 name

        Returns:
            Plotly figure
        """
        return create_points_trend(
            driver1_data, driver2_data, driver1_name, driver2_name
        )

    def position_distribution(
        self,
        driver1_data: pd.DataFrame,
        driver2_data: pd.DataFrame,
        driver1_name: str,
        driver2_name: str,
    ) -> go.Figure:
        """Create position distribution comparison.

        Args:
            driver1_data: Driver 1 race results
            driver2_data: Driver 2 race results
            driver1_name: Driver 1 name
            driver2_name: Driver 2 name

        Returns:
            Plotly figure
        """
        return create_position_distribution(
            driver1_data, driver2_data, driver1_name, driver2_name
        )

    def stats_comparison(self, stats: dict[str, Any]) -> go.Figure:
        """Create head-to-head stats comparison.

        Args:
            stats: Comparison statistics dictionary

        Returns:
            Plotly figure
        """
        return create_stats_comparison(stats)

    def position_changes(self, race_data: pd.DataFrame) -> go.Figure:
        """Create race position changes visualization.

        Args:
            race_data: DataFrame with lap, driver, and position

        Returns:
            Plotly figure
        """
        return create_position_changes_chart(race_data)

    def lap_time_heatmap(self, lap_times: pd.DataFrame) -> go.Figure:
        """Create lap time performance heatmap.

        Args:
            lap_times: DataFrame with driver, lap, and time

        Returns:
            Plotly figure
        """
        return create_lap_time_heatmap(lap_times)

    def driver_radar(self, driver: str, metrics: dict[str, Any]) -> go.Figure:
        """Create driver performance radar chart.

        Args:
            driver: Driver name
            metrics: Performance metrics dictionary

        Returns:
            Plotly figure
        """
        return create_driver_radar_chart(driver, metrics)

    # ===== ANALYTICS VISUALIZATIONS =====

    def win_rate(self, win_rate_data: pd.DataFrame) -> go.Figure:
        """Create win rate by team chart.

        Args:
            win_rate_data: DataFrame with team and win_rate columns

        Returns:
            Plotly figure
        """
        return create_win_rate_chart(win_rate_data)

    def reliability(self, reliability_data: pd.DataFrame) -> go.Figure:
        """Create reliability analysis chart.

        Args:
            reliability_data: DataFrame with team, finishes, and dnfs

        Returns:
            Plotly figure
        """
        return create_reliability_chart(reliability_data)

    def qualifying_vs_race(self, perf_data: pd.DataFrame) -> go.Figure:
        """Create qualifying vs race performance scatter.

        Args:
            perf_data: DataFrame with grid_position, race_position, team

        Returns:
            Plotly figure
        """
        return create_qualifying_vs_race_chart(perf_data)

    def circuit_heatmap(self, heatmap_data: pd.DataFrame) -> go.Figure:
        """Create circuit performance heatmap.

        Args:
            heatmap_data: DataFrame with teams as index, circuits as columns

        Returns:
            Plotly figure
        """
        return create_circuit_heatmap(heatmap_data)

    def points_progression(
        self,
        trend_data: pd.DataFrame,
        teams: Optional[list[str]] = None,
    ) -> go.Figure:
        """Create points progression line chart.

        Args:
            trend_data: DataFrame with race_number, team, points
            teams: Optional list of teams to highlight

        Returns:
            Plotly figure
        """
        return create_points_progression_chart(trend_data, teams)

    def points_distribution(self, distribution_data: pd.DataFrame) -> go.Figure:
        """Create points distribution box plot.

        Args:
            distribution_data: DataFrame with team and distribution stats

        Returns:
            Plotly figure
        """
        return create_points_distribution_chart(distribution_data)

    def standings(
        self,
        standings_data: pd.DataFrame,
        driver_standings: bool = True,
    ) -> go.Figure:
        """Create championship standings chart.

        Args:
            standings_data: DataFrame with standings (position, name, points)
            driver_standings: True for drivers, False for teams

        Returns:
            Plotly figure
        """
        return create_standings_chart(standings_data, driver_standings)

    def confidence_distribution(
        self,
        confidence_data: pd.DataFrame,
    ) -> go.Figure:
        """Create prediction confidence distribution.

        Args:
            confidence_data: DataFrame with confidence scores

        Returns:
            Plotly figure
        """
        return create_prediction_confidence_distribution(confidence_data)

    def feature_importance_waterfall(
        self,
        importance_data: pd.DataFrame,
    ) -> go.Figure:
        """Create feature importance waterfall chart.

        Args:
            importance_data: DataFrame with features and importance values

        Returns:
            Plotly figure
        """
        return create_feature_importance_waterfall(importance_data)

    def circuit_sector_performance(
        self,
        sector_data: pd.DataFrame,
    ) -> go.Figure:
        """Create circuit sector performance chart.

        Args:
            sector_data: DataFrame with sector, driver, and time

        Returns:
            Plotly figure
        """
        return create_circuit_sector_performance(sector_data)

    def get_theme_colors(self) -> dict[str, str]:
        """Get current theme colors.

        Returns:
            Dictionary with all theme colors
        """
        return {
            "background": self.theme.BACKGROUND,
            "surface": self.theme.SURFACE,
            "surface_hover": self.theme.SURFACE_HOVER,
            "text_primary": self.theme.TEXT_PRIMARY,
            "text_secondary": self.theme.TEXT_SECONDARY,
            "divider": self.theme.DIVIDER,
            "primary": self.theme.PRIMARY,
            "primary_hover": self.theme.PRIMARY_HOVER,
            "success": self.theme.SUCCESS,
            "warning": self.theme.WARNING,
            "danger": self.theme.DANGER,
        }

    def apply_theme_to_figure(
        self,
        fig: go.Figure,
        **layout_updates: Any,
    ) -> go.Figure:
        """Apply theme to a Plotly figure.

        Args:
            fig: Plotly figure to update
            **layout_updates: Additional layout updates

        Returns:
            Updated figure
        """
        fig.update_layout(
            template=self.theme.PLOTLY_TEMPLATE,
            font={"color": self.theme.TEXT_PRIMARY},
            **layout_updates,
        )
        return fig
