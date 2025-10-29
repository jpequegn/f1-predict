"""Tests for F1 visualization utilities.

Tests for all visualization functions in the web utils module,
including race visualizations, analytics dashboards, and the central
F1Visualizer class.
"""

import numpy as np
import pandas as pd
import pytest
from plotly import graph_objects as go

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
from f1_predict.web.utils.f1_visualizer import F1Visualizer, F1VisualizerTheme
from f1_predict.web.utils.visualization import (
    create_driver_radar_chart,
    create_lap_time_heatmap,
    create_position_changes_chart,
    create_position_distribution,
    create_points_trend,
    create_race_results_comparison,
    create_stats_comparison,
)


@pytest.fixture
def sample_race_data():
    """Create sample race data for testing."""
    return pd.DataFrame(
        {
            "position": [1, 2, 3, 4, 5],
            "points": [25, 18, 15, 12, 10],
            "driver": ["Max", "Lewis", "Fernando", "Lando", "Oscar"],
        }
    )


@pytest.fixture
def sample_lap_data():
    """Create sample lap data for testing."""
    return pd.DataFrame(
        {
            "lap": [1, 2, 3, 1, 2, 3],
            "driver": ["Max", "Max", "Max", "Lewis", "Lewis", "Lewis"],
            "position": [1, 1, 1, 2, 2, 2],
            "time": [95.2, 95.1, 95.0, 96.1, 96.0, 95.9],
        }
    )


@pytest.fixture
def sample_team_data():
    """Create sample team data for testing."""
    return pd.DataFrame(
        {
            "team": ["Red Bull", "Mercedes", "Ferrari", "McLaren"],
            "win_rate": [45.0, 35.0, 15.0, 5.0],
            "finishes": [20, 18, 15, 12],
            "dnfs": [2, 4, 5, 8],
        }
    )


@pytest.fixture
def sample_standings_data():
    """Create sample standings data for testing."""
    return pd.DataFrame(
        {
            "position": [1, 2, 3, 4, 5],
            "driver_id": ["max_verstappen", "lewis_hamilton", "fernando_alonso", "lando_norris", "oscar_piastri"],
            "points": [502, 346, 321, 310, 275],
        }
    )


@pytest.fixture
def sample_circuit_data():
    """Create sample circuit performance data for testing."""
    return pd.DataFrame(
        {
            "Monza": [1.5, 2.0, 2.5],
            "Monaco": [1.3, 1.5, 2.0],
            "Silverstone": [1.8, 2.2, 1.5],
            "Spa": [1.2, 1.8, 2.3],
            "Suzuka": [2.0, 2.3, 1.8],
        },
        index=["Red Bull", "Mercedes", "Ferrari"],
    )


class TestRaceVisualizations:
    """Tests for race-specific visualization functions."""

    def test_race_position_comparison_creates_figure(self, sample_race_data):
        """Test that race position comparison creates valid figure."""
        fig = create_race_results_comparison(
            sample_race_data,
            sample_race_data,
            "Driver 1",
            "Driver 2",
        )
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Race Results Comparison"

    def test_race_position_comparison_empty_data(self):
        """Test race position comparison with empty data."""
        empty_df = pd.DataFrame()
        fig = create_race_results_comparison(empty_df, empty_df, "D1", "D2")
        assert isinstance(fig, go.Figure)

    def test_points_trend_creates_figure(self, sample_race_data):
        """Test that points trend creates valid figure."""
        fig = create_points_trend(
            sample_race_data,
            sample_race_data,
            "Driver 1",
            "Driver 2",
        )
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Cumulative Points Trend"

    def test_position_distribution_creates_figure(self, sample_race_data):
        """Test that position distribution creates valid figure."""
        fig = create_position_distribution(
            sample_race_data,
            sample_race_data,
            "Driver 1",
            "Driver 2",
        )
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Finishing Position Distribution"

    def test_stats_comparison_creates_figure(self):
        """Test that stats comparison creates valid figure."""
        stats = {
            "driver1_id": "max",
            "driver2_id": "lewis",
            "wins": {"max": 15, "lewis": 10},
            "podiums": {"max": 20, "lewis": 18},
            "races_competed": {"max": 22, "lewis": 22},
        }
        fig = create_stats_comparison(stats)
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Head-to-Head Statistics"

    def test_position_changes_creates_figure(self, sample_lap_data):
        """Test that position changes chart creates valid figure."""
        fig = create_position_changes_chart(sample_lap_data)
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Race Position Changes"

    def test_position_changes_empty_data(self):
        """Test position changes with empty data."""
        fig = create_position_changes_chart(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_lap_time_heatmap_creates_figure(self, sample_lap_data):
        """Test that lap time heatmap creates valid figure."""
        fig = create_lap_time_heatmap(sample_lap_data)
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Lap Time Performance Heatmap"

    def test_lap_time_heatmap_empty_data(self):
        """Test lap time heatmap with empty data."""
        fig = create_lap_time_heatmap(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_driver_radar_creates_figure(self):
        """Test that driver radar chart creates valid figure."""
        metrics = {
            "Race Pace": 95,
            "Qualifying": 92,
            "Tire Management": 88,
            "Consistency": 90,
            "Wet Weather": 85,
            "Strategy": 89,
        }
        fig = create_driver_radar_chart("Max Verstappen", metrics)
        assert isinstance(fig, go.Figure)
        assert "Max Verstappen" in fig.layout.title.text


class TestAnalyticsVisualizations:
    """Tests for analytics dashboard visualization functions."""

    def test_win_rate_chart_creates_figure(self, sample_team_data):
        """Test that win rate chart creates valid figure."""
        fig = create_win_rate_chart(sample_team_data)
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Win Rate by Team"

    def test_win_rate_chart_empty_data(self):
        """Test win rate chart with empty data."""
        fig = create_win_rate_chart(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_reliability_chart_creates_figure(self, sample_team_data):
        """Test that reliability chart creates valid figure."""
        fig = create_reliability_chart(sample_team_data)
        assert isinstance(fig, go.Figure)
        assert "Reliability" in fig.layout.title.text

    def test_reliability_chart_empty_data(self):
        """Test reliability chart with empty data."""
        fig = create_reliability_chart(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_qualifying_vs_race_creates_figure(self, sample_race_data):
        """Test that qualifying vs race chart creates valid figure."""
        perf_data = sample_race_data.copy()
        perf_data["grid_position"] = [2, 1, 3, 4, 5]
        perf_data["team"] = ["Red Bull", "Mercedes", "Ferrari", "McLaren", "Alpine"]
        fig = create_qualifying_vs_race_chart(perf_data)
        assert isinstance(fig, go.Figure)
        assert "Qualifying" in fig.layout.title.text

    def test_circuit_heatmap_creates_figure(self, sample_circuit_data):
        """Test that circuit heatmap creates valid figure."""
        fig = create_circuit_heatmap(sample_circuit_data)
        assert isinstance(fig, go.Figure)
        assert "Circuit" in fig.layout.title.text

    def test_circuit_heatmap_empty_data(self):
        """Test circuit heatmap with empty data."""
        fig = create_circuit_heatmap(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_points_progression_creates_figure(self, sample_race_data):
        """Test that points progression creates valid figure."""
        trend_data = sample_race_data.copy()
        trend_data["race_number"] = [1, 2, 3, 4, 5]
        trend_data["team"] = ["Red Bull", "Mercedes", "Ferrari", "McLaren", "Alpine"]
        fig = create_points_progression_chart(trend_data)
        assert isinstance(fig, go.Figure)
        assert "Points Progression" in fig.layout.title.text

    def test_points_distribution_creates_figure(self, sample_team_data):
        """Test that points distribution creates valid figure."""
        dist_data = sample_team_data.copy()
        dist_data["min_points"] = [10, 8, 5, 3]
        dist_data["avg_points"] = [15, 12, 10, 7]
        dist_data["max_points"] = [25, 18, 15, 10]
        fig = create_points_distribution_chart(dist_data)
        assert isinstance(fig, go.Figure)
        assert "Points Distribution" in fig.layout.title.text

    def test_standings_chart_creates_figure(self, sample_standings_data):
        """Test that standings chart creates valid figure."""
        fig = create_standings_chart(sample_standings_data, driver_standings=True)
        assert isinstance(fig, go.Figure)
        assert "Driver Championship" in fig.layout.title.text

    def test_standings_chart_constructor_mode(self, sample_standings_data):
        """Test standings chart in constructor mode."""
        sample_standings_data["driver_id"] = [
            "red_bull",
            "mercedes",
            "ferrari",
            "mclaren",
            "alpine",
        ]
        fig = create_standings_chart(sample_standings_data, driver_standings=False)
        assert isinstance(fig, go.Figure)
        assert "Constructor Championship" in fig.layout.title.text

    def test_confidence_distribution_creates_figure(self):
        """Test that confidence distribution creates valid figure."""
        confidence_data = pd.DataFrame(
            {
                "confidence": np.random.rand(100),
                "category": np.random.choice(["Podium", "Points", "Other"], 100),
            }
        )
        fig = create_prediction_confidence_distribution(confidence_data)
        assert isinstance(fig, go.Figure)
        assert "Confidence Distribution" in fig.layout.title.text

    def test_feature_importance_waterfall_creates_figure(self):
        """Test that feature importance waterfall creates valid figure."""
        importance_data = pd.DataFrame(
            {
                "feature": ["Pace", "Tire", "Weather", "Strategy", "Reliability"],
                "importance": [0.35, 0.25, 0.2, 0.15, 0.05],
            }
        )
        fig = create_feature_importance_waterfall(importance_data)
        assert isinstance(fig, go.Figure)
        assert "Feature Importance" in fig.layout.title.text

    def test_circuit_sector_performance_creates_figure(self):
        """Test that circuit sector performance creates valid figure."""
        sector_data = pd.DataFrame(
            {
                "driver": ["Max", "Max", "Max", "Lewis", "Lewis", "Lewis"],
                "sector": [1, 2, 3, 1, 2, 3],
                "time": [45.2, 50.1, 48.9, 46.1, 51.0, 49.8],
            }
        )
        fig = create_circuit_sector_performance(sector_data)
        assert isinstance(fig, go.Figure)
        assert "Sector" in fig.layout.title.text


class TestF1VisualizerTheme:
    """Tests for F1VisualizerTheme class."""

    def test_theme_colors_are_defined(self):
        """Test that all theme colors are properly defined."""
        theme = F1VisualizerTheme()
        assert theme.PRIMARY == "#1F4E8C"
        assert theme.SUCCESS == "#28A745"
        assert theme.DANGER == "#DC3545"
        assert theme.WARNING == "#FFC107"

    def test_theme_plotly_template(self):
        """Test that theme has correct Plotly template."""
        theme = F1VisualizerTheme()
        assert theme.PLOTLY_TEMPLATE == "plotly_dark"


class TestF1Visualizer:
    """Tests for F1Visualizer central class."""

    def test_visualizer_initialization(self):
        """Test that visualizer initializes properly."""
        viz = F1Visualizer()
        assert viz.theme is not None
        assert isinstance(viz.theme, F1VisualizerTheme)

    def test_visualizer_invalid_theme(self):
        """Test that visualizer raises error for invalid theme."""
        with pytest.raises(ValueError, match="Unknown theme"):
            F1Visualizer(theme="invalid_theme")

    def test_visualizer_race_methods(self, sample_lap_data):
        """Test that visualizer has all race visualization methods."""
        viz = F1Visualizer()

        # Convert lap data to race-like format
        race_data = sample_lap_data[["position", "driver"]].drop_duplicates()

        # Test that methods exist and return figures
        fig = viz.race_position_comparison(race_data, race_data, "D1", "D2")
        assert isinstance(fig, go.Figure)

        race_data_with_points = race_data.copy()
        race_data_with_points["points"] = [25, 18]
        fig = viz.points_trend(race_data_with_points, race_data_with_points, "D1", "D2")
        assert isinstance(fig, go.Figure)

        fig = viz.position_distribution(race_data, race_data, "D1", "D2")
        assert isinstance(fig, go.Figure)

    def test_visualizer_analytics_methods(self, sample_team_data, sample_standings_data):
        """Test that visualizer has all analytics visualization methods."""
        viz = F1Visualizer()

        # Test win rate
        fig = viz.win_rate(sample_team_data)
        assert isinstance(fig, go.Figure)

        # Test reliability
        fig = viz.reliability(sample_team_data)
        assert isinstance(fig, go.Figure)

        # Test standings
        fig = viz.standings(sample_standings_data, driver_standings=True)
        assert isinstance(fig, go.Figure)

    def test_visualizer_get_theme_colors(self):
        """Test that visualizer returns theme colors."""
        viz = F1Visualizer()
        colors = viz.get_theme_colors()

        assert isinstance(colors, dict)
        assert "primary" in colors
        assert "success" in colors
        assert "danger" in colors
        assert colors["primary"] == "#1F4E8C"

    def test_visualizer_apply_theme_to_figure(self, sample_lap_data):
        """Test that visualizer can apply theme to figure."""
        viz = F1Visualizer()

        # Create a basic figure
        race_data = sample_lap_data[["position", "driver"]].drop_duplicates()
        fig = viz.race_position_comparison(race_data, race_data, "D1", "D2")

        # Apply theme
        fig = viz.apply_theme_to_figure(fig, height=500)

        # Check that theme was applied (template becomes a Template object after update_layout)
        # Check font color is set to theme primary color
        assert fig.layout.font.color == "#E0E6F0"
        assert fig.layout.height == 500

    def test_visualizer_all_methods_return_figures(self, sample_lap_data, sample_team_data, sample_standings_data):
        """Test that all visualizer methods return valid figures."""
        viz = F1Visualizer()

        # Create race data from lap data
        race_data = sample_lap_data[["position", "driver"]].drop_duplicates()
        race_data_with_points = race_data.copy()
        race_data_with_points["points"] = [25, 18]

        # Race visualizations
        assert isinstance(viz.race_position_comparison(race_data, race_data, "D1", "D2"), go.Figure)
        assert isinstance(viz.points_trend(race_data_with_points, race_data_with_points, "D1", "D2"), go.Figure)
        assert isinstance(viz.position_distribution(race_data, race_data, "D1", "D2"), go.Figure)
        assert isinstance(viz.stats_comparison({"driver1_id": "max", "driver2_id": "lewis", "wins": {}, "podiums": {}, "races_competed": {}}), go.Figure)

        # Analytics visualizations
        assert isinstance(viz.win_rate(sample_team_data), go.Figure)
        assert isinstance(viz.reliability(sample_team_data), go.Figure)
        assert isinstance(viz.standings(sample_standings_data), go.Figure)
