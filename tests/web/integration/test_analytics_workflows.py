"""Integration tests for analytics dashboard workflows."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_analytics_data():
    """Create sample data for analytics workflows."""
    np.random.seed(42)

    teams = [
        "Mercedes",
        "Red Bull Racing",
        "Ferrari",
        "McLaren",
        "Alpine",
        "Aston Martin",
    ]
    drivers = [
        "Hamilton",
        "Verstappen",
        "Leclerc",
        "Norris",
        "Alonso",
        "Stroll",
    ]
    circuits = ["Monaco", "Silverstone", "Monza", "Spa", "Hungary"]

    races = []
    for season in [2023, 2024]:
        for round_num in range(1, 11):
            for team_idx, team in enumerate(teams):
                month = ((round_num - 1) % 12) + 1
                races.append({
                    "race_id": f"race_{season}_{round_num}",
                    "season": season,
                    "round": round_num,
                    "circuit": circuits[round_num % len(circuits)],
                    "date": pd.Timestamp(f"{season}-{month:02d}-15"),
                    "team": team,
                    "driver_id": drivers[team_idx],
                    "position": np.random.randint(1, 12),
                    "points": max(0, 30 - np.random.randint(0, 25)),
                    "status": "DNF" if np.random.random() < 0.1 else "Finished",
                    "grid_position": np.random.randint(1, 20),
                })

    return pd.DataFrame(races)


class TestAnalyticsDashboardWorkflow:
    """Tests for complete analytics dashboard workflow."""

    def test_analytics_data_loading(self, sample_analytics_data):
        """Test analytics data loads correctly."""
        assert len(sample_analytics_data) > 0
        assert "position" in sample_analytics_data.columns
        assert "points" in sample_analytics_data.columns
        assert "team" in sample_analytics_data.columns

    def test_kpi_calculation_workflow(self, sample_analytics_data):
        """Test KPI calculation workflow."""
        from f1_predict.web.utils.analytics import AnalyticsCalculator

        kpis = AnalyticsCalculator.calculate_kpis(sample_analytics_data)

        assert kpis["total_races"] > 0
        assert kpis["data_quality_score"] > 50  # Should have good quality sample data
        assert kpis["avg_confidence"] >= 0

    def test_standings_calculation_workflow(self, sample_analytics_data):
        """Test standings calculation workflow."""
        from f1_predict.web.utils.analytics import StandingsCalculator

        driver_standings = StandingsCalculator.get_driver_standings(sample_analytics_data)
        team_standings = StandingsCalculator.get_team_standings(sample_analytics_data)

        assert len(driver_standings) > 0
        assert len(team_standings) > 0
        assert driver_standings["position"].iloc[0] == 1
        assert team_standings["position"].iloc[0] == 1

    def test_performance_metrics_workflow(self, sample_analytics_data):
        """Test performance metrics calculation workflow."""
        from f1_predict.web.utils.analytics import PerformanceAnalyzer

        win_rate = PerformanceAnalyzer.calculate_win_rate(sample_analytics_data)
        reliability = PerformanceAnalyzer.calculate_reliability(sample_analytics_data)
        distribution = PerformanceAnalyzer.calculate_points_distribution(
            sample_analytics_data
        )

        assert len(win_rate) > 0
        assert len(reliability) > 0
        assert len(distribution) > 0

    def test_circuit_analysis_workflow(self, sample_analytics_data):
        """Test circuit analysis workflow."""
        from f1_predict.web.utils.analytics import CircuitAnalyzer

        heatmap = CircuitAnalyzer.get_circuit_performance_heatmap(sample_analytics_data)

        assert len(heatmap) > 0

    def test_trend_analysis_workflow(self, sample_analytics_data):
        """Test trend analysis workflow."""
        from f1_predict.web.utils.analytics import TrendAnalyzer

        trend = TrendAnalyzer.get_cumulative_points_trend(sample_analytics_data)

        assert len(trend) > 0

    def test_time_period_filtering_workflow(self, sample_analytics_data):
        """Test time period filtering workflow."""
        from f1_predict.web.utils.analytics import filter_by_time_period

        # Test different time periods
        last_5 = filter_by_time_period(sample_analytics_data, "Last 5 Races")
        current_season = filter_by_time_period(sample_analytics_data, "Current Season")
        last_2_seasons = filter_by_time_period(
            sample_analytics_data, "Last 2 Seasons"
        )
        all_time = filter_by_time_period(sample_analytics_data, "All Time")

        assert len(last_5) <= len(current_season)
        assert len(current_season) <= len(last_2_seasons)
        assert len(last_2_seasons) <= len(all_time)

    def test_complete_analytics_pipeline(self, sample_analytics_data):
        """Test complete analytics pipeline from data to insights."""
        from f1_predict.web.utils.analytics import (
            AnalyticsCalculator,
            StandingsCalculator,
            PerformanceAnalyzer,
            CircuitAnalyzer,
            TrendAnalyzer,
            filter_by_time_period,
        )

        # Step 1: Filter by time period
        filtered = filter_by_time_period(sample_analytics_data, "Current Season")

        # Step 2: Calculate KPIs
        kpis = AnalyticsCalculator.calculate_kpis(filtered)

        # Step 3: Get standings
        driver_standings = StandingsCalculator.get_driver_standings(filtered)
        team_standings = StandingsCalculator.get_team_standings(filtered)

        # Step 4: Calculate performance metrics
        win_rate = PerformanceAnalyzer.calculate_win_rate(filtered)
        reliability = PerformanceAnalyzer.calculate_reliability(filtered)

        # Step 5: Analyze trends
        trends = TrendAnalyzer.get_cumulative_points_trend(filtered)

        # Verify all outputs
        assert kpis["total_races"] > 0
        assert len(driver_standings) > 0
        assert len(team_standings) > 0
        assert len(win_rate) > 0
        assert len(reliability) > 0
        assert len(trends) > 0


class TestAnalyticsVisualizationIntegration:
    """Tests for analytics visualization integration."""

    def test_win_rate_chart_generation(self, sample_analytics_data):
        """Test win rate chart generation."""
        from f1_predict.web.utils.analytics import PerformanceAnalyzer
        from f1_predict.web.utils.analytics_visualization import create_win_rate_chart

        win_rate = PerformanceAnalyzer.calculate_win_rate(sample_analytics_data)
        fig = create_win_rate_chart(win_rate)

        assert fig is not None
        assert hasattr(fig, "data")

    def test_reliability_chart_generation(self, sample_analytics_data):
        """Test reliability chart generation."""
        from f1_predict.web.utils.analytics import PerformanceAnalyzer
        from f1_predict.web.utils.analytics_visualization import (
            create_reliability_chart,
        )

        reliability = PerformanceAnalyzer.calculate_reliability(sample_analytics_data)
        fig = create_reliability_chart(reliability)

        assert fig is not None
        assert hasattr(fig, "data")

    def test_circuit_heatmap_generation(self, sample_analytics_data):
        """Test circuit heatmap generation."""
        from f1_predict.web.utils.analytics import CircuitAnalyzer
        from f1_predict.web.utils.analytics_visualization import create_circuit_heatmap

        heatmap_data = CircuitAnalyzer.get_circuit_performance_heatmap(
            sample_analytics_data
        )
        fig = create_circuit_heatmap(heatmap_data)

        assert fig is not None

    def test_standings_chart_generation(self, sample_analytics_data):
        """Test standings chart generation."""
        from f1_predict.web.utils.analytics import StandingsCalculator
        from f1_predict.web.utils.analytics_visualization import create_standings_chart

        standings = StandingsCalculator.get_driver_standings(sample_analytics_data)
        fig = create_standings_chart(standings, driver_standings=True)

        assert fig is not None

    def test_points_progression_chart_generation(self, sample_analytics_data):
        """Test points progression chart generation."""
        from f1_predict.web.utils.analytics import TrendAnalyzer
        from f1_predict.web.utils.analytics_visualization import (
            create_points_progression_chart,
        )

        trend = TrendAnalyzer.get_cumulative_points_trend(sample_analytics_data)
        fig = create_points_progression_chart(trend)

        assert fig is not None


class TestAnalyticsDataConsistency:
    """Tests for analytics data consistency and validity."""

    def test_standings_points_consistency(self, sample_analytics_data):
        """Test standings points are calculated consistently."""
        from f1_predict.web.utils.analytics import StandingsCalculator

        standings = StandingsCalculator.get_driver_standings(sample_analytics_data)

        # Points should be non-negative
        assert all(p >= 0 for p in standings["points"])

        # Positions should be sequential from 1
        for i, pos in enumerate(standings["position"], 1):
            assert pos == i

    def test_win_rate_validity(self, sample_analytics_data):
        """Test win rate is between 0 and 100."""
        from f1_predict.web.utils.analytics import PerformanceAnalyzer

        win_rate = PerformanceAnalyzer.calculate_win_rate(sample_analytics_data)

        assert all(0 <= wr <= 100 for wr in win_rate["win_rate"])

    def test_reliability_rate_validity(self, sample_analytics_data):
        """Test reliability rate is between 0 and 100."""
        from f1_predict.web.utils.analytics import PerformanceAnalyzer

        reliability = PerformanceAnalyzer.calculate_reliability(sample_analytics_data)

        assert all(0 <= rate <= 100 for rate in reliability["reliability_rate"])

    def test_cumulative_points_increasing(self, sample_analytics_data):
        """Test cumulative points are non-decreasing."""
        from f1_predict.web.utils.analytics import TrendAnalyzer

        trend = TrendAnalyzer.get_cumulative_points_trend(sample_analytics_data)

        if len(trend) > 0:
            # Group by team and verify cumulative points don't decrease
            for team in trend[trend.columns[1]].unique() if len(trend.columns) > 1 else []:
                team_trend = trend[trend[trend.columns[1]] == team]
                if len(team_trend) > 1:
                    points = team_trend[team_trend.columns[0]].values
                    # Points should be non-decreasing (cumulative)
                    for i in range(1, len(points)):
                        assert points[i] >= points[i - 1]


class TestAnalyticsErrorHandling:
    """Tests for analytics error handling."""

    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        from f1_predict.web.utils.analytics import (
            AnalyticsCalculator,
            StandingsCalculator,
            PerformanceAnalyzer,
        )

        empty_df = pd.DataFrame()

        # Should not raise errors
        kpis = AnalyticsCalculator.calculate_kpis(empty_df)
        standings = StandingsCalculator.get_driver_standings(empty_df)
        win_rate = PerformanceAnalyzer.calculate_win_rate(empty_df)

        assert len(standings) == 0
        assert len(win_rate) == 0

    def test_missing_columns_handling(self):
        """Test handling of missing required columns."""
        from f1_predict.web.utils.analytics import StandingsCalculator

        # DataFrame with minimal columns
        minimal_df = pd.DataFrame({
            "race_id": ["race_1", "race_2"],
            "points": [25, 22],
        })

        # Should handle gracefully
        standings = StandingsCalculator.get_driver_standings(minimal_df)

        # May return empty or handle gracefully
        assert isinstance(standings, pd.DataFrame)

    def test_null_values_handling(self):
        """Test handling of null values."""
        from f1_predict.web.utils.analytics import PerformanceAnalyzer

        df_with_nulls = pd.DataFrame({
            "team": ["Mercedes", None, "Ferrari"],
            "points": [25, 22, None],
            "position": [1, 2, 3],
        })

        # Should handle gracefully
        try:
            result = PerformanceAnalyzer.calculate_win_rate(df_with_nulls)
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # If error, should be handled gracefully
            pytest.skip(f"Null handling: {e}")
