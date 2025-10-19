"""Unit tests for analytics utilities and calculations."""

import pytest
import pandas as pd
import numpy as np

from f1_predict.web.utils.analytics import (
    AnalyticsCalculator,
    StandingsCalculator,
    PerformanceAnalyzer,
    CircuitAnalyzer,
    TrendAnalyzer,
    filter_by_time_period,
)


@pytest.fixture
def sample_race_data():
    """Create sample race data for testing."""
    np.random.seed(42)

    teams = ["Mercedes", "Red Bull Racing", "Ferrari"]
    drivers = ["Hamilton", "Verstappen", "Leclerc"]

    races = []
    for season in [2023, 2024]:
        for round_num in range(1, 6):
            for team_idx, team in enumerate(teams):
                races.append({
                    "race_id": f"race_{season}_{round_num}",
                    "season": season,
                    "round": round_num,
                    "circuit": "Monaco",
                    "date": pd.Timestamp(f"{season}-{round_num+3:02d}-15"),
                    "team": team,
                    "driver_id": drivers[team_idx],
                    "position": np.random.randint(1, 10),
                    "points": np.random.randint(0, 25),
                    "status": "DNF" if np.random.random() < 0.2 else "Finished",
                    "grid_position": np.random.randint(1, 20),
                })

    return pd.DataFrame(races)


class TestAnalyticsCalculator:
    """Tests for AnalyticsCalculator."""

    def test_calculate_kpis_with_data(self, sample_race_data):
        """Test KPI calculation with valid data."""
        kpis = AnalyticsCalculator.calculate_kpis(sample_race_data)

        assert kpis["total_races"] > 0
        assert kpis["data_quality_score"] >= 0
        assert kpis["data_quality_score"] <= 100
        assert kpis["avg_confidence"] >= 0

    def test_calculate_kpis_empty_data(self):
        """Test KPI calculation with empty data."""
        empty_df = pd.DataFrame()
        kpis = AnalyticsCalculator.calculate_kpis(empty_df)

        assert kpis["total_races"] == 0
        assert kpis["data_quality_score"] == 0.0
        assert kpis["avg_confidence"] == 0.0

    def test_calculate_accuracy_delta(self):
        """Test accuracy delta calculation."""
        current = 85.0
        historical = [80.0, 82.0, 83.0, 84.0, 84.5]

        delta = AnalyticsCalculator.calculate_accuracy_delta(current, historical)

        assert delta > 0  # Should be positive improvement

    def test_calculate_accuracy_delta_no_history(self):
        """Test accuracy delta with no historical data."""
        delta = AnalyticsCalculator.calculate_accuracy_delta(85.0)

        assert delta == 0.0


class TestStandingsCalculator:
    """Tests for StandingsCalculator."""

    def test_get_driver_standings(self, sample_race_data):
        """Test driver standings calculation."""
        standings = StandingsCalculator.get_driver_standings(sample_race_data)

        assert len(standings) > 0
        assert "position" in standings.columns
        assert "points" in standings.columns
        assert standings["position"].iloc[0] == 1

    def test_get_driver_standings_sorted(self, sample_race_data):
        """Test driver standings are sorted by points."""
        standings = StandingsCalculator.get_driver_standings(sample_race_data)

        points_list = standings["points"].tolist()
        assert points_list == sorted(points_list, reverse=True)

    def test_get_driver_standings_empty_data(self):
        """Test driver standings with empty data."""
        standings = StandingsCalculator.get_driver_standings(pd.DataFrame())

        assert len(standings) == 0

    def test_get_team_standings(self, sample_race_data):
        """Test team standings calculation."""
        standings = StandingsCalculator.get_team_standings(sample_race_data)

        assert len(standings) > 0
        assert "position" in standings.columns
        assert "team" in standings.columns
        assert standings["position"].iloc[0] == 1

    def test_get_team_standings_with_season_filter(self, sample_race_data):
        """Test team standings with season filter."""
        standings = StandingsCalculator.get_team_standings(sample_race_data, season=2024)

        # All races should be from 2024
        assert len(standings) > 0


class TestPerformanceAnalyzer:
    """Tests for PerformanceAnalyzer."""

    def test_calculate_win_rate(self, sample_race_data):
        """Test win rate calculation."""
        win_rate = PerformanceAnalyzer.calculate_win_rate(sample_race_data)

        assert len(win_rate) > 0
        assert "team" in win_rate.columns
        assert "win_rate" in win_rate.columns
        assert all(0 <= wr <= 100 for wr in win_rate["win_rate"])

    def test_calculate_win_rate_empty_data(self):
        """Test win rate with empty data."""
        win_rate = PerformanceAnalyzer.calculate_win_rate(pd.DataFrame())

        assert len(win_rate) == 0

    def test_calculate_reliability(self, sample_race_data):
        """Test reliability calculation."""
        reliability = PerformanceAnalyzer.calculate_reliability(sample_race_data)

        assert len(reliability) > 0
        assert "finishes" in reliability.columns
        assert "dnfs" in reliability.columns
        assert "reliability_rate" in reliability.columns
        assert all(0 <= rate <= 100 for rate in reliability["reliability_rate"])

    def test_calculate_reliability_rate_valid(self, sample_race_data):
        """Test reliability rate is logically consistent."""
        reliability = PerformanceAnalyzer.calculate_reliability(sample_race_data)

        for _, row in reliability.iterrows():
            total = row["finishes"] + row["dnfs"]
            expected_rate = (row["finishes"] / total * 100) if total > 0 else 0
            assert abs(row["reliability_rate"] - expected_rate) < 0.1

    def test_calculate_points_distribution(self, sample_race_data):
        """Test points distribution calculation."""
        distribution = PerformanceAnalyzer.calculate_points_distribution(sample_race_data)

        assert len(distribution) > 0
        assert "avg_points" in distribution.columns
        assert "std_points" in distribution.columns
        assert "races" in distribution.columns

    def test_get_qualifying_vs_race_performance(self, sample_race_data):
        """Test qualifying vs race performance."""
        perf = PerformanceAnalyzer.get_qualifying_vs_race_performance(sample_race_data)

        assert len(perf) > 0


class TestCircuitAnalyzer:
    """Tests for CircuitAnalyzer."""

    def test_get_circuit_performance_heatmap(self, sample_race_data):
        """Test circuit performance heatmap generation."""
        heatmap = CircuitAnalyzer.get_circuit_performance_heatmap(sample_race_data)

        assert len(heatmap) > 0
        # Heatmap should have teams as rows

    def test_get_circuit_statistics(self, sample_race_data):
        """Test circuit statistics."""
        stats = CircuitAnalyzer.get_circuit_statistics(sample_race_data, "Monaco")

        assert "circuit" in stats
        assert "races" in stats
        assert stats["circuit"] == "Monaco"

    def test_get_circuit_statistics_nonexistent(self, sample_race_data):
        """Test circuit statistics for non-existent circuit."""
        stats = CircuitAnalyzer.get_circuit_statistics(sample_race_data, "NonExistent")

        assert stats["races"] == 0
        assert stats["unique_winners"] == 0


class TestTrendAnalyzer:
    """Tests for TrendAnalyzer."""

    def test_get_cumulative_points_trend(self, sample_race_data):
        """Test cumulative points trend calculation."""
        trend = TrendAnalyzer.get_cumulative_points_trend(sample_race_data)

        assert len(trend) > 0
        assert "points" in trend.columns

    def test_get_cumulative_points_trend_with_teams(self, sample_race_data):
        """Test cumulative points trend with team filter."""
        teams = ["Mercedes"]
        trend = TrendAnalyzer.get_cumulative_points_trend(sample_race_data, teams=teams)

        assert len(trend) > 0

    def test_get_performance_over_season(self, sample_race_data):
        """Test performance metrics over season."""
        performance = TrendAnalyzer.get_performance_over_season(
            sample_race_data, "Mercedes"
        )

        assert len(performance) > 0
        assert "cumulative_points" in performance.columns

    def test_get_performance_over_season_empty(self, sample_race_data):
        """Test performance for non-existent team."""
        performance = TrendAnalyzer.get_performance_over_season(
            sample_race_data, "NonExistent"
        )

        assert len(performance) == 0


class TestFilterByTimePeriod:
    """Tests for time period filtering."""

    def test_filter_last_5_races(self, sample_race_data):
        """Test filtering last 5 races."""
        filtered = filter_by_time_period(sample_race_data, "Last 5 Races")

        unique_races = filtered["race_id"].nunique()
        assert unique_races <= 5

    def test_filter_current_season(self, sample_race_data):
        """Test filtering current season."""
        filtered = filter_by_time_period(sample_race_data, "Current Season")

        # Should get 2024 season
        assert len(filtered) > 0

    def test_filter_last_2_seasons(self, sample_race_data):
        """Test filtering last 2 seasons."""
        filtered = filter_by_time_period(sample_race_data, "Last 2 Seasons")

        assert len(filtered) > 0

    def test_filter_all_time(self, sample_race_data):
        """Test filtering all time data."""
        filtered = filter_by_time_period(sample_race_data, "All Time")

        assert len(filtered) == len(sample_race_data)

    def test_filter_empty_data(self):
        """Test filtering empty data."""
        empty_df = pd.DataFrame()
        filtered = filter_by_time_period(empty_df, "Current Season")

        assert len(filtered) == 0
