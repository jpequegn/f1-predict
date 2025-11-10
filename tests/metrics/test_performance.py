"""Unit tests for performance metrics module.

Tests cover:
- Championship points analysis and trends
- Driver consistency and win rates
- Podium performance metrics
- Head-to-head comparisons
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest
import numpy as np

from f1_predict.metrics.performance import ChampionshipPointsAnalyzer


class TestChampionshipPointsAnalyzer:
    """Tests for ChampionshipPointsAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> ChampionshipPointsAnalyzer:
        """Create a championship points analyzer instance."""
        return ChampionshipPointsAnalyzer()

    @pytest.fixture
    def sample_race_results(self) -> pd.DataFrame:
        """Create sample race results for a season."""
        base_date = datetime(2024, 1, 1)
        drivers = ["driver_1", "driver_2", "driver_3"]
        data_list = []

        for race_num in range(10):
            for i, driver in enumerate(drivers):
                position = (i + race_num) % 3 + 1  # Rotating 1-3
                points = {1: 25, 2: 18, 3: 15}.get(position, 0)

                data_list.append({
                    "date": base_date + timedelta(days=14 * race_num),
                    "driver_id": driver,
                    "position": position,
                    "points": points,
                    "season": "2024",
                })

        return pd.DataFrame(data_list)

    @pytest.fixture
    def dominant_driver_results(self) -> pd.DataFrame:
        """Create results for a dominant driver."""
        base_date = datetime(2024, 1, 1)
        data = {
            "date": [base_date + timedelta(days=14 * i) for i in range(10)],
            "driver_id": ["driver_1"] * 10,
            "position": [1, 1, 1, 2, 1, 1, 1, 2, 1, 1],  # 8 wins, 2 2nd places
            "points": [25, 25, 25, 18, 25, 25, 25, 18, 25, 25],
            "season": ["2024"] * 10,
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def struggling_driver_results(self) -> pd.DataFrame:
        """Create results for a struggling driver."""
        base_date = datetime(2024, 1, 1)
        data = {
            "date": [base_date + timedelta(days=14 * i) for i in range(10)],
            "driver_id": ["driver_2"] * 10,
            "position": [15, 16, 14, 17, 18, 16, 15, 17, 16, 15],  # Poor positions
            "points": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No points
            "season": ["2024"] * 10,
        }
        return pd.DataFrame(data)

    def test_initialization_default(self, analyzer):
        """Test ChampionshipPointsAnalyzer initialization."""
        assert analyzer.season is None

    def test_initialization_with_season(self):
        """Test ChampionshipPointsAnalyzer with specific season."""
        analyzer = ChampionshipPointsAnalyzer(season="2024")
        assert analyzer.season == "2024"

    def test_analyze_driver_points_trend_basic(self, analyzer, sample_race_results):
        """Test basic points trend analysis."""
        trend = analyzer.analyze_driver_points_trend(sample_race_results, "driver_1")

        assert "total_points" in trend
        assert "avg_points_per_race" in trend
        assert "points_trend" in trend
        assert "podium_rate" in trend
        assert "win_rate" in trend
        assert "consistency_score" in trend
        assert "races_count" in trend

    def test_analyze_driver_points_trend_values(self, analyzer, sample_race_results):
        """Test that points trend values are reasonable."""
        trend = analyzer.analyze_driver_points_trend(sample_race_results, "driver_1")

        assert trend["total_points"] > 0
        assert trend["races_count"] == 10
        assert 0 <= trend["podium_rate"] <= 1
        assert 0 <= trend["win_rate"] <= 1
        assert 0 <= trend["consistency_score"] <= 100

    def test_analyze_dominant_driver(self, analyzer, dominant_driver_results):
        """Test analysis of dominant driver."""
        trend = analyzer.analyze_driver_points_trend(dominant_driver_results, "driver_1")

        # Dominant driver metrics
        assert trend["total_points"] > 200  # 8*25 + 2*18 = 236
        assert trend["win_rate"] >= 0.8  # Should have 80%+ win rate
        assert trend["podium_rate"] == 1.0  # 100% podiums

    def test_analyze_struggling_driver(self, analyzer, struggling_driver_results):
        """Test analysis of struggling driver."""
        trend = analyzer.analyze_driver_points_trend(struggling_driver_results, "driver_2")

        # Struggling driver metrics
        assert trend["total_points"] == 0  # No points
        assert trend["win_rate"] == 0.0  # No wins
        assert trend["podium_rate"] == 0.0  # No podiums

    def test_analyze_driver_no_results(self, analyzer):
        """Test analysis when driver has no results."""
        empty_df = pd.DataFrame({
            "date": [],
            "driver_id": [],
            "position": [],
            "points": [],
            "season": [],
        })

        trend = analyzer.analyze_driver_points_trend(empty_df, "unknown_driver")

        # Should return zero metrics
        assert trend["total_points"] == 0.0
        assert trend["races_count"] == 0
        assert trend["podium_rate"] == 0.0

    def test_analyze_driver_points_trend_with_season_filter(self):
        """Test points trend with season filtering."""
        base_date = datetime(2023, 1, 1)
        data_list = []

        # Add multiple seasons
        for season_num in [2023, 2024]:
            for race in range(5):
                data_list.append({
                    "date": base_date + timedelta(days=365 * (season_num - 2023) + 14 * race),
                    "driver_id": "driver_1",
                    "position": 1,
                    "points": 25,
                    "season": str(season_num),
                })

        df = pd.DataFrame(data_list)

        analyzer_2024 = ChampionshipPointsAnalyzer(season="2024")
        trend = analyzer_2024.analyze_driver_points_trend(df, "driver_1")

        # Should only count 2024 races
        assert trend["races_count"] == 5
        assert trend["total_points"] == 125.0  # 5 * 25

    def test_consistency_score_calculation(self, analyzer):
        """Test consistency score calculation."""
        base_date = datetime(2024, 1, 1)

        # Consistent performer
        consistent_data = {
            "date": [base_date + timedelta(days=14 * i) for i in range(5)],
            "driver_id": ["driver_1"] * 5,
            "position": [1, 1, 1, 1, 1],
            "points": [25, 25, 25, 25, 25],
            "season": ["2024"] * 5,
        }
        df_consistent = pd.DataFrame(consistent_data)

        trend = analyzer.analyze_driver_points_trend(df_consistent, "driver_1")

        # Consistent points should have high consistency score
        assert trend["consistency_score"] > 80

    def test_consistency_score_inconsistent(self, analyzer):
        """Test consistency score for inconsistent performer."""
        base_date = datetime(2024, 1, 1)

        # Highly variable performer
        variable_data = {
            "date": [base_date + timedelta(days=14 * i) for i in range(5)],
            "driver_id": ["driver_1"] * 5,
            "position": [1, 20, 1, 20, 1],
            "points": [25, 0, 25, 0, 25],
            "season": ["2024"] * 5,
        }
        df_variable = pd.DataFrame(variable_data)

        trend = analyzer.analyze_driver_points_trend(df_variable, "driver_1")

        # Variable points should have lower consistency score
        assert trend["consistency_score"] < 80

    def test_trend_direction_improving(self, analyzer):
        """Test trend detection for improving driver."""
        base_date = datetime(2024, 1, 1)

        # Improving performance
        improving_data = {
            "date": [base_date + timedelta(days=14 * i) for i in range(10)],
            "driver_id": ["driver_1"] * 10,
            "position": [20, 18, 15, 12, 10, 8, 5, 3, 2, 1],
            "points": [0, 0, 0, 0, 0, 1, 8, 15, 18, 25],
            "season": ["2024"] * 10,
        }
        df = pd.DataFrame(improving_data)

        trend = analyzer.analyze_driver_points_trend(df, "driver_1")

        # Should detect improving trend (positive slope)
        assert trend["points_trend"] >= 0

    def test_trend_direction_declining(self, analyzer):
        """Test trend detection for declining driver."""
        base_date = datetime(2024, 1, 1)

        # Declining performance
        declining_data = {
            "date": [base_date + timedelta(days=14 * i) for i in range(10)],
            "driver_id": ["driver_1"] * 10,
            "position": [1, 2, 3, 5, 8, 10, 12, 15, 18, 20],
            "points": [25, 18, 15, 8, 1, 0, 0, 0, 0, 0],
            "season": ["2024"] * 10,
        }
        df = pd.DataFrame(declining_data)

        trend = analyzer.analyze_driver_points_trend(df, "driver_1")

        # Should detect declining trend (negative slope)
        assert trend["points_trend"] <= 0

    def test_avg_points_per_race(self, analyzer):
        """Test average points per race calculation."""
        base_date = datetime(2024, 1, 1)

        data = {
            "date": [base_date + timedelta(days=14 * i) for i in range(5)],
            "driver_id": ["driver_1"] * 5,
            "position": [1, 2, 3, 1, 2],
            "points": [25, 18, 15, 25, 18],
            "season": ["2024"] * 5,
        }
        df = pd.DataFrame(data)

        trend = analyzer.analyze_driver_points_trend(df, "driver_1")

        # Total = 25+18+15+25+18 = 101, avg = 101/5 = 20.2
        assert abs(trend["avg_points_per_race"] - 20.2) < 0.1

    def test_podium_rate_calculation(self, analyzer):
        """Test podium rate calculation."""
        base_date = datetime(2024, 1, 1)

        # 6 podiums out of 10 races
        data = {
            "date": [base_date + timedelta(days=14 * i) for i in range(10)],
            "driver_id": ["driver_1"] * 10,
            "position": [1, 2, 3, 4, 5, 1, 2, 3, 10, 15],
            "points": [25, 18, 15, 12, 10, 25, 18, 15, 0, 0],
            "season": ["2024"] * 10,
        }
        df = pd.DataFrame(data)

        trend = analyzer.analyze_driver_points_trend(df, "driver_1")

        # Podium rate = 6/10 = 0.6
        assert abs(trend["podium_rate"] - 0.6) < 0.01

    def test_win_rate_calculation(self, analyzer):
        """Test win rate calculation."""
        base_date = datetime(2024, 1, 1)

        # 3 wins out of 10 races
        data = {
            "date": [base_date + timedelta(days=14 * i) for i in range(10)],
            "driver_id": ["driver_1"] * 10,
            "position": [1, 2, 3, 1, 5, 1, 2, 3, 10, 15],
            "points": [25, 18, 15, 25, 10, 25, 18, 15, 0, 0],
            "season": ["2024"] * 10,
        }
        df = pd.DataFrame(data)

        trend = analyzer.analyze_driver_points_trend(df, "driver_1")

        # Win rate = 3/10 = 0.3
        assert abs(trend["win_rate"] - 0.3) < 0.01


class TestPerformanceMetricsIntegration:
    """Integration tests for performance metrics."""

    @pytest.fixture
    def multi_season_data(self) -> pd.DataFrame:
        """Create multi-season race results."""
        base_date = datetime(2022, 1, 1)
        data_list = []

        for season_year in [2022, 2023, 2024]:
            drivers = ["driver_1", "driver_2", "driver_3"]
            for race_num in range(20):
                for i, driver in enumerate(drivers):
                    position = (i + race_num + (season_year - 2022) * 5) % 3 + 1
                    points = {1: 25, 2: 18, 3: 15}.get(position, 0)

                    data_list.append({
                        "date": base_date + timedelta(days=365 * (season_year - 2022) + 14 * race_num),
                        "driver_id": driver,
                        "position": position,
                        "points": points,
                        "season": str(season_year),
                    })

        return pd.DataFrame(data_list)

    def test_multi_season_analysis(self, multi_season_data):
        """Test analysis across multiple seasons."""
        analyzer = ChampionshipPointsAnalyzer()

        for driver in ["driver_1", "driver_2", "driver_3"]:
            trend = analyzer.analyze_driver_points_trend(multi_season_data, driver)
            assert trend["races_count"] == 60  # 20 races × 3 seasons

    def test_season_specific_analysis(self, multi_season_data):
        """Test season-specific analysis filters correctly."""
        for season in ["2022", "2023", "2024"]:
            analyzer = ChampionshipPointsAnalyzer(season=season)
            trend = analyzer.analyze_driver_points_trend(multi_season_data, "driver_1")
            assert trend["races_count"] == 20  # 20 races per season
