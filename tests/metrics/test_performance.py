"""Tests for performance metrics calculators."""

import pandas as pd
import pytest

from f1_predict.metrics.performance import (
    ChampionshipPointsAnalyzer,
    DNFReliabilityAnalyzer,
    PerformanceMetricsCalculator,
    QualifyingAnalyzer,
    TeamCircuitAnalyzer,
    TeammateComparisonAnalyzer,
)


@pytest.fixture
def sample_race_results():
    """Create sample race results for testing."""
    return pd.DataFrame(
        {
            "season": ["2024"] * 10,
            "round": ["1", "2", "3", "4", "5", "1", "2", "3", "4", "5"],
            "driver_id": ["hamilton"] * 5 + ["verstappen"] * 5,
            "constructor_id": ["mercedes"] * 5 + ["red_bull"] * 5,
            "circuit_id": ["bahrain", "saudi", "australia", "japan", "china"] * 2,
            "position": [3, 2, 4, 1, 2, 1, 1, 2, 3, 1],
            "points": [15.0, 18.0, 12.0, 25.0, 18.0, 25.0, 25.0, 18.0, 15.0, 25.0],
            "status_id": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # All finished
            "date": pd.to_datetime(
                [
                    "2024-03-02",
                    "2024-03-09",
                    "2024-03-24",
                    "2024-04-07",
                    "2024-04-21",
                ]
                * 2
            ),
        }
    )


@pytest.fixture
def sample_qualifying_results():
    """Create sample qualifying results for testing."""
    return pd.DataFrame(
        {
            "season": ["2024"] * 10,
            "round": ["1", "2", "3", "4", "5", "1", "2", "3", "4", "5"],
            "driver_id": ["hamilton"] * 5 + ["verstappen"] * 5,
            "position": [4, 3, 5, 2, 3, 1, 1, 1, 2, 1],
        }
    )


@pytest.fixture
def sample_results_with_dnf():
    """Create sample race results with DNFs."""
    return pd.DataFrame(
        {
            "season": ["2024"] * 8,
            "round": ["1", "2", "3", "4", "1", "2", "3", "4"],
            "driver_id": ["alonso"] * 4 + ["perez"] * 4,
            "constructor_id": ["aston_martin"] * 4 + ["red_bull"] * 4,
            "circuit_id": ["bahrain", "saudi", "australia", "japan"] * 2,
            "position": [5, 20, 7, 4, 2, 20, 3, 20],
            "points": [10.0, 0.0, 6.0, 12.0, 18.0, 0.0, 15.0, 0.0],
            "status_id": [1, 5, 1, 1, 1, 4, 1, 3],  # Mix of finishes and DNFs
            "date": pd.to_datetime(
                ["2024-03-02", "2024-03-09", "2024-03-24", "2024-04-07"] * 2
            ),
        }
    )


class TestChampionshipPointsAnalyzer:
    """Tests for ChampionshipPointsAnalyzer."""

    def test_analyze_driver_points_trend(self, sample_race_results):
        """Test driver points trend analysis."""
        analyzer = ChampionshipPointsAnalyzer()
        metrics = analyzer.analyze_driver_points_trend(sample_race_results, "hamilton")

        assert "total_points" in metrics
        assert "avg_points_per_race" in metrics
        assert "points_trend" in metrics
        assert "podium_rate" in metrics
        assert "win_rate" in metrics
        assert "consistency_score" in metrics

        assert metrics["total_points"] == 88.0
        assert metrics["races_count"] == 5
        assert 0 <= metrics["podium_rate"] <= 1
        assert 0 <= metrics["win_rate"] <= 1

    def test_analyze_driver_no_results(self, sample_race_results):
        """Test analysis with no results."""
        analyzer = ChampionshipPointsAnalyzer()
        metrics = analyzer.analyze_driver_points_trend(
            sample_race_results, "nonexistent"
        )

        assert metrics["total_points"] == 0.0
        assert metrics["avg_points_per_race"] == 0.0

    def test_get_championship_standings(self, sample_race_results):
        """Test championship standings calculation."""
        analyzer = ChampionshipPointsAnalyzer()
        standings = analyzer.get_championship_standings(sample_race_results)

        assert len(standings) == 2
        assert "driver_id" in standings.columns
        assert "points" in standings.columns
        assert "position" in standings.columns

        # Verstappen should be ahead (108 points vs 88)
        assert standings.iloc[0]["driver_id"] == "verstappen"
        assert standings.iloc[0]["position"] == 1

    def test_season_specific_analysis(self, sample_race_results):
        """Test season-specific analysis."""
        analyzer = ChampionshipPointsAnalyzer(season="2024")
        metrics = analyzer.analyze_driver_points_trend(sample_race_results, "hamilton")

        assert metrics["races_count"] == 5

        # Non-existent season should return no results
        analyzer_wrong = ChampionshipPointsAnalyzer(season="2023")
        metrics_wrong = analyzer_wrong.analyze_driver_points_trend(
            sample_race_results, "hamilton"
        )
        assert metrics_wrong["races_count"] == 0


class TestTeamCircuitAnalyzer:
    """Tests for TeamCircuitAnalyzer."""

    def test_analyze_team_at_circuit(self, sample_race_results):
        """Test team performance at specific circuit."""
        analyzer = TeamCircuitAnalyzer()
        metrics = analyzer.analyze_team_at_circuit(
            sample_race_results, "mercedes", "bahrain"
        )

        assert "races_count" in metrics
        assert "avg_position" in metrics
        assert "best_position" in metrics
        assert "total_points" in metrics
        assert "podium_rate" in metrics
        assert "dnf_rate" in metrics

        assert metrics["races_count"] == 1
        assert metrics["best_position"] == 3

    def test_analyze_team_no_results(self, sample_race_results):
        """Test analysis with no results at circuit."""
        analyzer = TeamCircuitAnalyzer()
        metrics = analyzer.analyze_team_at_circuit(
            sample_race_results, "mercedes", "monaco"
        )

        assert metrics["races_count"] == 0
        assert metrics["total_points"] == 0.0

    def test_get_best_circuits(self, sample_race_results):
        """Test finding best circuits for a team."""
        analyzer = TeamCircuitAnalyzer()
        best_circuits = analyzer.get_best_circuits_for_team(
            sample_race_results, "mercedes", top_n=3
        )

        assert len(best_circuits) <= 3
        assert "circuit_id" in best_circuits.columns
        assert "avg_points" in best_circuits.columns

        # Should be sorted by avg_points descending
        if len(best_circuits) > 1:
            assert (
                best_circuits.iloc[0]["avg_points"]
                >= best_circuits.iloc[1]["avg_points"]
            )


class TestQualifyingAnalyzer:
    """Tests for QualifyingAnalyzer."""

    def test_analyze_driver_qualifying(
        self, sample_race_results, sample_qualifying_results
    ):
        """Test qualifying vs race analysis."""
        analyzer = QualifyingAnalyzer()
        metrics = analyzer.analyze_driver_qualifying(
            sample_race_results, sample_qualifying_results, "hamilton"
        )

        assert "avg_quali_position" in metrics
        assert "avg_race_position" in metrics
        assert "avg_position_change" in metrics
        assert "positions_gained_rate" in metrics
        assert "overtaking_score" in metrics

        assert 0 <= metrics["overtaking_score"] <= 100
        assert metrics["races_analyzed"] == 5

    def test_qualifying_with_position_gains(
        self, sample_race_results, sample_qualifying_results
    ):
        """Test qualifying analysis with position gains."""
        analyzer = QualifyingAnalyzer()
        hamilton_metrics = analyzer.analyze_driver_qualifying(
            sample_race_results, sample_qualifying_results, "hamilton"
        )

        # Hamilton qualifies worse but finishes better (position_change > 0)
        assert hamilton_metrics["avg_position_change"] > 0
        assert hamilton_metrics["overtaking_score"] > 50


class TestDNFReliabilityAnalyzer:
    """Tests for DNFReliabilityAnalyzer."""

    def test_analyze_driver_reliability(self, sample_race_results):
        """Test driver reliability analysis."""
        analyzer = DNFReliabilityAnalyzer()
        metrics = analyzer.analyze_driver_reliability(sample_race_results, "hamilton")

        assert "finish_rate" in metrics
        assert "dnf_rate" in metrics
        assert "mechanical_dnf_rate" in metrics
        assert "accident_dnf_rate" in metrics
        assert "reliability_score" in metrics

        assert 0 <= metrics["finish_rate"] <= 1
        assert 0 <= metrics["reliability_score"] <= 100

    def test_reliability_with_dnf(self, sample_results_with_dnf):
        """Test reliability with DNFs."""
        analyzer = DNFReliabilityAnalyzer()

        alonso_metrics = analyzer.analyze_driver_reliability(
            sample_results_with_dnf, "alonso"
        )
        perez_metrics = analyzer.analyze_driver_reliability(
            sample_results_with_dnf, "perez"
        )

        # Alonso has 1 DNF, Perez has 3 DNFs
        assert alonso_metrics["finish_rate"] > perez_metrics["finish_rate"]
        assert alonso_metrics["reliability_score"] > perez_metrics["reliability_score"]

    def test_analyze_team_reliability(self, sample_results_with_dnf):
        """Test team reliability analysis."""
        analyzer = DNFReliabilityAnalyzer()
        metrics = analyzer.analyze_team_reliability(
            sample_results_with_dnf, "aston_martin"
        )

        assert "entries_count" in metrics
        assert "finish_rate" in metrics
        assert "mechanical_dnf_rate" in metrics
        assert "reliability_score" in metrics

        assert metrics["entries_count"] == 4


class TestTeammateComparisonAnalyzer:
    """Tests for TeammateComparisonAnalyzer."""

    def test_compare_teammates(self, sample_race_results, sample_qualifying_results):
        """Test teammate comparison."""
        # Add teammate results
        race_results = pd.concat(
            [
                sample_race_results,
                pd.DataFrame(
                    {
                        "season": ["2024"] * 5,
                        "round": ["1", "2", "3", "4", "5"],
                        "driver_id": ["russell"] * 5,
                        "constructor_id": ["mercedes"] * 5,
                        "circuit_id": [
                            "bahrain",
                            "saudi",
                            "australia",
                            "japan",
                            "china",
                        ],
                        "position": [5, 4, 6, 3, 4],
                        "points": [10.0, 12.0, 8.0, 15.0, 12.0],
                        "status_id": [1, 1, 1, 1, 1],
                        "date": pd.to_datetime(
                            [
                                "2024-03-02",
                                "2024-03-09",
                                "2024-03-24",
                                "2024-04-07",
                                "2024-04-21",
                            ]
                        ),
                    }
                ),
            ]
        )

        quali_results = pd.concat(
            [
                sample_qualifying_results,
                pd.DataFrame(
                    {
                        "season": ["2024"] * 5,
                        "round": ["1", "2", "3", "4", "5"],
                        "driver_id": ["russell"] * 5,
                        "position": [5, 4, 6, 3, 4],
                    }
                ),
            ]
        )

        analyzer = TeammateComparisonAnalyzer()
        metrics = analyzer.compare_teammates(
            race_results, quali_results, "hamilton", "russell"
        )

        assert "common_races" in metrics
        assert "driver1_wins" in metrics
        assert "driver2_wins" in metrics
        assert "avg_position_diff" in metrics
        assert "points_diff" in metrics
        assert "common_qualifying" in metrics

        assert metrics["common_races"] == 5
        assert (
            metrics["driver1_wins"] + metrics["driver2_wins"] <= metrics["common_races"]
        )

    def test_compare_no_common_races(
        self, sample_race_results, sample_qualifying_results
    ):
        """Test comparison with no common races."""
        analyzer = TeammateComparisonAnalyzer()
        metrics = analyzer.compare_teammates(
            sample_race_results, sample_qualifying_results, "hamilton", "alonso"
        )

        assert metrics["common_races"] == 0


class TestPerformanceMetricsCalculator:
    """Tests for PerformanceMetricsCalculator."""

    def test_generate_driver_report(
        self, sample_race_results, sample_qualifying_results
    ):
        """Test comprehensive driver report generation."""
        calculator = PerformanceMetricsCalculator()
        report = calculator.generate_driver_report(
            sample_race_results, sample_qualifying_results, "hamilton"
        )

        assert "driver_id" in report
        assert "championship_points" in report
        assert "qualifying_performance" in report
        assert "reliability" in report

        assert report["driver_id"] == "hamilton"

    def test_generate_team_report(self, sample_race_results):
        """Test comprehensive team report generation."""
        calculator = PerformanceMetricsCalculator()
        report = calculator.generate_team_report(sample_race_results, "mercedes")

        assert "constructor_id" in report
        assert "reliability" in report
        assert "best_circuits" in report

        assert report["constructor_id"] == "mercedes"

    def test_season_specific_reports(
        self, sample_race_results, sample_qualifying_results
    ):
        """Test season-specific report generation."""
        calculator = PerformanceMetricsCalculator(season="2024")
        report = calculator.generate_driver_report(
            sample_race_results, sample_qualifying_results, "hamilton"
        )

        assert report["season"] == "2024"
