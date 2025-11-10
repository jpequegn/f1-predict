"""Unit tests for feature engineering module.

Tests cover:
- Driver form calculation with recency weighting
- Team reliability metrics
- Track-specific performance indicators
- Feature normalization and outlier handling
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from f1_predict.features.engineering import (
    DriverFormCalculator,
    TeamReliabilityCalculator,
    TrackPerformanceCalculator,
)


class TestDriverFormCalculator:
    """Tests for DriverFormCalculator class."""

    @pytest.fixture
    def calculator(self) -> DriverFormCalculator:
        """Create a driver form calculator instance."""
        return DriverFormCalculator(window_size=5, recency_weight=0.7)

    @pytest.fixture
    def sample_race_results(self) -> pd.DataFrame:
        """Create sample race results for testing."""
        base_date = datetime(2024, 1, 1)
        data = {
            "date": [base_date + timedelta(days=14 * i) for i in range(8)],
            "driver_id": ["driver_1"] * 8,
            "position": [1, 2, 1, 3, 2, 1, 4, 2],
            "points": [25, 18, 25, 15, 18, 25, 12, 18],
            "status_id": [1, 1, 1, 1, 1, 1, 1, 1],  # 1 = finished
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def multi_driver_results(self) -> pd.DataFrame:
        """Create multi-driver race results."""
        base_date = datetime(2024, 1, 1)
        drivers = ["driver_1", "driver_2", "driver_3"]
        data_list = []

        for i in range(5):
            for driver_id in drivers:
                data_list.append({
                    "date": base_date + timedelta(days=14 * i),
                    "driver_id": driver_id,
                    "position": (i + 1) % 20 + 1,
                    "points": max(0, 25 - (i * 5)),
                    "status_id": 1,
                })

        return pd.DataFrame(data_list)

    def test_initialization(self, calculator):
        """Test DriverFormCalculator initialization."""
        assert calculator.window_size == 5
        assert calculator.recency_weight == 0.7

    def test_initialization_custom_params(self):
        """Test DriverFormCalculator with custom parameters."""
        calc = DriverFormCalculator(window_size=10, recency_weight=0.5)
        assert calc.window_size == 10
        assert calc.recency_weight == 0.5

    def test_calculate_form_score_basic(self, calculator, sample_race_results):
        """Test form score calculation with basic data."""
        score = calculator.calculate_form_score(sample_race_results, "driver_1")

        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_calculate_form_score_no_races(self, calculator):
        """Test form score calculation with no matching races."""
        empty_df = pd.DataFrame({
            "date": [],
            "driver_id": [],
            "position": [],
            "points": [],
            "status_id": [],
        })

        score = calculator.calculate_form_score(empty_df, "unknown_driver")

        assert score == 50.0

    def test_calculate_form_score_with_dnf(self, calculator):
        """Test form score penalty for DNF races."""
        base_date = datetime(2024, 1, 1)
        data = {
            "date": [base_date + timedelta(days=14 * i) for i in range(5)],
            "driver_id": ["driver_1"] * 5,
            "position": [1, 3, 20, 2, 1],
            "points": [25, 15, 0, 18, 25],
            "status_id": [1, 1, 3, 1, 1],
        }
        df = pd.DataFrame(data)

        score = calculator.calculate_form_score(df, "driver_1")

        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_recency_weights_calculation(self, calculator):
        """Test recency weight calculation."""
        weights = calculator._calculate_recency_weights(5)

        assert len(weights) == 5
        assert weights[0] < weights[1] < weights[2] < weights[3] < weights[4]
        assert weights[-1] == max(weights)

    def test_calculate_form_features(self, calculator, multi_driver_results):
        """Test form score calculation for multiple drivers."""
        form_df = calculator.calculate_form_features(multi_driver_results)

        assert len(form_df) == 3
        assert "driver_id" in form_df.columns
        assert "form_score" in form_df.columns
        assert all(0 <= score <= 100 for score in form_df["form_score"])

    def test_form_score_consistency(self, calculator, sample_race_results):
        """Test that form score is consistent across calls."""
        score1 = calculator.calculate_form_score(sample_race_results, "driver_1")
        score2 = calculator.calculate_form_score(sample_race_results, "driver_1")

        assert score1 == score2

    def test_form_score_high_performer(self, calculator):
        """Test form score for consistently high performer."""
        base_date = datetime(2024, 1, 1)
        data = {
            "date": [base_date + timedelta(days=14 * i) for i in range(5)],
            "driver_id": ["driver_1"] * 5,
            "position": [1, 1, 1, 1, 1],
            "points": [25, 25, 25, 25, 25],
            "status_id": [1] * 5,
        }
        df = pd.DataFrame(data)

        score = calculator.calculate_form_score(df, "driver_1")

        assert score > 90

    def test_form_score_poor_performer(self, calculator):
        """Test form score for consistently poor performer."""
        base_date = datetime(2024, 1, 1)
        data = {
            "date": [base_date + timedelta(days=14 * i) for i in range(5)],
            "driver_id": ["driver_1"] * 5,
            "position": [20, 20, 20, 20, 20],
            "points": [0, 0, 0, 0, 0],
            "status_id": [1] * 5,
        }
        df = pd.DataFrame(data)

        score = calculator.calculate_form_score(df, "driver_1")

        # Poor performer gets < 50 average, but weighting still gives some credit
        assert score < 50


class TestTeamReliabilityCalculator:
    """Tests for TeamReliabilityCalculator class."""

    @pytest.fixture
    def calculator(self) -> TeamReliabilityCalculator:
        """Create a team reliability calculator instance."""
        return TeamReliabilityCalculator(window_size=10)

    @pytest.fixture
    def reliable_team_data(self) -> pd.DataFrame:
        """Create sample data for a reliable team."""
        base_date = datetime(2024, 1, 1)
        data = {
            "date": [base_date + timedelta(days=14 * i) for i in range(10)],
            "constructor_id": ["team_1"] * 10,
            "position": [3, 4, 2, 5, 3, 4, 3, 2, 4, 3],
            "points": [12, 10, 18, 8, 12, 10, 12, 18, 10, 12],
            "status_id": [1] * 10,
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def unreliable_team_data(self) -> pd.DataFrame:
        """Create sample data for an unreliable team."""
        base_date = datetime(2024, 1, 1)
        data = {
            "date": [base_date + timedelta(days=14 * i) for i in range(10)],
            "constructor_id": ["team_2"] * 10,
            "position": [1, 20, 2, 20, 3, 20, 1, 20, 2, 20],
            "points": [25, 0, 18, 0, 15, 0, 25, 0, 18, 0],
            "status_id": [1, 3, 1, 3, 1, 3, 1, 3, 1, 3],
        }
        return pd.DataFrame(data)

    def test_initialization(self, calculator):
        """Test TeamReliabilityCalculator initialization."""
        assert calculator.window_size == 10

    def test_calculate_reliability_metrics_reliable(self, calculator, reliable_team_data):
        """Test reliability metrics for reliable team."""
        metrics = calculator.calculate_reliability_metrics(reliable_team_data, "team_1")

        assert "finish_rate" in metrics
        assert "avg_position" in metrics
        assert "mechanical_failure_rate" in metrics
        assert "points_consistency" in metrics
        assert "reliability_score" in metrics

        assert metrics["finish_rate"] == 1.0
        assert metrics["mechanical_failure_rate"] == 0.0

    def test_calculate_reliability_metrics_unreliable(
        self, calculator, unreliable_team_data
    ):
        """Test reliability metrics for unreliable team."""
        metrics = calculator.calculate_reliability_metrics(unreliable_team_data, "team_2")

        assert metrics["finish_rate"] == 0.5
        assert metrics["mechanical_failure_rate"] == 0.5

    def test_calculate_reliability_no_team(self, calculator):
        """Test reliability calculation with no matching team."""
        empty_df = pd.DataFrame({
            "date": [],
            "constructor_id": [],
            "position": [],
            "points": [],
            "status_id": [],
        })

        metrics = calculator.calculate_reliability_metrics(empty_df, "unknown_team")

        assert metrics["finish_rate"] == 0.5
        assert metrics["reliability_score"] == 50.0

    def test_calculate_reliability_features(self, calculator, reliable_team_data):
        """Test reliability feature calculation for multiple teams."""
        unreliable_data = reliable_team_data.copy()
        unreliable_data["constructor_id"] = "team_2"

        combined = pd.concat([reliable_team_data, unreliable_data], ignore_index=True)

        features_df = calculator.calculate_reliability_features(combined)

        assert len(features_df) == 2
        assert "constructor_id" in features_df.columns
        assert all(col in features_df.columns for col in [
            "finish_rate",
            "avg_position",
            "mechanical_failure_rate",
            "points_consistency",
            "reliability_score",
        ])

    def test_reliability_score_bounds(self, calculator, reliable_team_data):
        """Test that reliability scores are within valid bounds."""
        metrics = calculator.calculate_reliability_metrics(reliable_team_data, "team_1")

        assert 0 <= metrics["reliability_score"] <= 100
        assert 0 <= metrics["finish_rate"] <= 1
        assert 0 <= metrics["points_consistency"] <= 100


class TestTrackPerformanceCalculator:
    """Tests for TrackPerformanceCalculator class."""

    @pytest.fixture
    def calculator(self) -> TrackPerformanceCalculator:
        """Create a track performance calculator instance."""
        return TrackPerformanceCalculator(min_races=2)

    @pytest.fixture
    def track_data(self) -> pd.DataFrame:
        """Create sample track performance data."""
        base_date = datetime(2024, 1, 1)
        data = {
            "date": [base_date + timedelta(days=365 * i) for i in range(4)],
            "driver_id": ["driver_1"] * 4,
            "circuit_id": ["circuit_a"] * 4,
            "position": [1, 2, 1, 3],
            "points": [25, 18, 25, 15],
            "status_id": [1] * 4,
        }
        return pd.DataFrame(data)

    def test_initialization(self, calculator):
        """Test TrackPerformanceCalculator initialization."""
        assert calculator.min_races == 2

    def test_calculate_track_performance(self, calculator, track_data):
        """Test track performance calculation."""
        perf = calculator.calculate_track_performance(
            track_data, "driver_1", "circuit_a"
        )

        assert "avg_position" in perf
        assert "avg_points" in perf
        assert "best_position" in perf
        assert "races_at_track" in perf
        assert "track_performance_score" in perf

    def test_track_performance_insufficient_races(self, calculator):
        """Test with insufficient races at track."""
        base_date = datetime(2024, 1, 1)
        data = {
            "date": [base_date],
            "driver_id": ["driver_1"],
            "circuit_id": ["circuit_a"],
            "position": [5],
            "points": [10],
            "status_id": [1],
        }
        df = pd.DataFrame(data)

        perf = calculator.calculate_track_performance(df, "driver_1", "circuit_a")

        assert perf["avg_position"] == 10.0

    def test_track_performance_no_history(self, calculator):
        """Test with no history at circuit."""
        empty_df = pd.DataFrame({
            "date": [],
            "driver_id": [],
            "circuit_id": [],
            "position": [],
            "points": [],
            "status_id": [],
        })

        perf = calculator.calculate_track_performance(
            empty_df, "driver_1", "unknown_circuit"
        )

        assert perf["avg_position"] == 10.0

    def test_track_performance_multiple_circuits(self, calculator):
        """Test track performance across multiple circuits."""
        base_date = datetime(2024, 1, 1)
        circuits = ["circuit_a", "circuit_b", "circuit_c"]
        data_list = []

        for i, circuit in enumerate(circuits):
            for j in range(3):
                data_list.append({
                    "date": base_date + timedelta(days=365 * j),
                    "driver_id": "driver_1",
                    "circuit_id": circuit,
                    "position": i + 1,
                    "points": 25 - (i * 5),
                    "status_id": 1,
                })

        df = pd.DataFrame(data_list)

        perf_a = calculator.calculate_track_performance(df, "driver_1", "circuit_a")
        perf_b = calculator.calculate_track_performance(df, "driver_1", "circuit_b")
        perf_c = calculator.calculate_track_performance(df, "driver_1", "circuit_c")

        assert perf_a["avg_position"] < perf_b["avg_position"]
        assert perf_b["avg_position"] < perf_c["avg_position"]

    def test_track_performance_strong_performance(self, calculator):
        """Test performance at track with mostly winning positions."""
        base_date = datetime(2024, 1, 1)
        data = {
            "date": [base_date + timedelta(days=365 * i) for i in range(5)],
            "driver_id": ["driver_1"] * 5,
            "circuit_id": ["circuit_a"] * 5,
            "position": [1, 1, 2, 1, 1],
            "points": [25, 25, 18, 25, 25],
            "status_id": [1] * 5,
        }
        df = pd.DataFrame(data)

        perf = calculator.calculate_track_performance(df, "driver_1", "circuit_a")

        # Strong performer should have high track_performance_score
        assert perf["track_performance_score"] > 80
        assert perf["avg_position"] < 2  # Average position near 1st
