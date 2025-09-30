"""Tests for feature engineering calculators."""

from datetime import datetime

import pandas as pd
import pytest

from f1_predict.features.engineering import (
    DriverFormCalculator,
    FeatureEngineer,
    QualifyingRaceGapCalculator,
    TeamReliabilityCalculator,
    TrackPerformanceCalculator,
    WeatherFeatureCalculator,
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
def sample_race_results_with_dnf():
    """Create sample race results with DNFs for testing."""
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


@pytest.fixture
def sample_qualifying_results():
    """Create sample qualifying results for testing."""
    return pd.DataFrame(
        {
            "season": ["2024"] * 10,
            "round": ["1", "2", "3", "4", "5", "1", "2", "3", "4", "5"],
            "driver_id": ["hamilton"] * 5 + ["verstappen"] * 5,
            "position": [4, 3, 5, 2, 3, 1, 1, 1, 2, 1],
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


class TestDriverFormCalculator:
    """Tests for DriverFormCalculator."""

    def test_calculate_form_score(self, sample_race_results):
        """Test basic form score calculation."""
        calculator = DriverFormCalculator(window_size=5)
        score = calculator.calculate_form_score(sample_race_results, "hamilton")

        assert 0 <= score <= 100
        assert isinstance(score, float)

    def test_form_score_with_limited_data(self, sample_race_results):
        """Test form score with limited race data."""
        calculator = DriverFormCalculator(window_size=10)  # More than available
        score = calculator.calculate_form_score(sample_race_results.head(2), "hamilton")

        assert 0 <= score <= 100

    def test_form_score_no_data(self, sample_race_results):
        """Test form score with no race data."""
        calculator = DriverFormCalculator()
        score = calculator.calculate_form_score(sample_race_results, "nonexistent")

        assert score == 50.0  # Should return neutral score

    def test_form_score_with_dnf(self, sample_race_results_with_dnf):
        """Test form score calculation with DNFs."""
        calculator = DriverFormCalculator(window_size=4)

        alonso_score = calculator.calculate_form_score(
            sample_race_results_with_dnf, "alonso"
        )
        perez_score = calculator.calculate_form_score(
            sample_race_results_with_dnf, "perez"
        )

        # Alonso with 1 DNF should have higher score than Perez with 3 DNFs
        assert alonso_score > perez_score

    def test_calculate_form_features(self, sample_race_results):
        """Test calculating form features for all drivers."""
        calculator = DriverFormCalculator()
        features = calculator.calculate_form_features(sample_race_results)

        assert len(features) == 2  # hamilton and verstappen
        assert "driver_id" in features.columns
        assert "form_score" in features.columns
        assert all(features["form_score"].between(0, 100))

    def test_form_up_to_date(self, sample_race_results):
        """Test form calculation up to a specific date."""
        calculator = DriverFormCalculator()
        up_to_date = datetime(2024, 3, 15)

        score = calculator.calculate_form_score(
            sample_race_results, "hamilton", up_to_date
        )

        # Should only consider first 2 races
        assert 0 <= score <= 100

    def test_recency_weights(self):
        """Test recency weight calculation."""
        calculator = DriverFormCalculator(recency_weight=0.7)
        weights = calculator._calculate_recency_weights(5)

        assert len(weights) == 5
        assert weights[-1] > weights[0]  # More recent race has higher weight


class TestTeamReliabilityCalculator:
    """Tests for TeamReliabilityCalculator."""

    def test_calculate_reliability_metrics(self, sample_race_results):
        """Test basic reliability metrics calculation."""
        calculator = TeamReliabilityCalculator()
        metrics = calculator.calculate_reliability_metrics(
            sample_race_results, "mercedes"
        )

        assert "finish_rate" in metrics
        assert "avg_position" in metrics
        assert "mechanical_failure_rate" in metrics
        assert "points_consistency" in metrics
        assert "reliability_score" in metrics

        assert 0 <= metrics["finish_rate"] <= 1
        assert 0 <= metrics["reliability_score"] <= 100

    def test_reliability_with_dnf(self, sample_race_results_with_dnf):
        """Test reliability calculation with DNFs."""
        calculator = TeamReliabilityCalculator()

        aston_metrics = calculator.calculate_reliability_metrics(
            sample_race_results_with_dnf, "aston_martin"
        )
        red_bull_metrics = calculator.calculate_reliability_metrics(
            sample_race_results_with_dnf, "red_bull"
        )

        # Aston Martin (1 DNF) should have better reliability than Red Bull (3 DNFs)
        assert aston_metrics["finish_rate"] > red_bull_metrics["finish_rate"]
        assert (
            aston_metrics["mechanical_failure_rate"]
            < red_bull_metrics["mechanical_failure_rate"]
        )

    def test_calculate_reliability_features(self, sample_race_results):
        """Test calculating reliability features for all teams."""
        calculator = TeamReliabilityCalculator()
        features = calculator.calculate_reliability_features(sample_race_results)

        assert len(features) == 2  # mercedes and red_bull
        assert "constructor_id" in features.columns
        assert all(features["finish_rate"].between(0, 1))


class TestTrackPerformanceCalculator:
    """Tests for TrackPerformanceCalculator."""

    def test_calculate_track_performance(self, sample_race_results):
        """Test track-specific performance calculation."""
        calculator = TrackPerformanceCalculator(min_races=1)
        metrics = calculator.calculate_track_performance(
            sample_race_results, "hamilton", "bahrain"
        )

        assert "avg_position" in metrics
        assert "avg_points" in metrics
        assert "best_position" in metrics
        assert "races_at_track" in metrics
        assert "track_performance_score" in metrics

        assert metrics["races_at_track"] >= 1
        assert 0 <= metrics["track_performance_score"] <= 100

    def test_track_performance_insufficient_data(self, sample_race_results):
        """Test track performance with insufficient data."""
        calculator = TrackPerformanceCalculator(min_races=5)
        metrics = calculator.calculate_track_performance(
            sample_race_results, "hamilton", "bahrain"
        )

        # Should return default values
        assert metrics["track_performance_score"] == 50.0

    def test_calculate_track_features(self, sample_race_results):
        """Test calculating track features for all drivers."""
        calculator = TrackPerformanceCalculator(min_races=1)
        features = calculator.calculate_track_features(sample_race_results, "bahrain")

        assert len(features) == 2  # hamilton and verstappen
        assert "driver_id" in features.columns
        assert "circuit_id" in features.columns
        assert all(features["track_performance_score"].between(0, 100))


class TestQualifyingRaceGapCalculator:
    """Tests for QualifyingRaceGapCalculator."""

    def test_calculate_performance_gap(
        self, sample_race_results, sample_qualifying_results
    ):
        """Test qualifying vs race performance gap calculation."""
        calculator = QualifyingRaceGapCalculator()
        metrics = calculator.calculate_performance_gap(
            sample_race_results, sample_qualifying_results, "hamilton"
        )

        assert "avg_quali_position" in metrics
        assert "avg_race_position" in metrics
        assert "avg_position_gain" in metrics
        assert "position_gain_consistency" in metrics
        assert "racecraft_score" in metrics

        assert 0 <= metrics["racecraft_score"] <= 100

    def test_performance_gap_race_improver(
        self, sample_race_results, sample_qualifying_results
    ):
        """Test gap calculation for driver who improves in races."""
        calculator = QualifyingRaceGapCalculator()

        hamilton_metrics = calculator.calculate_performance_gap(
            sample_race_results, sample_qualifying_results, "hamilton"
        )

        # Hamilton qualifies worse but finishes better in races
        assert hamilton_metrics["avg_position_gain"] > 0  # Improves in race
        assert hamilton_metrics["racecraft_score"] > 50  # Good racecraft

    def test_calculate_gap_features(
        self, sample_race_results, sample_qualifying_results
    ):
        """Test calculating gap features for all drivers."""
        calculator = QualifyingRaceGapCalculator()
        features = calculator.calculate_gap_features(
            sample_race_results, sample_qualifying_results
        )

        assert len(features) == 2  # hamilton and verstappen
        assert "driver_id" in features.columns
        assert all(features["racecraft_score"].between(0, 100))


class TestWeatherFeatureCalculator:
    """Tests for WeatherFeatureCalculator."""

    def test_calculate_weather_features_placeholder(self, sample_race_results):
        """Test weather feature calculation with no weather data."""
        calculator = WeatherFeatureCalculator()
        features = calculator.calculate_weather_features(sample_race_results)

        assert "driver_id" in features.columns
        assert "wet_performance_score" in features.columns
        assert "variable_conditions_score" in features.columns
        assert "temperature_adaptation_score" in features.columns

        # Should return neutral scores (50.0) for all drivers
        assert all(features["wet_performance_score"] == 50.0)


class TestFeatureEngineer:
    """Tests for FeatureEngineer orchestrator."""

    def test_generate_features(self, sample_race_results, sample_qualifying_results):
        """Test complete feature generation."""
        engineer = FeatureEngineer()
        features = engineer.generate_features(
            sample_race_results, sample_qualifying_results
        )

        # Check that all feature types are present
        assert "driver_id" in features.columns
        assert "form_score" in features.columns
        assert "avg_quali_position" in features.columns
        assert "wet_performance_score" in features.columns

        assert len(features) == 2  # hamilton and verstappen

    def test_generate_features_with_circuit(
        self, sample_race_results, sample_qualifying_results
    ):
        """Test feature generation with track-specific features."""
        engineer = FeatureEngineer()
        features = engineer.generate_features(
            sample_race_results, sample_qualifying_results, circuit_id="bahrain"
        )

        # Should include track-specific features
        assert "track_performance_score" in features.columns
        assert len(features) == 2

    def test_generate_features_up_to_date(
        self, sample_race_results, sample_qualifying_results
    ):
        """Test feature generation up to specific date."""
        engineer = FeatureEngineer()
        up_to_date = datetime(2024, 3, 15)

        features = engineer.generate_features(
            sample_race_results, sample_qualifying_results, up_to_date=up_to_date
        )

        # Should only consider early races
        assert len(features) == 2

    def test_save_features_csv(
        self, sample_race_results, sample_qualifying_results, tmp_path
    ):
        """Test saving features to CSV."""
        engineer = FeatureEngineer()
        features = engineer.generate_features(
            sample_race_results, sample_qualifying_results
        )

        output_path = tmp_path / "features.csv"
        engineer.save_features(features, output_path)

        assert output_path.exists()

        # Verify file can be read back
        loaded = pd.read_csv(output_path)
        assert len(loaded) == len(features)

    def test_save_features_json(
        self, sample_race_results, sample_qualifying_results, tmp_path
    ):
        """Test saving features to JSON."""
        engineer = FeatureEngineer()
        features = engineer.generate_features(
            sample_race_results, sample_qualifying_results
        )

        output_path = tmp_path / "features.json"
        engineer.save_features(features, output_path)

        assert output_path.exists()

        # Verify file can be read back
        loaded = pd.read_json(output_path, orient="records")
        assert len(loaded) == len(features)

    def test_save_features_invalid_format(
        self, sample_race_results, sample_qualifying_results, tmp_path
    ):
        """Test saving features with invalid format raises error."""
        engineer = FeatureEngineer()
        features = engineer.generate_features(
            sample_race_results, sample_qualifying_results
        )

        output_path = tmp_path / "features.txt"

        with pytest.raises(ValueError, match="Unsupported file format"):
            engineer.save_features(features, output_path)
