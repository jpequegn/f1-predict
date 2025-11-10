"""Unit tests for logistic regression model.

Tests cover:
- Podium, points, and win prediction
- Probability calibration
- Model fitting and prediction
"""

import pandas as pd
import pytest
import numpy as np

from f1_predict.models.logistic import LogisticRacePredictor


class TestLogisticRacePredictor:
    """Tests for LogisticRacePredictor class."""

    @pytest.fixture
    def sample_train_data(self):
        """Create sample training data."""
        np.random.seed(42)
        features = pd.DataFrame({
            "driver_id": [f"driver_{i}" for i in range(50)],
            "qualifying_position": np.random.randint(1, 21, 50),
            "driver_form_score": np.random.randint(40, 100, 50),
            "team_reliability_score": np.random.randint(50, 100, 50),
            "circuit_performance_score": np.random.randint(40, 100, 50),
        })
        positions = np.where(features["qualifying_position"] <= 5,
                           np.random.randint(1, 4, 50),
                           np.random.randint(8, 20, 50))
        race_results = pd.DataFrame({
            "position": positions,
        })
        return features, race_results

    @pytest.fixture
    def podium_predictor(self) -> LogisticRacePredictor:
        """Create a podium prediction model."""
        return LogisticRacePredictor(target="podium", random_state=42)

    @pytest.fixture
    def points_predictor(self) -> LogisticRacePredictor:
        """Create a points prediction model."""
        return LogisticRacePredictor(target="points", random_state=42)

    @pytest.fixture
    def win_predictor(self) -> LogisticRacePredictor:
        """Create a win prediction model."""
        return LogisticRacePredictor(target="win", random_state=42)

    def test_initialization_podium(self, podium_predictor):
        """Test podium predictor initialization."""
        assert podium_predictor.target == "podium"
        assert podium_predictor.random_state == 42
        assert podium_predictor.max_iter == 1000
        assert not podium_predictor.is_fitted

    def test_initialization_points(self, points_predictor):
        """Test points predictor initialization."""
        assert points_predictor.target == "points"

    def test_initialization_win(self, win_predictor):
        """Test win predictor initialization."""
        assert win_predictor.target == "win"

    def test_initialization_invalid_target(self):
        """Test that invalid target raises ValueError."""
        with pytest.raises(ValueError, match="Invalid target"):
            LogisticRacePredictor(target="invalid")

    def test_fit_model(self, podium_predictor, sample_train_data):
        """Test model fitting."""
        features, race_results = sample_train_data

        result = podium_predictor.fit(features, race_results)

        assert podium_predictor.is_fitted
        assert len(podium_predictor.feature_names) > 0
        assert result is podium_predictor  # Method chaining

    def test_predict_proba_after_fit(self, podium_predictor, sample_train_data):
        """Test probability predictions after fitting."""
        features, race_results = sample_train_data

        podium_predictor.fit(features, race_results)
        proba = podium_predictor.predict_proba(features.iloc[:5])

        assert proba is not None
        assert len(proba) == 5
        assert all(0 <= p <= 1 for p in proba)

    def test_predict_before_fit(self, podium_predictor, sample_train_data):
        """Test that predict before fit raises error."""
        features, _ = sample_train_data

        with pytest.raises(ValueError):
            podium_predictor.predict_proba(features.iloc[:5])

    def test_fit_with_different_targets(self, sample_train_data):
        """Test fitting with different target types."""
        features, race_results = sample_train_data

        for target in ["podium", "points", "win"]:
            predictor = LogisticRacePredictor(target=target)
            predictor.fit(features, race_results)
            assert predictor.is_fitted

    def test_feature_scaling(self, podium_predictor, sample_train_data):
        """Test that features are scaled."""
        features, race_results = sample_train_data

        podium_predictor.fit(features, race_results)

        # Scaler should be fitted
        assert podium_predictor.scaler is not None

    def test_probability_bounds(self, podium_predictor, sample_train_data):
        """Test that probabilities are in valid range."""
        features, race_results = sample_train_data

        podium_predictor.fit(features, race_results)
        proba = podium_predictor.predict_proba(features)

        assert all(0 <= p <= 1 for p in proba)

    def test_consistent_predictions(self, podium_predictor, sample_train_data):
        """Test that predictions are consistent."""
        features, race_results = sample_train_data

        podium_predictor.fit(features, race_results)

        proba1 = podium_predictor.predict_proba(features.iloc[:5])
        proba2 = podium_predictor.predict_proba(features.iloc[:5])

        np.testing.assert_array_equal(proba1, proba2)

    def test_different_max_iter(self):
        """Test initialization with different max_iter."""
        predictor = LogisticRacePredictor(max_iter=500)
        assert predictor.max_iter == 500

    def test_empty_features_fit(self, podium_predictor):
        """Test with empty feature data."""
        empty_features = pd.DataFrame({
            "qualifying_position": [],
            "driver_form_score": [],
        })
        empty_results = pd.DataFrame({"position": []})

        with pytest.raises(ValueError):
            podium_predictor.fit(empty_features, empty_results)

    def test_single_sample_prediction(self, podium_predictor, sample_train_data):
        """Test prediction for single sample."""
        features, race_results = sample_train_data

        podium_predictor.fit(features, race_results)
        proba = podium_predictor.predict_proba(features.iloc[[0]])

        assert len(proba) == 1
        assert 0 <= proba[0] <= 1

    def test_all_targets_can_fit(self, sample_train_data):
        """Test that all target types can be fitted and predict."""
        features, race_results = sample_train_data

        for target in ["podium", "points", "win"]:
            predictor = LogisticRacePredictor(target=target)
            predictor.fit(features, race_results)
            proba = predictor.predict_proba(features.iloc[:5])

            assert len(proba) == 5
            assert all(0 <= p <= 1 for p in proba)

    def test_mismatched_data_length(self, podium_predictor):
        """Test that mismatched feature/result lengths raise error."""
        features = pd.DataFrame({
            "qualifying_position": [1, 2, 3],
            "driver_form_score": [75, 80, 85],
            "team_reliability_score": [70, 75, 80],
            "circuit_performance_score": [75, 80, 85],
        })
        race_results = pd.DataFrame({"position": [1, 2]})  # Mismatched

        with pytest.raises(ValueError):
            podium_predictor.fit(features, race_results)
