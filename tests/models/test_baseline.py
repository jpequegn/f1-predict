"""Tests for rule-based prediction model."""

import pandas as pd
import pytest

from f1_predict.models.baseline import RuleBasedPredictor


@pytest.fixture
def sample_features():
    """Create sample feature data for testing."""
    return pd.DataFrame(
        {
            "driver_id": ["hamilton", "verstappen", "leclerc", "sainz", "perez"],
            "qualifying_position": [3, 1, 2, 5, 4],
            "driver_form_score": [85.0, 92.0, 78.0, 72.0, 80.0],
            "team_reliability_score": [90.0, 95.0, 85.0, 85.0, 92.0],
            "circuit_performance_score": [88.0, 90.0, 82.0, 80.0, 85.0],
        }
    )


class TestRuleBasedPredictor:
    """Tests for RuleBasedPredictor."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = RuleBasedPredictor()
        assert predictor.quali_weight == 0.6
        assert predictor.form_weight == 0.2
        assert predictor.reliability_weight == 0.1
        assert predictor.circuit_weight == 0.1

    def test_initialization_with_custom_weights(self):
        """Test predictor with custom weights."""
        predictor = RuleBasedPredictor(
            quali_weight=0.5, form_weight=0.3, reliability_weight=0.1, circuit_weight=0.1
        )
        assert predictor.quali_weight == 0.5
        assert predictor.form_weight == 0.3

    def test_initialization_invalid_weights(self):
        """Test that invalid weights raise error."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            RuleBasedPredictor(
                quali_weight=0.5, form_weight=0.3, reliability_weight=0.1, circuit_weight=0.2
            )

    def test_predict_basic(self, sample_features):
        """Test basic prediction functionality."""
        predictor = RuleBasedPredictor()
        predictions = predictor.predict(sample_features)

        assert len(predictions) == 5
        assert "driver_id" in predictions.columns
        assert "predicted_position" in predictions.columns
        assert "confidence" in predictions.columns

        # Check positions are 1-5
        assert set(predictions["predicted_position"]) == {1, 2, 3, 4, 5}

        # Confidence should be 0-100
        assert (predictions["confidence"] >= 0).all()
        assert (predictions["confidence"] <= 100).all()

    def test_predict_empty_features(self):
        """Test prediction with empty DataFrame."""
        predictor = RuleBasedPredictor()
        predictions = predictor.predict(pd.DataFrame())

        assert predictions.empty
        assert list(predictions.columns) == ["driver_id", "predicted_position", "confidence"]

    def test_predict_missing_columns(self):
        """Test prediction with missing required columns."""
        predictor = RuleBasedPredictor()
        incomplete_features = pd.DataFrame(
            {
                "driver_id": ["hamilton"],
                "qualifying_position": [1],
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            predictor.predict(incomplete_features)

    def test_predict_qualifying_weight_dominates(self):
        """Test that qualifying position has strong influence."""
        features = pd.DataFrame(
            {
                "driver_id": ["driver1", "driver2"],
                "qualifying_position": [1, 2],
                "driver_form_score": [50.0, 90.0],  # driver2 has much better form
                "team_reliability_score": [50.0, 90.0],
                "circuit_performance_score": [50.0, 90.0],
            }
        )

        predictor = RuleBasedPredictor(quali_weight=0.8, form_weight=0.1, reliability_weight=0.05, circuit_weight=0.05)
        predictions = predictor.predict(features)

        # driver1 should still be first due to quali position
        top_driver = predictions[predictions["predicted_position"] == 1]["driver_id"].values[0]
        assert top_driver == "driver1"

    def test_get_feature_importance(self):
        """Test feature importance retrieval."""
        predictor = RuleBasedPredictor()
        importance = predictor.get_feature_importance()

        assert len(importance) == 4
        assert importance["qualifying_position"] == 0.6
        assert importance["driver_form_score"] == 0.2
        assert importance["team_reliability_score"] == 0.1
        assert importance["circuit_performance_score"] == 0.1

    def test_predict_confidence_variation(self):
        """Test that confidence varies with score separation."""
        # High separation scenario
        high_sep_features = pd.DataFrame(
            {
                "driver_id": ["fast", "slow"],
                "qualifying_position": [1, 10],
                "driver_form_score": [95.0, 30.0],
                "team_reliability_score": [95.0, 30.0],
                "circuit_performance_score": [95.0, 30.0],
            }
        )

        # Low separation scenario
        low_sep_features = pd.DataFrame(
            {
                "driver_id": ["driver1", "driver2"],
                "qualifying_position": [1, 2],
                "driver_form_score": [80.0, 79.0],
                "team_reliability_score": [80.0, 79.0],
                "circuit_performance_score": [80.0, 79.0],
            }
        )

        predictor = RuleBasedPredictor()

        high_sep_pred = predictor.predict(high_sep_features)
        low_sep_pred = predictor.predict(low_sep_features)

        # High separation should have larger confidence difference
        high_conf_range = high_sep_pred["confidence"].max() - high_sep_pred["confidence"].min()
        low_conf_range = low_sep_pred["confidence"].max() - low_sep_pred["confidence"].min()

        assert high_conf_range > low_conf_range