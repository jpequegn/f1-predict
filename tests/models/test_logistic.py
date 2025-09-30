"""Tests for logistic regression prediction model."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from f1_predict.models.logistic import LogisticRacePredictor


@pytest.fixture
def sample_features():
    """Create sample feature data for testing."""
    return pd.DataFrame(
        {
            "driver_id": [f"driver{i}" for i in range(20)],
            "qualifying_position": list(range(1, 21)),
            "driver_form_score": [85.0 - i * 3 for i in range(20)],
            "team_reliability_score": [90.0 - i * 2 for i in range(20)],
            "circuit_performance_score": [88.0 - i * 2.5 for i in range(20)],
        }
    )


@pytest.fixture
def sample_race_results():
    """Create sample race results for testing."""
    return pd.DataFrame(
        {
            "driver_id": [f"driver{i}" for i in range(20)],
            "position": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        }
    )


class TestLogisticRacePredictor:
    """Tests for LogisticRacePredictor."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = LogisticRacePredictor()
        assert predictor.target == "podium"
        assert predictor.random_state == 42
        assert not predictor.is_fitted

    def test_initialization_with_targets(self):
        """Test initialization with different targets."""
        for target in ["podium", "points", "win"]:
            predictor = LogisticRacePredictor(target=target)
            assert predictor.target == target

    def test_initialization_invalid_target(self):
        """Test that invalid target raises error."""
        with pytest.raises(ValueError, match="Invalid target"):
            LogisticRacePredictor(target="invalid")

    def test_prepare_target_podium(self, sample_race_results):
        """Test target preparation for podium prediction."""
        predictor = LogisticRacePredictor(target="podium")
        target = predictor._prepare_target(sample_race_results)

        assert len(target) == 20
        assert target.sum() == 3  # Only top 3 are 1
        assert (target.iloc[:3] == 1).all()
        assert (target.iloc[3:] == 0).all()

    def test_prepare_target_points(self, sample_race_results):
        """Test target preparation for points prediction."""
        predictor = LogisticRacePredictor(target="points")
        target = predictor._prepare_target(sample_race_results)

        assert target.sum() == 10  # Top 10 score points
        assert (target.iloc[:10] == 1).all()
        assert (target.iloc[10:] == 0).all()

    def test_prepare_target_win(self, sample_race_results):
        """Test target preparation for win prediction."""
        predictor = LogisticRacePredictor(target="win")
        target = predictor._prepare_target(sample_race_results)

        assert target.sum() == 1  # Only winner is 1
        assert target.iloc[0] == 1
        assert (target.iloc[1:] == 0).all()

    def test_fit_basic(self, sample_features, sample_race_results):
        """Test basic model fitting."""
        predictor = LogisticRacePredictor()
        predictor.fit(sample_features, sample_race_results)

        assert predictor.is_fitted
        assert len(predictor.feature_names) == 4
        assert "qualifying_position" in predictor.feature_names
        assert "driver_form_score" in predictor.feature_names

    def test_fit_empty_data(self):
        """Test fitting with empty data raises error."""
        predictor = LogisticRacePredictor()

        with pytest.raises(ValueError, match="Cannot train on empty data"):
            predictor.fit(pd.DataFrame(), pd.DataFrame())

    def test_fit_mismatched_lengths(self, sample_features):
        """Test fitting with mismatched data lengths."""
        predictor = LogisticRacePredictor()
        short_results = pd.DataFrame({"position": [1, 2, 3]})

        with pytest.raises(ValueError, match="Feature count .* != result count"):
            predictor.fit(sample_features, short_results)

    def test_predict_proba(self, sample_features, sample_race_results):
        """Test probability prediction."""
        predictor = LogisticRacePredictor()
        predictor.fit(sample_features, sample_race_results)

        probabilities = predictor.predict_proba(sample_features)

        assert len(probabilities) == 20
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()

    def test_predict_proba_unfitted(self, sample_features):
        """Test that prediction fails on unfitted model."""
        predictor = LogisticRacePredictor()

        with pytest.raises(ValueError, match="Model must be fitted"):
            predictor.predict_proba(sample_features)

    def test_predict(self, sample_features, sample_race_results):
        """Test prediction functionality."""
        predictor = LogisticRacePredictor()
        predictor.fit(sample_features, sample_race_results)

        predictions = predictor.predict(sample_features)

        assert len(predictions) == 20
        assert "driver_id" in predictions.columns
        assert "predicted_outcome" in predictions.columns
        assert "confidence" in predictions.columns

        # Confidence should be 0-100
        assert (predictions["confidence"] >= 0).all()
        assert (predictions["confidence"] <= 100).all()

        # Predicted outcome should be binary
        assert set(predictions["predicted_outcome"].unique()).issubset({0, 1})

    def test_predict_empty_features(self, sample_features, sample_race_results):
        """Test prediction with empty DataFrame."""
        predictor = LogisticRacePredictor()
        predictor.fit(sample_features, sample_race_results)

        predictions = predictor.predict(pd.DataFrame())

        assert predictions.empty
        assert list(predictions.columns) == ["driver_id", "predicted_outcome", "confidence"]

    def test_predict_with_threshold(self, sample_features, sample_race_results):
        """Test prediction with custom threshold."""
        predictor = LogisticRacePredictor()
        predictor.fit(sample_features, sample_race_results)

        # Low threshold should predict more positives
        low_threshold_pred = predictor.predict(sample_features, threshold=0.3)
        high_threshold_pred = predictor.predict(sample_features, threshold=0.7)

        assert low_threshold_pred["predicted_outcome"].sum() >= high_threshold_pred["predicted_outcome"].sum()

    def test_get_feature_importance(self, sample_features, sample_race_results):
        """Test feature importance retrieval."""
        predictor = LogisticRacePredictor()
        predictor.fit(sample_features, sample_race_results)

        importance = predictor.get_feature_importance()

        assert len(importance) == 4
        assert all(v >= 0 for v in importance.values())
        # Importance should approximately sum to 1 (normalized)
        assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_get_feature_importance_unfitted(self):
        """Test that feature importance fails on unfitted model."""
        predictor = LogisticRacePredictor()

        with pytest.raises(ValueError, match="Model must be fitted"):
            predictor.get_feature_importance()

    def test_save_and_load(self, sample_features, sample_race_results):
        """Test model save and load functionality."""
        predictor = LogisticRacePredictor(target="podium")
        predictor.fit(sample_features, sample_race_results)

        # Get predictions before save
        original_predictions = predictor.predict(sample_features)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"

            # Save model
            predictor.save(model_path)
            assert model_path.exists()

            # Load model
            loaded_predictor = LogisticRacePredictor.load(model_path)

            # Verify loaded model
            assert loaded_predictor.is_fitted
            assert loaded_predictor.target == "podium"
            assert loaded_predictor.feature_names == predictor.feature_names

            # Verify predictions are identical
            loaded_predictions = loaded_predictor.predict(sample_features)
            pd.testing.assert_frame_equal(original_predictions, loaded_predictions)

    def test_save_unfitted_model(self):
        """Test that saving unfitted model raises error."""
        predictor = LogisticRacePredictor()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"

            with pytest.raises(ValueError, match="Model must be fitted before saving"):
                predictor.save(model_path)

    def test_load_nonexistent_file(self):
        """Test loading non-existent model file."""
        with pytest.raises(FileNotFoundError):
            LogisticRacePredictor.load("nonexistent_model.pkl")

    def test_different_targets_different_predictions(self, sample_features, sample_race_results):
        """Test that different targets produce different predictions."""
        podium_predictor = LogisticRacePredictor(target="podium")
        points_predictor = LogisticRacePredictor(target="points")

        podium_predictor.fit(sample_features, sample_race_results)
        points_predictor.fit(sample_features, sample_race_results)

        podium_pred = podium_predictor.predict(sample_features)
        points_pred = points_predictor.predict(sample_features)

        # Points predictions should have more positive outcomes
        assert points_pred["predicted_outcome"].sum() >= podium_pred["predicted_outcome"].sum()