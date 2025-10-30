"""Tests for Random Forest prediction model."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from f1_predict.models.random_forest import RandomForestRacePredictor


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
            "position": list(range(1, 21)),
        }
    )


class TestRandomForestRacePredictor:
    """Tests for RandomForestRacePredictor."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = RandomForestRacePredictor()
        assert predictor.target == "podium"
        assert predictor.random_state == 42
        assert predictor.n_estimators == 100
        assert not predictor.is_fitted

    def test_initialization_with_targets(self):
        """Test initialization with different targets."""
        for target in ["podium", "points", "win"]:
            predictor = RandomForestRacePredictor(target=target)
            assert predictor.target == target

    def test_initialization_with_hyperparameters(self):
        """Test initialization with custom hyperparameters."""
        predictor = RandomForestRacePredictor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="log2",
            random_state=123,
            oob_score=False,
        )
        assert predictor.n_estimators == 200
        assert predictor.max_depth == 10
        assert predictor.min_samples_split == 5
        assert predictor.min_samples_leaf == 2
        assert predictor.max_features == "log2"
        assert predictor.random_state == 123
        assert not predictor.oob_score_enabled

    def test_initialization_invalid_target(self):
        """Test that invalid target raises error."""
        with pytest.raises(ValueError, match="Invalid target"):
            RandomForestRacePredictor(target="invalid")

    def test_prepare_target_podium(self, sample_race_results):
        """Test target preparation for podium prediction."""
        predictor = RandomForestRacePredictor(target="podium")
        target = predictor._prepare_target(sample_race_results)

        assert len(target) == 20
        assert target.sum() == 3  # Only top 3 are 1
        assert (target.iloc[:3] == 1).all()
        assert (target.iloc[3:] == 0).all()

    def test_prepare_target_points(self, sample_race_results):
        """Test target preparation for points prediction."""
        predictor = RandomForestRacePredictor(target="points")
        target = predictor._prepare_target(sample_race_results)

        assert target.sum() == 10  # Top 10 score points
        assert (target.iloc[:10] == 1).all()
        assert (target.iloc[10:] == 0).all()

    def test_prepare_target_win(self, sample_race_results):
        """Test target preparation for win prediction."""
        predictor = RandomForestRacePredictor(target="win")
        target = predictor._prepare_target(sample_race_results)

        assert target.sum() == 1  # Only winner is 1
        assert target.iloc[0] == 1
        assert (target.iloc[1:] == 0).all()

    def test_fit_basic(self, sample_features, sample_race_results):
        """Test basic model fitting."""
        predictor = RandomForestRacePredictor(n_estimators=10)  # Faster for testing
        predictor.fit(sample_features, sample_race_results)

        assert predictor.is_fitted
        assert len(predictor.feature_names) == 4
        assert "qualifying_position" in predictor.feature_names
        assert "driver_form_score" in predictor.feature_names

    def test_fit_empty_data(self):
        """Test fitting with empty data raises error."""
        predictor = RandomForestRacePredictor()

        with pytest.raises(ValueError, match="Cannot train on empty data"):
            predictor.fit(pd.DataFrame(), pd.DataFrame())

    def test_fit_mismatched_lengths(self, sample_features):
        """Test fitting with mismatched data lengths."""
        predictor = RandomForestRacePredictor()
        short_results = pd.DataFrame({"position": [1, 2, 3]})

        with pytest.raises(ValueError, match="Feature count .* != result count"):
            predictor.fit(sample_features, short_results)

    def test_predict_proba(self, sample_features, sample_race_results):
        """Test probability prediction."""
        predictor = RandomForestRacePredictor(n_estimators=10)
        predictor.fit(sample_features, sample_race_results)

        probabilities = predictor.predict_proba(sample_features)

        assert len(probabilities) == 20
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()

    def test_predict_proba_unfitted(self, sample_features):
        """Test that prediction fails on unfitted model."""
        predictor = RandomForestRacePredictor()

        with pytest.raises(ValueError, match="Model must be fitted"):
            predictor.predict_proba(sample_features)

    def test_predict(self, sample_features, sample_race_results):
        """Test prediction functionality."""
        predictor = RandomForestRacePredictor(n_estimators=10)
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
        predictor = RandomForestRacePredictor(n_estimators=10)
        predictor.fit(sample_features, sample_race_results)

        predictions = predictor.predict(pd.DataFrame())

        assert predictions.empty
        assert list(predictions.columns) == [
            "driver_id",
            "predicted_outcome",
            "confidence",
        ]

    def test_predict_with_threshold(self, sample_features, sample_race_results):
        """Test prediction with custom threshold."""
        predictor = RandomForestRacePredictor(n_estimators=10)
        predictor.fit(sample_features, sample_race_results)

        # Low threshold should predict more positives
        low_threshold_pred = predictor.predict(sample_features, threshold=0.3)
        high_threshold_pred = predictor.predict(sample_features, threshold=0.7)

        assert (
            low_threshold_pred["predicted_outcome"].sum()
            >= high_threshold_pred["predicted_outcome"].sum()
        )

    def test_get_feature_importance(self, sample_features, sample_race_results):
        """Test feature importance retrieval."""
        predictor = RandomForestRacePredictor(n_estimators=10)
        predictor.fit(sample_features, sample_race_results)

        importance = predictor.get_feature_importance()

        assert len(importance) == 4
        assert all(v >= 0 for v in importance.values())
        # Importance should approximately sum to 1 (normalized)
        assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_get_feature_importance_unfitted(self):
        """Test that feature importance fails on unfitted model."""
        predictor = RandomForestRacePredictor()

        with pytest.raises(ValueError, match="Model must be fitted"):
            predictor.get_feature_importance()

    def test_get_oob_score(self, sample_features, sample_race_results):
        """Test OOB score retrieval."""
        predictor = RandomForestRacePredictor(n_estimators=10, oob_score=True)
        predictor.fit(sample_features, sample_race_results)

        oob_score = predictor.get_oob_score()

        assert oob_score is not None
        assert 0 <= oob_score <= 1

    def test_get_oob_score_disabled(self, sample_features, sample_race_results):
        """Test OOB score when disabled."""
        predictor = RandomForestRacePredictor(n_estimators=10, oob_score=False)
        predictor.fit(sample_features, sample_race_results)

        oob_score = predictor.get_oob_score()

        assert oob_score is None

    def test_get_oob_score_unfitted(self):
        """Test that OOB score fails on unfitted model."""
        predictor = RandomForestRacePredictor()

        with pytest.raises(ValueError, match="Model must be fitted"):
            predictor.get_oob_score()

    def test_save_and_load(self, sample_features, sample_race_results):
        """Test model save and load functionality."""
        predictor = RandomForestRacePredictor(target="podium", n_estimators=10)
        predictor.fit(sample_features, sample_race_results)

        # Get predictions before save
        original_predictions = predictor.predict(sample_features)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"

            # Save model
            predictor.save(model_path)
            assert model_path.exists()

            # Load model
            loaded_predictor = RandomForestRacePredictor.load(model_path)

            # Verify loaded model
            assert loaded_predictor.is_fitted
            assert loaded_predictor.target == "podium"
            assert loaded_predictor.n_estimators == 10
            assert loaded_predictor.feature_names == predictor.feature_names

            # Verify predictions are identical
            loaded_predictions = loaded_predictor.predict(sample_features)
            pd.testing.assert_frame_equal(original_predictions, loaded_predictions)

    def test_save_unfitted_model(self):
        """Test that saving unfitted model raises error."""
        predictor = RandomForestRacePredictor()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"

            with pytest.raises(ValueError, match="Model must be fitted before saving"):
                predictor.save(model_path)

    def test_load_nonexistent_file(self):
        """Test loading non-existent model file."""
        with pytest.raises(FileNotFoundError):
            RandomForestRacePredictor.load("nonexistent_model.pkl")

    def test_different_targets_different_predictions(
        self, sample_features, sample_race_results
    ):
        """Test that different targets produce different predictions."""
        podium_predictor = RandomForestRacePredictor(target="podium", n_estimators=10)
        points_predictor = RandomForestRacePredictor(target="points", n_estimators=10)

        podium_predictor.fit(sample_features, sample_race_results)
        points_predictor.fit(sample_features, sample_race_results)

        podium_pred = podium_predictor.predict(sample_features)
        points_pred = points_predictor.predict(sample_features)

        # Points predictions should have more positive outcomes
        assert (
            points_pred["predicted_outcome"].sum()
            >= podium_pred["predicted_outcome"].sum()
        )

    def test_different_n_estimators_affect_performance(
        self, sample_features, sample_race_results
    ):
        """Test that different n_estimators affect model."""
        predictor_small = RandomForestRacePredictor(n_estimators=5)
        predictor_large = RandomForestRacePredictor(n_estimators=50)

        predictor_small.fit(sample_features, sample_race_results)
        predictor_large.fit(sample_features, sample_race_results)

        # Both should be fitted
        assert predictor_small.is_fitted
        assert predictor_large.is_fitted

        # Predictions should be similar but potentially different
        pred_small = predictor_small.predict(sample_features)
        pred_large = predictor_large.predict(sample_features)

        assert len(pred_small) == len(pred_large)

    @patch("f1_predict.models.random_forest.ConfigLoader")
    def test_use_optimized_params_true(self, mock_config_loader):
        """Test that ConfigLoader is called when use_optimized_params=True."""
        # Setup mock
        mock_config_loader.get_hyperparameters.return_value = {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 3,
            "min_samples_leaf": 1,
            "max_features": "log2",
            "random_state": 42,
        }

        # Create predictor with optimized params
        predictor = RandomForestRacePredictor(target="podium", use_optimized_params=True)

        # Verify ConfigLoader was called
        mock_config_loader.get_hyperparameters.assert_called_once_with("random_forest")

        # Verify optimized params were used
        assert predictor.n_estimators == 200
        assert predictor.max_depth == 15
        assert predictor.min_samples_split == 3

    @patch("f1_predict.models.random_forest.ConfigLoader")
    def test_use_optimized_params_false(self, mock_config_loader):
        """Test that ConfigLoader is not called when use_optimized_params=False."""
        # Create predictor with optimization disabled
        predictor = RandomForestRacePredictor(
            target="podium",
            use_optimized_params=False,
            n_estimators=50,
            max_depth=5,
        )

        # Verify ConfigLoader was NOT called
        mock_config_loader.get_hyperparameters.assert_not_called()

        # Verify provided params were used
        assert predictor.n_estimators == 50
        assert predictor.max_depth == 5

    @patch("f1_predict.models.random_forest.ConfigLoader")
    def test_kwargs_override_optimized_params(self, mock_config_loader):
        """Test that kwargs override optimized params."""
        # Setup mock
        mock_config_loader.get_hyperparameters.return_value = {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 3,
            "min_samples_leaf": 1,
            "max_features": "log2",
            "random_state": 42,
        }

        # Create predictor with optimized params but override n_estimators
        predictor = RandomForestRacePredictor(
            target="podium",
            use_optimized_params=True,
            n_estimators=100,  # Override optimized value
        )

        # Verify optimized params were used except for override
        assert predictor.n_estimators == 100  # Overridden
        assert predictor.max_depth == 15  # From optimized
        assert predictor.min_samples_split == 3  # From optimized

    @patch("f1_predict.models.random_forest.ConfigLoader")
    def test_fallback_to_defaults_if_no_optimized(self, mock_config_loader):
        """Test fallback to defaults if ConfigLoader returns empty dict."""
        # Setup mock to return empty dict (no optimization file found)
        mock_config_loader.get_hyperparameters.return_value = {}

        # Create predictor - should use defaults
        predictor = RandomForestRacePredictor(target="podium", use_optimized_params=True)

        # Verify defaults were used
        assert predictor.n_estimators == 100  # Default
        assert predictor.max_depth is None  # Default
        assert predictor.min_samples_split == 2  # Default
