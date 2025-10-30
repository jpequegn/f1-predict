"""Tests for LightGBM prediction model."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from f1_predict.models.lightgbm_model import LightGBMRacePredictor


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


class TestLightGBMRacePredictor:
    """Tests for LightGBMRacePredictor."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = LightGBMRacePredictor()
        assert predictor.target == "podium"
        assert predictor.random_state == 42
        assert predictor.n_estimators == 100
        assert not predictor.is_fitted

    def test_initialization_with_targets(self):
        """Test initialization with different targets."""
        for target in ["podium", "points", "win"]:
            predictor = LightGBMRacePredictor(target=target)
            assert predictor.target == target

    def test_initialization_with_hyperparameters(self):
        """Test initialization with custom hyperparameters."""
        predictor = LightGBMRacePredictor(
            learning_rate=0.05,
            n_estimators=50,
            num_leaves=15,
            max_depth=5,
            min_data_in_leaf=10,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=3,
            random_state=123,
            early_stopping_rounds=5,
        )
        assert predictor.learning_rate == 0.05
        assert predictor.n_estimators == 50
        assert predictor.num_leaves == 15
        assert predictor.max_depth == 5
        assert predictor.min_data_in_leaf == 10
        assert predictor.feature_fraction == 0.7
        assert predictor.bagging_fraction == 0.7
        assert predictor.bagging_freq == 3
        assert predictor.random_state == 123
        assert predictor.early_stopping_rounds == 5

    def test_initialization_invalid_target(self):
        """Test that invalid target raises error."""
        with pytest.raises(ValueError, match="Invalid target"):
            LightGBMRacePredictor(target="invalid")

    def test_fit_basic(self, sample_features, sample_race_results):
        """Test basic model fitting."""
        predictor = LightGBMRacePredictor(n_estimators=10)
        predictor.fit(sample_features, sample_race_results)

        assert predictor.is_fitted
        assert len(predictor.feature_names) == 4
        assert "qualifying_position" in predictor.feature_names
        assert "driver_form_score" in predictor.feature_names

    def test_fit_with_eval_set(self, sample_features, sample_race_results):
        """Test fitting with evaluation set."""
        # Split data
        train_features = sample_features.iloc[:15]
        train_results = sample_race_results.iloc[:15]
        eval_features = sample_features.iloc[15:]
        eval_results = sample_race_results.iloc[15:]

        predictor = LightGBMRacePredictor(
            n_estimators=50, early_stopping_rounds=10
        )
        predictor.fit(
            train_features,
            train_results,
            eval_set=(eval_features, eval_results),
        )

        assert predictor.is_fitted
        assert predictor.best_iteration is not None

    def test_fit_empty_data(self):
        """Test fitting with empty data raises error."""
        predictor = LightGBMRacePredictor()

        with pytest.raises(ValueError, match="Cannot train on empty data"):
            predictor.fit(pd.DataFrame(), pd.DataFrame())

    def test_predict_proba(self, sample_features, sample_race_results):
        """Test probability prediction."""
        predictor = LightGBMRacePredictor(n_estimators=10)
        predictor.fit(sample_features, sample_race_results)

        probabilities = predictor.predict_proba(sample_features)

        assert len(probabilities) == 20
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()

    def test_predict(self, sample_features, sample_race_results):
        """Test prediction functionality."""
        predictor = LightGBMRacePredictor(n_estimators=10)
        predictor.fit(sample_features, sample_race_results)

        predictions = predictor.predict(sample_features)

        assert len(predictions) == 20
        assert "driver_id" in predictions.columns
        assert "predicted_outcome" in predictions.columns
        assert "confidence" in predictions.columns

    def test_get_feature_importance(self, sample_features, sample_race_results):
        """Test feature importance retrieval with different types."""
        predictor = LightGBMRacePredictor(n_estimators=10)
        predictor.fit(sample_features, sample_race_results)

        for importance_type in ["split", "gain"]:
            importance = predictor.get_feature_importance(importance_type)
            assert len(importance) == 4
            assert all(v >= 0 for v in importance.values())

    def test_save_and_load(self, sample_features, sample_race_results):
        """Test model save and load functionality."""
        predictor = LightGBMRacePredictor(target="podium", n_estimators=10)
        predictor.fit(sample_features, sample_race_results)

        original_predictions = predictor.predict(sample_features)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"

            predictor.save(model_path)
            assert model_path.exists()
            assert model_path.with_suffix(".lgb").exists()

            loaded_predictor = LightGBMRacePredictor.load(model_path)

            assert loaded_predictor.is_fitted
            assert loaded_predictor.target == "podium"
            assert loaded_predictor.feature_names == predictor.feature_names

            loaded_predictions = loaded_predictor.predict(sample_features)
            pd.testing.assert_frame_equal(original_predictions, loaded_predictions)

    @patch("f1_predict.models.lightgbm_model.ConfigLoader")
    def test_use_optimized_params_true(self, mock_config_loader):
        """Test that ConfigLoader is called when use_optimized_params=True."""
        # Setup mock
        mock_config_loader.get_hyperparameters.return_value = {
            "learning_rate": 0.05,
            "n_estimators": 200,
            "num_leaves": 15,
            "max_depth": 7,
            "min_data_in_leaf": 10,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.85,
            "bagging_freq": 3,
            "early_stopping_rounds": 5,
        }

        # Create predictor with optimized params
        predictor = LightGBMRacePredictor(target="podium", use_optimized_params=True)

        # Verify ConfigLoader was called
        mock_config_loader.get_hyperparameters.assert_called_once_with("lightgbm")

        # Verify optimized params were used
        assert predictor.learning_rate == 0.05
        assert predictor.n_estimators == 200
        assert predictor.num_leaves == 15

    @patch("f1_predict.models.lightgbm_model.ConfigLoader")
    def test_use_optimized_params_false(self, mock_config_loader):
        """Test that ConfigLoader is not called when use_optimized_params=False."""
        # Create predictor with optimization disabled
        predictor = LightGBMRacePredictor(
            target="podium",
            use_optimized_params=False,
            learning_rate=0.05,
            n_estimators=50,
        )

        # Verify ConfigLoader was NOT called
        mock_config_loader.get_hyperparameters.assert_not_called()

        # Verify provided params were used
        assert predictor.learning_rate == 0.05
        assert predictor.n_estimators == 50

    @patch("f1_predict.models.lightgbm_model.ConfigLoader")
    def test_kwargs_override_optimized_params(self, mock_config_loader):
        """Test that kwargs override optimized params."""
        # Setup mock
        mock_config_loader.get_hyperparameters.return_value = {
            "learning_rate": 0.05,
            "n_estimators": 200,
            "num_leaves": 15,
            "max_depth": 7,
            "min_data_in_leaf": 10,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.85,
            "bagging_freq": 3,
            "early_stopping_rounds": 5,
        }

        # Create predictor with optimized params but override learning_rate
        predictor = LightGBMRacePredictor(
            target="podium",
            use_optimized_params=True,
            learning_rate=0.1,  # Override optimized value
        )

        # Verify optimized params were used except for override
        assert predictor.learning_rate == 0.1  # Overridden
        assert predictor.n_estimators == 200  # From optimized
        assert predictor.num_leaves == 15  # From optimized

    @patch("f1_predict.models.lightgbm_model.ConfigLoader")
    def test_fallback_to_defaults_if_no_optimized(self, mock_config_loader):
        """Test fallback to defaults if ConfigLoader returns empty dict."""
        # Setup mock to return empty dict (no optimization file found)
        mock_config_loader.get_hyperparameters.return_value = {}

        # Create predictor - should use defaults
        predictor = LightGBMRacePredictor(target="podium", use_optimized_params=True)

        # Verify defaults were used
        assert predictor.learning_rate == 0.1  # Default
        assert predictor.n_estimators == 100  # Default
        assert predictor.num_leaves == 31  # Default
