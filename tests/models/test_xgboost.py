"""Tests for XGBoost prediction model."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from f1_predict.models.xgboost_model import XGBoostRacePredictor


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


class TestXGBoostRacePredictor:
    """Tests for XGBoostRacePredictor."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = XGBoostRacePredictor()
        assert predictor.target == "podium"
        assert predictor.random_state == 42
        assert predictor.n_estimators == 100
        assert not predictor.is_fitted

    def test_initialization_with_targets(self):
        """Test initialization with different targets."""
        for target in ["podium", "points", "win"]:
            predictor = XGBoostRacePredictor(target=target)
            assert predictor.target == target

    def test_initialization_with_hyperparameters(self):
        """Test initialization with custom hyperparameters."""
        predictor = XGBoostRacePredictor(
            learning_rate=0.05,
            n_estimators=50,
            max_depth=4,
            min_child_weight=2,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=123,
            early_stopping_rounds=5,
        )
        assert predictor.learning_rate == 0.05
        assert predictor.n_estimators == 50
        assert predictor.max_depth == 4
        assert predictor.min_child_weight == 2
        assert predictor.subsample == 0.7
        assert predictor.colsample_bytree == 0.7
        assert predictor.random_state == 123
        assert predictor.early_stopping_rounds == 5

    def test_initialization_invalid_target(self):
        """Test that invalid target raises error."""
        with pytest.raises(ValueError, match="Invalid target"):
            XGBoostRacePredictor(target="invalid")

    def test_fit_basic(self, sample_features, sample_race_results):
        """Test basic model fitting."""
        predictor = XGBoostRacePredictor(n_estimators=10)
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

        predictor = XGBoostRacePredictor(
            n_estimators=50, early_stopping_rounds=10
        )
        predictor.fit(
            train_features,
            train_results,
            eval_set=(eval_features, eval_results),
        )

        assert predictor.is_fitted
        assert predictor.best_iteration is not None
        assert predictor.best_score is not None

    def test_fit_empty_data(self):
        """Test fitting with empty data raises error."""
        predictor = XGBoostRacePredictor()

        with pytest.raises(ValueError, match="Cannot train on empty data"):
            predictor.fit(pd.DataFrame(), pd.DataFrame())

    def test_fit_mismatched_lengths(self, sample_features):
        """Test fitting with mismatched data lengths."""
        predictor = XGBoostRacePredictor()
        short_results = pd.DataFrame({"position": [1, 2, 3]})

        with pytest.raises(ValueError, match="Feature count .* != result count"):
            predictor.fit(sample_features, short_results)

    def test_predict_proba(self, sample_features, sample_race_results):
        """Test probability prediction."""
        predictor = XGBoostRacePredictor(n_estimators=10)
        predictor.fit(sample_features, sample_race_results)

        probabilities = predictor.predict_proba(sample_features)

        assert len(probabilities) == 20
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()

    def test_predict_proba_unfitted(self, sample_features):
        """Test that prediction fails on unfitted model."""
        predictor = XGBoostRacePredictor()

        with pytest.raises(ValueError, match="Model must be fitted"):
            predictor.predict_proba(sample_features)

    def test_predict(self, sample_features, sample_race_results):
        """Test prediction functionality."""
        predictor = XGBoostRacePredictor(n_estimators=10)
        predictor.fit(sample_features, sample_race_results)

        predictions = predictor.predict(sample_features)

        assert len(predictions) == 20
        assert "driver_id" in predictions.columns
        assert "predicted_outcome" in predictions.columns
        assert "confidence" in predictions.columns
        assert (predictions["confidence"] >= 0).all()
        assert (predictions["confidence"] <= 100).all()

    def test_get_feature_importance(self, sample_features, sample_race_results):
        """Test feature importance retrieval with different types."""
        predictor = XGBoostRacePredictor(n_estimators=10)
        predictor.fit(sample_features, sample_race_results)

        for importance_type in ["weight", "gain", "cover"]:
            importance = predictor.get_feature_importance(importance_type)
            assert len(importance) == 4
            assert all(v >= 0 for v in importance.values())
            assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_get_feature_importance_invalid_type(
        self, sample_features, sample_race_results
    ):
        """Test that invalid importance type raises error."""
        predictor = XGBoostRacePredictor(n_estimators=10)
        predictor.fit(sample_features, sample_race_results)

        with pytest.raises(ValueError, match="Invalid importance_type"):
            predictor.get_feature_importance("invalid")

    def test_save_and_load(self, sample_features, sample_race_results):
        """Test model save and load functionality."""
        predictor = XGBoostRacePredictor(target="podium", n_estimators=10)
        predictor.fit(sample_features, sample_race_results)

        original_predictions = predictor.predict(sample_features)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"

            predictor.save(model_path)
            assert model_path.exists()
            assert model_path.with_suffix(".xgb").exists()

            loaded_predictor = XGBoostRacePredictor.load(model_path)

            assert loaded_predictor.is_fitted
            assert loaded_predictor.target == "podium"
            assert loaded_predictor.feature_names == predictor.feature_names

            loaded_predictions = loaded_predictor.predict(sample_features)
            pd.testing.assert_frame_equal(original_predictions, loaded_predictions)

    def test_save_unfitted_model(self):
        """Test that saving unfitted model raises error."""
        predictor = XGBoostRacePredictor()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"

            with pytest.raises(ValueError, match="Model must be fitted before saving"):
                predictor.save(model_path)

    def test_load_nonexistent_file(self):
        """Test loading non-existent model file."""
        with pytest.raises(FileNotFoundError):
            XGBoostRacePredictor.load("nonexistent_model.pkl")

    @patch("f1_predict.models.xgboost_model.ConfigLoader")
    def test_use_optimized_params_true(self, mock_config_loader):
        """Test that ConfigLoader is called when use_optimized_params=True."""
        # Setup mock
        mock_config_loader.get_hyperparameters.return_value = {
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 7,
            "min_child_weight": 2,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 0.5,
        }

        # Create predictor with optimized params
        predictor = XGBoostRacePredictor(target="podium", use_optimized_params=True)

        # Verify ConfigLoader was called
        mock_config_loader.get_hyperparameters.assert_called_once_with("xgboost")

        # Verify optimized params were used
        assert predictor.learning_rate == 0.05
        assert predictor.n_estimators == 200
        assert predictor.max_depth == 7

    @patch("f1_predict.models.xgboost_model.ConfigLoader")
    def test_use_optimized_params_false(self, mock_config_loader):
        """Test that ConfigLoader is not called when use_optimized_params=False."""
        # Create predictor with optimization disabled
        predictor = XGBoostRacePredictor(
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

    @patch("f1_predict.models.xgboost_model.ConfigLoader")
    def test_kwargs_override_optimized_params(self, mock_config_loader):
        """Test that kwargs override optimized params."""
        # Setup mock
        mock_config_loader.get_hyperparameters.return_value = {
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 7,
            "min_child_weight": 2,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 0.5,
        }

        # Create predictor with optimized params but override learning_rate
        predictor = XGBoostRacePredictor(
            target="podium",
            use_optimized_params=True,
            learning_rate=0.1,  # Override optimized value
        )

        # Verify optimized params were used except for override
        assert predictor.learning_rate == 0.1  # Overridden
        assert predictor.n_estimators == 200  # From optimized
        assert predictor.max_depth == 7  # From optimized

    @patch("f1_predict.models.xgboost_model.ConfigLoader")
    def test_fallback_to_defaults_if_no_optimized(self, mock_config_loader):
        """Test fallback to defaults if ConfigLoader returns empty dict."""
        # Setup mock to return empty dict (no optimization file found)
        mock_config_loader.get_hyperparameters.return_value = {}

        # Create predictor - should use defaults
        predictor = XGBoostRacePredictor(target="podium", use_optimized_params=True)

        # Verify defaults were used
        assert predictor.learning_rate == 0.1  # Default
        assert predictor.n_estimators == 100  # Default
        assert predictor.max_depth == 6  # Default
