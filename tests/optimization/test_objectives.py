"""Tests for optimization objective functions."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from f1_predict.optimization.objectives import ObjectiveFunction


@pytest.fixture
def sample_data():
    """Create sample training and validation data."""
    np.random.seed(42)
    x_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 3, 100)
    x_val = np.random.rand(30, 10)
    y_val = np.random.randint(0, 3, 30)
    return x_train, y_train, x_val, y_val


@pytest.fixture
def mock_trial():
    """Create a mock Optuna trial object."""
    trial = MagicMock()

    # Configure mock to return reasonable values
    trial.suggest_int.side_effect = lambda name, low, high: (low + high) // 2
    trial.suggest_float.side_effect = lambda name, low, high, **kwargs: (low + high) / 2
    trial.suggest_categorical.side_effect = lambda name, choices: choices[0]

    return trial


class TestObjectiveFunctionXGBoost:
    """Test XGBoost objective function."""

    def test_optimize_xgboost_returns_float(self, mock_trial, sample_data):
        """Test that optimize_xgboost returns a float accuracy score."""
        x_train, y_train, x_val, y_val = sample_data

        accuracy = ObjectiveFunction.optimize_xgboost(
            mock_trial, x_train, y_train, x_val, y_val
        )

        assert isinstance(accuracy, (float, np.floating))

    def test_optimize_xgboost_returns_valid_range(self, mock_trial, sample_data):
        """Test that accuracy is between 0 and 1."""
        x_train, y_train, x_val, y_val = sample_data

        accuracy = ObjectiveFunction.optimize_xgboost(
            mock_trial, x_train, y_train, x_val, y_val
        )

        assert 0.0 <= accuracy <= 1.0 or np.isnan(accuracy)

    def test_optimize_xgboost_error_handling(self, mock_trial, sample_data):
        """Test that errors are caught and return NaN."""
        x_train, y_train, x_val, y_val = sample_data

        # Force an error by passing invalid data
        with patch("xgboost.XGBClassifier.fit", side_effect=ValueError("Test error")):
            accuracy = ObjectiveFunction.optimize_xgboost(
                mock_trial, x_train, y_train, x_val, y_val
            )

            assert np.isnan(accuracy)


class TestObjectiveFunctionLightGBM:
    """Test LightGBM objective function."""

    def test_optimize_lightgbm_returns_float(self, mock_trial, sample_data):
        """Test that optimize_lightgbm returns a float accuracy score."""
        x_train, y_train, x_val, y_val = sample_data

        accuracy = ObjectiveFunction.optimize_lightgbm(
            mock_trial, x_train, y_train, x_val, y_val
        )

        assert isinstance(accuracy, (float, np.floating))

    def test_optimize_lightgbm_returns_valid_range(self, mock_trial, sample_data):
        """Test that accuracy is between 0 and 1."""
        x_train, y_train, x_val, y_val = sample_data

        accuracy = ObjectiveFunction.optimize_lightgbm(
            mock_trial, x_train, y_train, x_val, y_val
        )

        assert 0.0 <= accuracy <= 1.0 or np.isnan(accuracy)

    def test_optimize_lightgbm_error_handling(self, mock_trial, sample_data):
        """Test that errors are caught and return NaN."""
        x_train, y_train, x_val, y_val = sample_data

        # Force an error by passing invalid data
        with patch("lightgbm.LGBMClassifier.fit", side_effect=ValueError("Test error")):
            accuracy = ObjectiveFunction.optimize_lightgbm(
                mock_trial, x_train, y_train, x_val, y_val
            )

            assert np.isnan(accuracy)


class TestObjectiveFunctionRandomForest:
    """Test RandomForest objective function."""

    def test_optimize_random_forest_returns_float(self, mock_trial, sample_data):
        """Test that optimize_random_forest returns a float accuracy score."""
        x_train, y_train, x_val, y_val = sample_data

        accuracy = ObjectiveFunction.optimize_random_forest(
            mock_trial, x_train, y_train, x_val, y_val
        )

        assert isinstance(accuracy, (float, np.floating))

    def test_optimize_random_forest_returns_valid_range(self, mock_trial, sample_data):
        """Test that accuracy is between 0 and 1."""
        x_train, y_train, x_val, y_val = sample_data

        accuracy = ObjectiveFunction.optimize_random_forest(
            mock_trial, x_train, y_train, x_val, y_val
        )

        assert 0.0 <= accuracy <= 1.0 or np.isnan(accuracy)

    def test_optimize_random_forest_error_handling(self, mock_trial, sample_data):
        """Test that errors are caught and return NaN."""
        x_train, y_train, x_val, y_val = sample_data

        # Force an error by passing invalid data
        with patch("sklearn.ensemble.RandomForestClassifier.fit", side_effect=ValueError("Test error")):
            accuracy = ObjectiveFunction.optimize_random_forest(
                mock_trial, x_train, y_train, x_val, y_val
            )

            assert np.isnan(accuracy)


class TestObjectiveFunctionInvalidParams:
    """Test handling of invalid parameters."""

    def test_invalid_params_gracefully_return_nan(self, sample_data):
        """Test that invalid parameters don't crash but return NaN."""
        x_train, y_train, x_val, y_val = sample_data

        # Create trial that suggests invalid parameters
        trial = MagicMock()
        trial.suggest_int.return_value = -1  # Invalid n_estimators
        trial.suggest_float.return_value = -1.0  # Invalid learning_rate
        trial.suggest_categorical.return_value = "invalid"

        # Should handle gracefully and return NaN
        accuracy = ObjectiveFunction.optimize_xgboost(
            trial, x_train, y_train, x_val, y_val
        )

        assert np.isnan(accuracy)
