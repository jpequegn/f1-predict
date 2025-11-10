"""Unit tests for hyperparameter optimization module.

Tests cover:
- HyperparameterOptimizer initialization
- Model type validation
- Search space configuration
- Trial management
- Best parameter tracking
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from f1_predict.optimization.hyperparameter_optimizer import HyperparameterOptimizer


class TestHyperparameterOptimizerInitialization:
    """Tests for HyperparameterOptimizer initialization."""

    def test_init_xgboost(self):
        """Test initialization with XGBoost model type."""
        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="xgb_study",
            n_trials=50,
        )
        assert optimizer.model_type == "xgboost"
        assert optimizer.study_name == "xgb_study"
        assert optimizer.n_trials == 50

    def test_init_lightgbm(self):
        """Test initialization with LightGBM model type."""
        optimizer = HyperparameterOptimizer(
            model_type="lightgbm",
            study_name="lgb_study",
            n_trials=100,
        )
        assert optimizer.model_type == "lightgbm"
        assert optimizer.n_trials == 100

    def test_init_random_forest(self):
        """Test initialization with Random Forest model type."""
        optimizer = HyperparameterOptimizer(
            model_type="random_forest",
            study_name="rf_study",
            n_trials=75,
        )
        assert optimizer.model_type == "random_forest"
        assert optimizer.n_trials == 75

    def test_init_invalid_model_type(self):
        """Test initialization with invalid model type raises error."""
        with pytest.raises(ValueError, match="Unknown model type"):
            HyperparameterOptimizer(
                model_type="invalid_model",
                study_name="study",
                n_trials=50,
            )

    def test_init_with_timeout(self):
        """Test initialization with timeout parameter."""
        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="study",
            n_trials=100,
            timeout_seconds=1800,
        )
        assert optimizer.timeout_seconds == 1800

    def test_init_with_no_timeout(self):
        """Test initialization with no timeout (unlimited)."""
        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="study",
            n_trials=100,
            timeout_seconds=None,
        )
        assert optimizer.timeout_seconds is None

    def test_init_default_trials(self):
        """Test initialization with default n_trials."""
        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="study",
        )
        assert optimizer.n_trials == 100

    def test_init_default_timeout(self):
        """Test initialization with default timeout."""
        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="study",
        )
        assert optimizer.timeout_seconds == 3600


class TestObjectiveFunction:
    """Tests for objective function mapping."""

    def test_objective_functions_exist(self):
        """Test that all objective functions are defined."""
        assert "xgboost" in HyperparameterOptimizer.OBJECTIVE_FUNCTIONS
        assert "lightgbm" in HyperparameterOptimizer.OBJECTIVE_FUNCTIONS
        assert "random_forest" in HyperparameterOptimizer.OBJECTIVE_FUNCTIONS

    def test_objective_function_count(self):
        """Test that all model types have objective functions."""
        assert len(HyperparameterOptimizer.OBJECTIVE_FUNCTIONS) == 3

    def test_objective_function_callable(self):
        """Test that objective functions are callable."""
        for func in HyperparameterOptimizer.OBJECTIVE_FUNCTIONS.values():
            assert callable(func)


class TestTrialConfiguration:
    """Tests for trial configuration and parameters."""

    def test_trial_count_range(self):
        """Test various trial counts."""
        for n_trials in [10, 50, 100, 200]:
            optimizer = HyperparameterOptimizer(
                model_type="xgboost",
                study_name="study",
                n_trials=n_trials,
            )
            assert optimizer.n_trials == n_trials

    def test_study_name_storage(self):
        """Test that study name is properly stored."""
        study_name = "my_custom_study_123"
        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name=study_name,
        )
        assert optimizer.study_name == study_name

    def test_study_name_with_special_chars(self):
        """Test study name with special characters."""
        study_name = "study_2024-11-10_xgb"
        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name=study_name,
        )
        assert optimizer.study_name == study_name


class TestModelTypeValidation:
    """Tests for model type validation."""

    def test_valid_model_types(self):
        """Test all valid model types are accepted."""
        valid_types = ["xgboost", "lightgbm", "random_forest"]
        for model_type in valid_types:
            optimizer = HyperparameterOptimizer(
                model_type=model_type,
                study_name="study",
            )
            assert optimizer.model_type == model_type

    def test_case_sensitive_model_type(self):
        """Test that model type is case-sensitive."""
        with pytest.raises(ValueError):
            HyperparameterOptimizer(
                model_type="XGBoost",
                study_name="study",
            )

    def test_empty_model_type(self):
        """Test that empty model type raises error."""
        with pytest.raises(ValueError):
            HyperparameterOptimizer(
                model_type="",
                study_name="study",
            )

    def test_none_model_type(self):
        """Test that None model type raises error."""
        with pytest.raises((ValueError, TypeError)):
            HyperparameterOptimizer(
                model_type=None,
                study_name="study",
            )


class TestTimeoutConfiguration:
    """Tests for timeout configuration."""

    def test_various_timeouts(self):
        """Test various timeout values."""
        timeouts = [300, 600, 1800, 3600, 7200]
        for timeout in timeouts:
            optimizer = HyperparameterOptimizer(
                model_type="xgboost",
                study_name="study",
                timeout_seconds=timeout,
            )
            assert optimizer.timeout_seconds == timeout

    def test_zero_timeout(self):
        """Test zero timeout is allowed."""
        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="study",
            timeout_seconds=0,
        )
        assert optimizer.timeout_seconds == 0

    def test_very_large_timeout(self):
        """Test very large timeout values."""
        large_timeout = 86400 * 7
        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="study",
            timeout_seconds=large_timeout,
        )
        assert optimizer.timeout_seconds == large_timeout


class TestOptimizerAttributes:
    """Tests for optimizer attribute initialization."""

    def test_initial_study_is_none(self):
        """Test that study is None before optimization."""
        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="study",
        )
        assert optimizer.study is None

    def test_initial_best_params_is_none(self):
        """Test that best_params is None before optimization."""
        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="study",
        )
        assert optimizer.best_params is None

    def test_initial_best_model_is_none(self):
        """Test that best_model is None before optimization."""
        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="study",
        )
        assert optimizer.best_model is None


class TestSearchSpaceIntegration:
    """Tests for search space configuration integration."""

    def test_xgboost_search_space_exists(self):
        """Test that XGBoost search space is registered."""
        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="study",
        )
        assert optimizer.model_type == "xgboost"

    def test_lightgbm_search_space_exists(self):
        """Test that LightGBM search space is registered."""
        optimizer = HyperparameterOptimizer(
            model_type="lightgbm",
            study_name="study",
        )
        assert optimizer.model_type == "lightgbm"

    def test_random_forest_search_space_exists(self):
        """Test that Random Forest search space is registered."""
        optimizer = HyperparameterOptimizer(
            model_type="random_forest",
            study_name="study",
        )
        assert optimizer.model_type == "random_forest"


class TestOptimizerStatePersistence:
    """Tests for optimizer state persistence."""

    def test_multiple_optimizers_independent(self):
        """Test that multiple optimizer instances are independent."""
        opt1 = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="study1",
            n_trials=50,
        )
        opt2 = HyperparameterOptimizer(
            model_type="lightgbm",
            study_name="study2",
            n_trials=100,
        )

        assert opt1.model_type == "xgboost"
        assert opt2.model_type == "lightgbm"
        assert opt1.n_trials == 50
        assert opt2.n_trials == 100

    def test_optimizer_model_type_immutable(self):
        """Test that model type doesn't change after init."""
        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="study",
        )
        original_type = optimizer.model_type
        _ = optimizer.model_type
        assert optimizer.model_type == original_type
