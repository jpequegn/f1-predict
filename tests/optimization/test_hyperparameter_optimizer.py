"""Tests for hyperparameter optimizer."""

import numpy as np
import pytest

from f1_predict.optimization.hyperparameter_optimizer import HyperparameterOptimizer


class TestHyperparameterOptimizer:
    """Test suite for HyperparameterOptimizer class."""

    @pytest.fixture
    def sample_data(self):
        """Create small sample dataset for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        x_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 3, n_samples)  # 3 classes
        x_val = np.random.randn(n_samples // 2, n_features)
        y_val = np.random.randint(0, 3, n_samples // 2)

        return x_train, y_train, x_val, y_val

    def test_initialization_with_valid_model_type(self):
        """Test that HyperparameterOptimizer initializes with valid model type."""
        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="test_study",
            n_trials=10,
            timeout_seconds=60,
        )

        assert optimizer.model_type == "xgboost"
        assert optimizer.study_name == "test_study"
        assert optimizer.n_trials == 10
        assert optimizer.timeout_seconds == 60
        assert optimizer.study is None
        assert optimizer.best_params is None
        assert optimizer.best_model is None

    def test_initialization_with_invalid_model_type(self):
        """Test that HyperparameterOptimizer raises ValueError for invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type: invalid_model"):
            HyperparameterOptimizer(
                model_type="invalid_model",
                study_name="test_study",
                n_trials=10,
            )

    def test_optimize_runs_without_error(self, sample_data):
        """Test that optimize() runs without error on small dataset."""
        x_train, y_train, x_val, y_val = sample_data

        optimizer = HyperparameterOptimizer(
            model_type="xgboost", study_name="test_study", n_trials=5, timeout_seconds=60
        )

        # Should not raise any exception
        best_params, best_model = optimizer.optimize(x_train, y_train, x_val, y_val)

        # Verify optimization completed
        assert best_params is not None
        assert optimizer.study is not None

    def test_optimize_returns_best_params_dict(self, sample_data):
        """Test that optimize() returns best_params as dict and model."""
        x_train, y_train, x_val, y_val = sample_data

        optimizer = HyperparameterOptimizer(
            model_type="xgboost", study_name="test_study", n_trials=5, timeout_seconds=60
        )

        best_params, best_model = optimizer.optimize(x_train, y_train, x_val, y_val)

        assert isinstance(best_params, dict)
        assert len(best_params) > 0

    def test_get_study_stats_returns_dict_with_n_trials(self, sample_data):
        """Test that get_study_stats() returns dict with n_trials after optimization."""
        x_train, y_train, x_val, y_val = sample_data

        optimizer = HyperparameterOptimizer(
            model_type="xgboost", study_name="test_study", n_trials=5, timeout_seconds=60
        )

        # Before optimization, should return empty dict
        stats = optimizer.get_study_stats()
        assert stats == {}

        # After optimization
        optimizer.optimize(x_train, y_train, x_val, y_val)
        stats = optimizer.get_study_stats()

        assert isinstance(stats, dict)
        assert "n_trials" in stats
        assert stats["n_trials"] == 5
        assert "best_value" in stats
        assert "best_trial" in stats

    def test_best_params_contains_expected_keys(self, sample_data):
        """Test that best_params contains expected hyperparameter keys for XGBoost."""
        x_train, y_train, x_val, y_val = sample_data

        optimizer = HyperparameterOptimizer(
            model_type="xgboost", study_name="test_study", n_trials=5, timeout_seconds=60
        )

        best_params, _ = optimizer.optimize(x_train, y_train, x_val, y_val)

        # Check for expected XGBoost hyperparameters
        expected_keys = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
        ]

        for key in expected_keys:
            assert key in best_params, f"Expected key '{key}' not found in best_params"

    def test_get_best_params_method(self, sample_data):
        """Test get_best_params() method returns correct parameters."""
        x_train, y_train, x_val, y_val = sample_data

        optimizer = HyperparameterOptimizer(
            model_type="xgboost", study_name="test_study", n_trials=5, timeout_seconds=60
        )

        # Before optimization
        assert optimizer.get_best_params() is None

        # After optimization
        optimizer.optimize(x_train, y_train, x_val, y_val)
        best_params = optimizer.get_best_params()

        assert best_params is not None
        assert isinstance(best_params, dict)
        assert len(best_params) > 0

    def test_optimize_with_different_model_types(self, sample_data):
        """Test that optimization works for different model types."""
        x_train, y_train, x_val, y_val = sample_data

        for model_type in ["xgboost", "lightgbm", "random_forest"]:
            optimizer = HyperparameterOptimizer(
                model_type=model_type,
                study_name=f"test_{model_type}",
                n_trials=3,
                timeout_seconds=30,
            )

            best_params, _ = optimizer.optimize(x_train, y_train, x_val, y_val)

            assert best_params is not None
            assert isinstance(best_params, dict)
            assert len(best_params) > 0
