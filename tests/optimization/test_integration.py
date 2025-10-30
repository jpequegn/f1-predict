"""Integration tests for full hyperparameter optimization pipeline."""

# ruff: noqa: N806  # Allow uppercase variable names (ML convention for X, y)

import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from f1_predict.optimization.hyperparameter_optimizer import (
    HyperparameterOptimizer,
)


class TestOptimizationIntegration:
    """Integration tests for full optimization pipeline."""

    @pytest.fixture
    def sample_data(self) -> tuple:
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=500, n_features=20, n_classes=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def test_full_optimization_workflow_xgboost(self, sample_data: tuple) -> None:
        """Test complete optimization workflow with XGBoost."""
        X_train, X_val, _, y_train, y_val, _ = sample_data

        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="integration_test_xgb",
            n_trials=3,
            timeout_seconds=300,
        )

        best_params, _ = optimizer.optimize(X_train, y_train, X_val, y_val)

        # Verify results
        assert isinstance(best_params, dict)
        assert len(best_params) > 0
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert isinstance(best_params["n_estimators"], int)
        assert isinstance(best_params["max_depth"], int)

    def test_full_optimization_workflow_lightgbm(self, sample_data: tuple) -> None:
        """Test complete optimization workflow with LightGBM."""
        X_train, X_val, _, y_train, y_val, _ = sample_data

        optimizer = HyperparameterOptimizer(
            model_type="lightgbm",
            study_name="integration_test_lgb",
            n_trials=3,
            timeout_seconds=300,
        )

        best_params, _ = optimizer.optimize(X_train, y_train, X_val, y_val)

        assert isinstance(best_params, dict)
        assert len(best_params) > 0
        assert "n_estimators" in best_params

    def test_full_optimization_workflow_random_forest(self, sample_data: tuple) -> None:
        """Test complete optimization workflow with Random Forest."""
        X_train, X_val, _, y_train, y_val, _ = sample_data

        optimizer = HyperparameterOptimizer(
            model_type="random_forest",
            study_name="integration_test_rf",
            n_trials=3,
            timeout_seconds=300,
        )

        best_params, _ = optimizer.optimize(X_train, y_train, X_val, y_val)

        assert isinstance(best_params, dict)
        assert len(best_params) > 0
        assert "n_estimators" in best_params

    def test_optimization_returns_valid_params(self, sample_data: tuple) -> None:
        """Test that optimization returns valid hyperparameters."""
        X_train, X_val, _, y_train, y_val, _ = sample_data

        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="integration_test_valid",
            n_trials=2,
            timeout_seconds=300,
        )

        best_params, _ = optimizer.optimize(X_train, y_train, X_val, y_val)

        # Validate parameter ranges
        assert 100 <= best_params["n_estimators"] <= 500
        assert 3 <= best_params["max_depth"] <= 10
        assert 0.001 <= best_params["learning_rate"] <= 0.3

    def test_optimization_study_stats(self, sample_data: tuple) -> None:
        """Test that study statistics are tracked correctly."""
        X_train, X_val, _, y_train, y_val, _ = sample_data

        optimizer = HyperparameterOptimizer(
            model_type="xgboost",
            study_name="integration_test_stats",
            n_trials=3,
            timeout_seconds=300,
        )

        optimizer.optimize(X_train, y_train, X_val, y_val)
        stats = optimizer.get_study_stats()

        assert "n_trials" in stats
        assert "best_value" in stats
        assert "best_trial" in stats
        assert stats["n_trials"] > 0
