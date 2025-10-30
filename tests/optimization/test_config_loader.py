"""Tests for hyperparameter configuration loading and persistence."""

import json
from pathlib import Path
from typing import Any

import pytest

from f1_predict.optimization.config_loader import (
    DEFAULT_HYPERPARAMETERS,
    ConfigLoader,
)


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Create temporary directory for config files."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


@pytest.fixture
def sample_params() -> dict[str, Any]:
    """Sample optimized parameters."""
    return {
        "n_estimators": 200,
        "max_depth": 10,
        "learning_rate": 0.05,
        "subsample": 0.9,
    }


class TestConfigLoaderSave:
    """Tests for saving hyperparameters."""

    def test_save_best_params_creates_file(
        self, temp_config_dir: Path, sample_params: dict[str, Any]
    ) -> None:
        """Test that save_best_params creates JSON file."""
        filepath = temp_config_dir / "xgboost_best.json"
        ConfigLoader.save_best_params("xgboost", sample_params, filepath)

        assert filepath.exists()
        assert filepath.is_file()

    def test_save_best_params_correct_content(
        self, temp_config_dir: Path, sample_params: dict[str, Any]
    ) -> None:
        """Test that saved file contains correct content."""
        filepath = temp_config_dir / "xgboost_best.json"
        ConfigLoader.save_best_params("xgboost", sample_params, filepath)

        with open(filepath) as f:
            loaded = json.load(f)

        assert loaded == sample_params

    def test_save_best_params_creates_parent_dirs(
        self, tmp_path: Path, sample_params: dict[str, Any]
    ) -> None:
        """Test that save_best_params creates parent directories."""
        filepath = tmp_path / "nested" / "deep" / "config.json"
        ConfigLoader.save_best_params("xgboost", sample_params, filepath)

        assert filepath.exists()
        assert filepath.parent.exists()


class TestConfigLoaderLoad:
    """Tests for loading hyperparameters."""

    def test_load_best_params_reads_file(
        self, temp_config_dir: Path, sample_params: dict[str, Any]
    ) -> None:
        """Test that load_best_params reads JSON file correctly."""
        filepath = temp_config_dir / "xgboost_best.json"
        with open(filepath, "w") as f:
            json.dump(sample_params, f)

        loaded = ConfigLoader.load_best_params("xgboost", filepath)

        assert loaded == sample_params

    def test_load_best_params_nonexistent_file(
        self, temp_config_dir: Path
    ) -> None:
        """Test that load_best_params returns None for nonexistent file."""
        filepath = temp_config_dir / "nonexistent.json"
        loaded = ConfigLoader.load_best_params("xgboost", filepath)

        assert loaded is None

    def test_load_best_params_invalid_json(
        self, temp_config_dir: Path
    ) -> None:
        """Test that load_best_params handles invalid JSON gracefully."""
        filepath = temp_config_dir / "invalid.json"
        with open(filepath, "w") as f:
            f.write("not valid json {")

        loaded = ConfigLoader.load_best_params("xgboost", filepath)

        assert loaded is None


class TestConfigLoaderGet:
    """Tests for getting hyperparameters with fallback logic."""

    def test_get_hyperparameters_returns_defaults_no_optimized(self) -> None:
        """Test that get_hyperparameters returns defaults when no optimized config."""
        params = ConfigLoader.get_hyperparameters("xgboost")

        assert params == DEFAULT_HYPERPARAMETERS["xgboost"]
        # Verify it's a copy, not the original
        params["n_estimators"] = 999
        assert DEFAULT_HYPERPARAMETERS["xgboost"]["n_estimators"] != 999

    def test_get_hyperparameters_returns_optimized_when_available(
        self, temp_config_dir: Path, sample_params: dict[str, Any]
    ) -> None:
        """Test that get_hyperparameters returns optimized params when available."""
        filepath = temp_config_dir / "xgboost_best.json"
        with open(filepath, "w") as f:
            json.dump(sample_params, f)

        params = ConfigLoader.get_hyperparameters("xgboost", filepath)

        assert params == sample_params

    def test_get_hyperparameters_falls_back_to_defaults(
        self, temp_config_dir: Path
    ) -> None:
        """Test fallback to defaults when optimized config doesn't exist."""
        filepath = temp_config_dir / "nonexistent.json"
        params = ConfigLoader.get_hyperparameters("xgboost", filepath)

        assert params == DEFAULT_HYPERPARAMETERS["xgboost"]

    def test_get_hyperparameters_unknown_model(self) -> None:
        """Test handling of unknown model type."""
        params = ConfigLoader.get_hyperparameters("unknown_model")

        assert params == {}


class TestDefaultHyperparameters:
    """Tests for DEFAULT_HYPERPARAMETERS structure."""

    def test_default_hyperparameters_has_all_models(self) -> None:
        """Test that DEFAULT_HYPERPARAMETERS has all 3 models."""
        assert "xgboost" in DEFAULT_HYPERPARAMETERS
        assert "lightgbm" in DEFAULT_HYPERPARAMETERS
        assert "random_forest" in DEFAULT_HYPERPARAMETERS

    def test_default_hyperparameters_xgboost_structure(self) -> None:
        """Test XGBoost default parameters structure."""
        xgb_params = DEFAULT_HYPERPARAMETERS["xgboost"]
        expected_keys = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
        }
        assert set(xgb_params.keys()) == expected_keys

    def test_default_hyperparameters_lightgbm_structure(self) -> None:
        """Test LightGBM default parameters structure."""
        lgb_params = DEFAULT_HYPERPARAMETERS["lightgbm"]
        expected_keys = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
        }
        assert set(lgb_params.keys()) == expected_keys

    def test_default_hyperparameters_random_forest_structure(self) -> None:
        """Test Random Forest default parameters structure."""
        rf_params = DEFAULT_HYPERPARAMETERS["random_forest"]
        expected_keys = {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
        }
        assert set(rf_params.keys()) == expected_keys
