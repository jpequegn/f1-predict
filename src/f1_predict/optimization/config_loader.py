"""Configuration loader for hyperparameter persistence and defaults."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default hyperparameters for each model type
DEFAULT_HYPERPARAMETERS = {
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    "lightgbm": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
}


class ConfigLoader:
    """Manage loading and saving of optimized hyperparameters."""

    @staticmethod
    def save_best_params(
        model_type: str,
        best_params: dict[str, Any],
        filepath: Path,
    ) -> None:
        """Save optimized hyperparameters to file.

        Args:
            model_type: Type of model (xgboost, lightgbm, random_forest)
            best_params: Dictionary of optimized hyperparameters
            filepath: Path to save the configuration file

        """
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(best_params, f, indent=2)
            logger.info(f"Saved {model_type} best params to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save best params: {e}")

    @staticmethod
    def load_best_params(
        model_type: str,
        filepath: Path,
    ) -> Optional[dict[str, Any]]:
        """Load optimized hyperparameters from file.

        Args:
            model_type: Type of model (xgboost, lightgbm, random_forest)
            filepath: Path to the configuration file

        Returns:
            Dictionary of hyperparameters if file exists and is valid, None otherwise

        """
        if not filepath.exists():
            logger.debug(f"No optimized params found at {filepath}")
            return None

        try:
            with open(filepath) as f:
                params = json.load(f)
            logger.info(f"Loaded {model_type} params from {filepath}")
            return params
        except Exception as e:
            logger.error(f"Failed to load params from {filepath}: {e}")
            return None

    @staticmethod
    def get_hyperparameters(
        model_type: str,
        optimized_config_path: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Get hyperparameters, preferring optimized ones if available.

        Args:
            model_type: Type of model (xgboost, lightgbm, random_forest)
            optimized_config_path: Optional path to optimized configuration file

        Returns:
            Dictionary of hyperparameters (optimized if available, defaults otherwise)

        """
        # Try to load optimized params first
        if optimized_config_path:
            optimized = ConfigLoader.load_best_params(
                model_type, optimized_config_path
            )
            if optimized:
                return optimized

        # Fall back to defaults
        if model_type in DEFAULT_HYPERPARAMETERS:
            return DEFAULT_HYPERPARAMETERS[model_type].copy()

        logger.warning(f"Unknown model type {model_type}, returning empty dict")
        return {}
