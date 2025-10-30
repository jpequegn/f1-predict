"""Search space definitions for hyperparameter optimization.

This module defines the hyperparameter search spaces for different model types
used in F1 race prediction. Each search space specifies the type, range, and
distribution for hyperparameters to be optimized.
"""

from typing import Any


class SearchSpaceRegistry:
    """Registry of hyperparameter search spaces for different model types.

    This class provides predefined search spaces for various machine learning
    models, including gradient boosting models (XGBoost, LightGBM) and
    ensemble methods (Random Forest).

    Attributes:
    ----------
    XGBOOST_SPACE : dict
        Search space for XGBoost hyperparameters
    LIGHTGBM_SPACE : dict
        Search space for LightGBM hyperparameters
    RANDOM_FOREST_SPACE : dict
        Search space for Random Forest hyperparameters
    SPACES : dict
        Dictionary mapping model type names to their search spaces
    """

    XGBOOST_SPACE: dict[str, Any] = {
        "n_estimators": {"type": "int", "low": 100, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "min_child_weight": {"type": "int", "low": 1, "high": 10},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 0.0, "high": 1.0},
    }

    LIGHTGBM_SPACE: dict[str, Any] = {
        "n_estimators": {"type": "int", "low": 100, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "min_child_weight": {"type": "int", "low": 1, "high": 10},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 0.0, "high": 1.0},
    }

    RANDOM_FOREST_SPACE: dict[str, Any] = {
        "n_estimators": {"type": "int", "low": 100, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2"]},
    }

    SPACES: dict[str, dict[str, Any]] = {
        "xgboost": XGBOOST_SPACE,
        "lightgbm": LIGHTGBM_SPACE,
        "random_forest": RANDOM_FOREST_SPACE,
    }

    @staticmethod
    def get_search_space(model_type: str) -> dict[str, Any]:
        """Get the hyperparameter search space for a given model type.

        Parameters
        ----------
        model_type : str
            The type of model to get the search space for.
            Must be one of: 'xgboost', 'lightgbm', 'random_forest'

        Returns:
        -------
        Dict[str, Any]
            Dictionary defining the hyperparameter search space.
            Each key is a hyperparameter name, and the value is a dictionary
            specifying the parameter type and range/choices.

        Raises:
        ------
        ValueError
            If the model_type is not recognized

        Examples:
        --------
        >>> space = SearchSpaceRegistry.get_search_space("xgboost")
        >>> space["n_estimators"]
        {'type': 'int', 'low': 100, 'high': 500}
        >>> space["learning_rate"]
        {'type': 'float', 'low': 0.001, 'high': 0.3, 'log': True}
        """
        if model_type not in SearchSpaceRegistry.SPACES:
            raise ValueError(f"Unknown model type: {model_type}")
        return SearchSpaceRegistry.SPACES[model_type]
