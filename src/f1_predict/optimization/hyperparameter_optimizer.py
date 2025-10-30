"""Main hyperparameter optimizer using Optuna."""

import logging
from typing import Any, Optional

import numpy as np
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

from f1_predict.optimization.objectives import ObjectiveFunction
from f1_predict.optimization.search_spaces import SearchSpaceRegistry

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Orchestrate hyperparameter optimization using Optuna.

    This class manages the hyperparameter optimization workflow using Optuna's
    Bayesian optimization with Tree-structured Parzen Estimator (TPE) sampling
    and Successive Halving pruning.

    Attributes:
    ----------
    model_type : str
        Type of model to optimize ('xgboost', 'lightgbm', 'random_forest')
    study_name : str
        Name identifier for the optimization study
    n_trials : int
        Number of optimization trials to run
    timeout_seconds : Optional[int]
        Maximum time in seconds for optimization (None for unlimited)
    study : Optional[optuna.Study]
        Optuna study object after optimization
    best_params : Optional[Dict[str, Any]]
        Best hyperparameters found during optimization
    best_model : Optional[Any]
        Best trained model (currently unused, reserved for future use)
    """

    OBJECTIVE_FUNCTIONS: dict[str, Any] = {
        "xgboost": ObjectiveFunction.optimize_xgboost,
        "lightgbm": ObjectiveFunction.optimize_lightgbm,
        "random_forest": ObjectiveFunction.optimize_random_forest,
    }

    def __init__(
        self,
        model_type: str,
        study_name: str,
        n_trials: int = 100,
        timeout_seconds: Optional[int] = 3600,
    ) -> None:
        """Initialize HyperparameterOptimizer.

        Parameters
        ----------
        model_type : str
            Type of model to optimize. Must be one of: 'xgboost', 'lightgbm',
            'random_forest'
        study_name : str
            Name identifier for the optimization study
        n_trials : int, default=100
            Number of optimization trials to run
        timeout_seconds : Optional[int], default=3600
            Maximum time in seconds for optimization. None for unlimited.

        Raises:
        ------
        ValueError
            If model_type is not recognized
        """
        if model_type not in SearchSpaceRegistry.SPACES:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model_type = model_type
        self.study_name = study_name
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.objective_fn = self.OBJECTIVE_FUNCTIONS[model_type]
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[dict[str, Any]] = None
        self.best_model: Optional[Any] = None

    def optimize(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[dict[str, Any], Optional[Any]]:
        """Run hyperparameter optimization.

        Executes Bayesian optimization using TPE sampling and Successive Halving
        pruning to find optimal hyperparameters for the specified model type.

        Parameters
        ----------
        x_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        x_val : np.ndarray
            Validation features
        y_val : np.ndarray
            Validation labels

        Returns:
        -------
        tuple[Dict[str, Any], Optional[Any]]
            Tuple of (best_params, best_model). Currently best_model is None.
        """
        logger.info(f"Starting optimization for {self.model_type}")
        logger.info(f"Study: {self.study_name}, Trials: {self.n_trials}")

        # Create study with Bayesian optimization
        sampler = TPESampler(seed=42)
        pruner = SuccessiveHalvingPruner()

        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=self.study_name,
        )

        # Run optimization
        def objective(trial: optuna.Trial) -> float:
            return self.objective_fn(trial, x_train, y_train, x_val, y_val)

        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout_seconds,
            show_progress_bar=False,
        )

        # Extract best parameters
        self.best_params = self.study.best_params
        logger.info(f"Best validation accuracy: {self.study.best_value:.4f}")
        logger.info(f"Best hyperparameters: {self.best_params}")

        return self.best_params, self.best_model

    def get_best_params(self) -> Optional[dict[str, Any]]:
        """Get best hyperparameters found during optimization.

        Returns:
        -------
        Optional[Dict[str, Any]]
            Best hyperparameters, or None if optimization hasn't run yet
        """
        return self.best_params

    def get_study_stats(self) -> dict[str, Any]:
        """Get statistics about the optimization study.

        Returns:
        -------
        Dict[str, Any]
            Dictionary containing study statistics including:
            - n_trials: Number of completed trials
            - best_value: Best objective value achieved
            - best_trial: Trial number of best result
            Empty dict if optimization hasn't run yet.
        """
        if not self.study:
            return {}

        return {
            "n_trials": len(self.study.trials),
            "best_value": self.study.best_value,
            "best_trial": self.study.best_trial.number,
        }
