"""Objective functions for hyperparameter optimization."""

import logging

from lightgbm import LGBMClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class ObjectiveFunction:
    """Objective function for hyperparameter optimization trials."""

    @staticmethod
    def optimize_xgboost(trial, x_train, y_train, x_val, y_val) -> float:
        """XGBoost objective function.

        Args:
            trial: Optuna trial object for suggesting hyperparameters
            x_train: Training features
            y_train: Training labels
            x_val: Validation features
            y_val: Validation labels

        Returns:
            float: Accuracy score on validation set, or NaN if trial fails
        """
        try:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.001, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            }

            model = XGBClassifier(
                **params, random_state=42, verbosity=0, eval_metric="logloss"
            )
            model.fit(x_train, y_train, verbose=False)
            y_pred = model.predict(x_val)
            return accuracy_score(y_val, y_pred)

        except Exception as e:
            logger.warning(f"XGBoost trial failed: {e}")
            return np.nan

    @staticmethod
    def optimize_lightgbm(trial, x_train, y_train, x_val, y_val) -> float:
        """LightGBM objective function.

        Args:
            trial: Optuna trial object for suggesting hyperparameters
            x_train: Training features
            y_train: Training labels
            x_val: Validation features
            y_val: Validation labels

        Returns:
            float: Accuracy score on validation set, or NaN if trial fails
        """
        try:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.001, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            }
            model = LGBMClassifier(**params, random_state=42, verbose=-1)
            model.fit(x_train, y_train, verbose=False)
            y_pred = model.predict(x_val)
            return accuracy_score(y_val, y_pred)
        except Exception as e:
            logger.warning(f"LightGBM trial failed: {e}")
            return np.nan

    @staticmethod
    def optimize_random_forest(trial, x_train, y_train, x_val, y_val) -> float:
        """RandomForest objective function.

        Args:
            trial: Optuna trial object for suggesting hyperparameters
            x_train: Training features
            y_train: Training labels
            x_val: Validation features
            y_val: Validation labels

        Returns:
            float: Accuracy score on validation set, or NaN if trial fails
        """
        try:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2"]
                ),
            }
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)
            return accuracy_score(y_val, y_pred)
        except Exception as e:
            logger.warning(f"RandomForest trial failed: {e}")
            return np.nan
