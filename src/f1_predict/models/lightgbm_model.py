"""LightGBM model for F1 race outcome prediction.

This module provides a LightGBM classifier for predicting:
- Podium finish probability (top 3)
- Points finish probability (top 10)
- Race win probability
"""

from pathlib import Path
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import structlog

logger = structlog.get_logger(__name__)


class LightGBMRacePredictor:
    """LightGBM model for race outcome prediction.

    Predicts probability of various race outcomes:
    - Podium finish (position <= 3)
    - Points finish (position <= 10)
    - Race win (position == 1)

    Features used:
    - Qualifying position
    - Driver form score
    - Team reliability score
    - Circuit performance score
    - Championship position (optional)
    """

    def __init__(
        self,
        target: str = "podium",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_data_in_leaf: int = 20,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        random_state: int = 42,
        early_stopping_rounds: int | None = 10,
        verbose: int = -1,
    ):
        """Initialize LightGBM predictor.

        Args:
            target: Prediction target - "podium", "points", or "win"
            learning_rate: Learning rate (default: 0.1)
            n_estimators: Number of boosting iterations (default: 100)
            num_leaves: Maximum tree leaves (default: 31)
            max_depth: Maximum tree depth (-1 for no limit)
            min_data_in_leaf: Minimum data in one leaf (default: 20)
            feature_fraction: Fraction of features to use (default: 0.8)
            bagging_fraction: Fraction of data to use for bagging (default: 0.8)
            bagging_freq: Frequency for bagging (default: 5)
            random_state: Random seed for reproducibility
            early_stopping_rounds: Rounds without improvement before stopping
            verbose: Verbosity level (-1 for silent)
        """
        if target not in ["podium", "points", "win"]:
            msg = f"Invalid target: {target}. Must be 'podium', 'points', or 'win'"
            raise ValueError(msg)

        self.target = target
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

        self.model: lgb.Booster | None = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_fitted = False
        self.best_iteration: int | None = None
        self.best_score: float | None = None

        self.logger = logger.bind(model="lightgbm", target=target)

    def _prepare_target(self, race_results: pd.DataFrame) -> pd.Series:
        """Prepare target variable based on prediction type.

        Args:
            race_results: DataFrame with actual race positions

        Returns:
            Binary target series (1 for success, 0 for failure)
        """
        if self.target == "podium":
            return (race_results["position"] <= 3).astype(int)
        if self.target == "points":
            return (race_results["position"] <= 10).astype(int)
        # win
        return (race_results["position"] == 1).astype(int)

    def fit(
        self,
        features: pd.DataFrame,
        race_results: pd.DataFrame,
        eval_set: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    ) -> "LightGBMRacePredictor":
        """Train the LightGBM model.

        Args:
            features: DataFrame with predictor features
            race_results: DataFrame with actual race positions
            eval_set: Optional tuple of (eval_features, eval_results) for validation

        Returns:
            Self for method chaining
        """
        self.logger.info("training_model", num_samples=len(features))

        if features.empty or race_results.empty:
            msg = "Cannot train on empty data"
            raise ValueError(msg)

        # Validate alignment
        if len(features) != len(race_results):
            msg = (
                f"Feature count ({len(features)}) != result count ({len(race_results)})"
            )
            raise ValueError(msg)

        # Prepare features
        feature_cols = [
            col
            for col in features.columns
            if col
            not in [
                "driver_id",
                "race_id",
                "season",
                "round",
                "position",
                "predicted_position",
                "confidence",
            ]
        ]
        X = features[feature_cols].values
        self.feature_names = feature_cols

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Prepare target
        y = self._prepare_target(race_results)

        # Create dataset
        train_data = lgb.Dataset(X_scaled, label=y, feature_name=self.feature_names)

        # Prepare evaluation set if provided
        valid_sets = [train_data]
        valid_names = ["train"]
        if eval_set is not None:
            eval_features, eval_results = eval_set
            eval_X = eval_features[self.feature_names].values
            eval_X_scaled = self.scaler.transform(eval_X)
            eval_y = self._prepare_target(eval_results)
            eval_data = lgb.Dataset(
                eval_X_scaled,
                label=eval_y,
                feature_name=self.feature_names,
                reference=train_data,
            )
            valid_sets.append(eval_data)
            valid_names.append("eval")

        # LightGBM parameters
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

        # Train model
        callbacks = []
        if self.early_stopping_rounds and eval_set is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        self.is_fitted = True

        # Store best iteration info
        if self.early_stopping_rounds and eval_set is not None:
            self.best_iteration = self.model.best_iteration
            self.best_score = self.model.best_score.get("eval", {}).get(
                "binary_logloss"
            )

        # Log training metrics
        train_pred = self.model.predict(X_scaled)
        train_accuracy = ((train_pred >= 0.5) == y).mean()

        log_data = {
            "train_accuracy": train_accuracy,
            "num_features": len(self.feature_names),
            "positive_class_ratio": y.mean(),
            "n_estimators": self.n_estimators,
        }

        if self.best_iteration is not None:
            log_data["best_iteration"] = self.best_iteration
            log_data["best_score"] = self.best_score

        self.logger.info("training_complete", **log_data)

        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict probability of target outcome.

        Args:
            features: DataFrame with predictor features

        Returns:
            Array of probabilities for positive class (success)
        """
        if not self.is_fitted or self.model is None:
            msg = "Model must be fitted before prediction"
            raise ValueError(msg)

        if features.empty:
            return np.array([])

        # Prepare features
        X = features[self.feature_names].values
        X_scaled = self.scaler.transform(X)

        # Predict probabilities
        probs = self.model.predict(X_scaled)

        return probs

    def predict(self, features: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Predict race outcomes with confidence scores.

        Args:
            features: DataFrame with predictor features
            threshold: Probability threshold for positive prediction

        Returns:
            DataFrame with predictions and confidence scores
        """
        if not self.is_fitted:
            msg = "Model must be fitted before prediction"
            raise ValueError(msg)

        self.logger.info("predicting", num_samples=len(features))

        if features.empty:
            return pd.DataFrame(
                columns=["driver_id", "predicted_outcome", "confidence"]
            )

        # Get probabilities
        probabilities = self.predict_proba(features)

        # Create predictions
        result = pd.DataFrame(
            {
                "driver_id": features["driver_id"].values,
                "predicted_outcome": (probabilities >= threshold).astype(int),
                "confidence": probabilities * 100,  # Convert to percentage
            }
        )

        self.logger.info(
            "prediction_complete",
            num_predictions=len(result),
            avg_confidence=result["confidence"].mean(),
            predicted_positive=result["predicted_outcome"].sum(),
        )

        return result

    def get_feature_importance(
        self, importance_type: str = "split"
    ) -> dict[str, float]:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance - "split" or "gain"

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted or self.model is None:
            msg = "Model must be fitted to get feature importance"
            raise ValueError(msg)

        if importance_type not in ["split", "gain"]:
            msg = f"Invalid importance_type: {importance_type}"
            raise ValueError(msg)

        # Get feature importance from model
        importance_values = self.model.feature_importance(importance_type=importance_type)

        # Create dictionary
        importance = dict(zip(self.feature_names, importance_values, strict=False))

        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    def save(self, filepath: Path | str) -> None:
        """Save trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted or self.model is None:
            msg = "Model must be fitted before saving"
            raise ValueError(msg)

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save LightGBM model separately
        model_file = filepath.with_suffix(".lgb")
        self.model.save_model(str(model_file))

        # Save metadata
        model_data = {
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "target": self.target,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "random_state": self.random_state,
            "early_stopping_rounds": self.early_stopping_rounds,
            "verbose": self.verbose,
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        self.logger.info("model_saved", filepath=str(filepath))

    @classmethod
    def load(cls, filepath: Path | str) -> "LightGBMRacePredictor":
        """Load trained model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded predictor instance
        """
        filepath = Path(filepath)

        if not filepath.exists():
            msg = f"Model file not found: {filepath}"
            raise FileNotFoundError(msg)

        # Load metadata
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        # Reconstruct predictor
        predictor = cls(
            target=model_data["target"],
            learning_rate=model_data["learning_rate"],
            n_estimators=model_data["n_estimators"],
            num_leaves=model_data["num_leaves"],
            max_depth=model_data["max_depth"],
            min_data_in_leaf=model_data["min_data_in_leaf"],
            feature_fraction=model_data["feature_fraction"],
            bagging_fraction=model_data["bagging_fraction"],
            bagging_freq=model_data["bagging_freq"],
            random_state=model_data["random_state"],
            early_stopping_rounds=model_data["early_stopping_rounds"],
            verbose=model_data["verbose"],
        )

        # Load LightGBM model
        model_file = filepath.with_suffix(".lgb")
        predictor.model = lgb.Booster(model_file=str(model_file))

        predictor.scaler = model_data["scaler"]
        predictor.feature_names = model_data["feature_names"]
        predictor.best_iteration = model_data.get("best_iteration")
        predictor.best_score = model_data.get("best_score")
        predictor.is_fitted = True

        logger.info("model_loaded", filepath=str(filepath))

        return predictor
