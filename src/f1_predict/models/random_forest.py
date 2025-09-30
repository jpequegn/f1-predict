"""Random Forest model for F1 race outcome prediction.

This module provides a Random Forest classifier for predicting:
- Podium finish probability (top 3)
- Points finish probability (top 10)
- Race win probability
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)


class RandomForestRacePredictor:
    """Random Forest model for race outcome prediction.

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
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | float = "sqrt",
        random_state: int = 42,
        oob_score: bool = True,
        n_jobs: int = -1,
    ):
        """Initialize Random Forest predictor.

        Args:
            target: Prediction target - "podium", "points", or "win"
            n_estimators: Number of trees in the forest (default: 100)
            max_depth: Maximum tree depth (None for unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in leaf node
            max_features: Number of features for best split
            random_state: Random seed for reproducibility
            oob_score: Whether to use out-of-bag samples for validation
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        if target not in ["podium", "points", "win"]:
            msg = f"Invalid target: {target}. Must be 'podium', 'points', or 'win'"
            raise ValueError(msg)

        self.target = target
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.oob_score_enabled = oob_score
        self.n_jobs = n_jobs

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            oob_score=oob_score,
            n_jobs=n_jobs,
        )
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_fitted = False

        self.logger = logger.bind(model="random_forest", target=target)

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
    ) -> "RandomForestRacePredictor":
        """Train the Random Forest model.

        Args:
            features: DataFrame with predictor features
            race_results: DataFrame with actual race positions

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

        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Log training metrics
        train_score = self.model.score(X_scaled, y)
        log_data = {
            "train_accuracy": train_score,
            "num_features": len(self.feature_names),
            "positive_class_ratio": y.mean(),
            "n_estimators": self.n_estimators,
        }

        # Add OOB score if enabled
        if self.oob_score_enabled:
            log_data["oob_score"] = self.model.oob_score_

        self.logger.info("training_complete", **log_data)

        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict probability of target outcome.

        Args:
            features: DataFrame with predictor features

        Returns:
            Array of probabilities for positive class (success)
        """
        if not self.is_fitted:
            msg = "Model must be fitted before prediction"
            raise ValueError(msg)

        if features.empty:
            return np.array([])

        # Prepare features
        X = features[self.feature_names].values
        X_scaled = self.scaler.transform(X)

        # Predict probabilities (return only positive class probability)
        probs = self.model.predict_proba(X_scaled)[:, 1]

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

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance based on tree impurity reduction.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            msg = "Model must be fitted to get feature importance"
            raise ValueError(msg)

        # Get feature importances from the model
        importances = self.model.feature_importances_

        # Normalize to sum to 1 (should already be normalized)
        if importances.sum() > 0:
            importances = importances / importances.sum()

        return dict(zip(self.feature_names, importances.tolist(), strict=False))

    def get_oob_score(self) -> float | None:
        """Get out-of-bag score if enabled.

        Returns:
            OOB score or None if not enabled
        """
        if not self.is_fitted:
            msg = "Model must be fitted to get OOB score"
            raise ValueError(msg)

        if not self.oob_score_enabled:
            return None

        return float(self.model.oob_score_)

    def save(self, filepath: Path | str) -> None:
        """Save trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            msg = "Model must be fitted before saving"
            raise ValueError(msg)

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "target": self.target,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "random_state": self.random_state,
            "oob_score_enabled": self.oob_score_enabled,
            "n_jobs": self.n_jobs,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        self.logger.info("model_saved", filepath=str(filepath))

    @classmethod
    def load(cls, filepath: Path | str) -> "RandomForestRacePredictor":
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

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        # Reconstruct predictor
        predictor = cls(
            target=model_data["target"],
            n_estimators=model_data["n_estimators"],
            max_depth=model_data["max_depth"],
            min_samples_split=model_data["min_samples_split"],
            min_samples_leaf=model_data["min_samples_leaf"],
            max_features=model_data["max_features"],
            random_state=model_data["random_state"],
            oob_score=model_data["oob_score_enabled"],
            n_jobs=model_data["n_jobs"],
        )
        predictor.model = model_data["model"]
        predictor.scaler = model_data["scaler"]
        predictor.feature_names = model_data["feature_names"]
        predictor.is_fitted = True

        logger.info("model_loaded", filepath=str(filepath))

        return predictor
