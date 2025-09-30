"""Logistic regression model for F1 race outcome prediction.

This module provides a logistic regression classifier for predicting:
- Podium finish probability (top 3)
- Points finish probability (top 10)
- Race win probability
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)


class LogisticRacePredictor:
    """Logistic regression model for race outcome prediction.

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
        random_state: int = 42,
        max_iter: int = 1000,
    ):
        """Initialize logistic regression predictor.

        Args:
            target: Prediction target - "podium", "points", or "win"
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for solver
        """
        if target not in ["podium", "points", "win"]:
            msg = f"Invalid target: {target}. Must be 'podium', 'points', or 'win'"
            raise ValueError(msg)

        self.target = target
        self.random_state = random_state
        self.max_iter = max_iter

        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            solver="lbfgs",
        )
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_fitted = False

        self.logger = logger.bind(model="logistic_regression", target=target)

    def _prepare_target(self, race_results: pd.DataFrame) -> pd.Series:
        """Prepare target variable based on prediction type.

        Args:
            race_results: DataFrame with actual race positions

        Returns:
            Binary target series (1 for success, 0 for failure)
        """
        if self.target == "podium":
            return (race_results["position"] <= 3).astype(int)
        elif self.target == "points":
            return (race_results["position"] <= 10).astype(int)
        else:  # win
            return (race_results["position"] == 1).astype(int)

    def fit(
        self,
        features: pd.DataFrame,
        race_results: pd.DataFrame,
    ) -> "LogisticRacePredictor":
        """Train the logistic regression model.

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
            msg = f"Feature count ({len(features)}) != result count ({len(race_results)})"
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
        self.logger.info(
            "training_complete",
            train_accuracy=train_score,
            num_features=len(self.feature_names),
            positive_class_ratio=y.mean(),
        )

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
            return pd.DataFrame(columns=["driver_id", "predicted_outcome", "confidence"])

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
        """Get feature importance based on model coefficients.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            msg = "Model must be fitted to get feature importance"
            raise ValueError(msg)

        # Get absolute coefficients as importance
        coefficients = np.abs(self.model.coef_[0])

        # Normalize to sum to 1
        if coefficients.sum() > 0:
            coefficients = coefficients / coefficients.sum()

        return dict(zip(self.feature_names, coefficients.tolist(), strict=False))

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
            "random_state": self.random_state,
            "max_iter": self.max_iter,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        self.logger.info("model_saved", filepath=str(filepath))

    @classmethod
    def load(cls, filepath: Path | str) -> "LogisticRacePredictor":
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
            random_state=model_data["random_state"],
            max_iter=model_data["max_iter"],
        )
        predictor.model = model_data["model"]
        predictor.scaler = model_data["scaler"]
        predictor.feature_names = model_data["feature_names"]
        predictor.is_fitted = True

        logger.info("model_loaded", filepath=str(filepath))

        return predictor