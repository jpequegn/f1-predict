"""Ensemble prediction system for F1 race outcomes.

This module provides ensemble methods combining multiple models:
- Voting Classifier (hard/soft voting)
- Weighted averaging
"""

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class EnsemblePredictor:
    """Ensemble predictor combining multiple models.

    Supports:
    - Soft voting (weighted average of probabilities)
    - Hard voting (majority vote of predictions)
    - Custom weights for each model
    """

    def __init__(
        self,
        models: list[Any],
        weights: list[float] | None = None,
        voting: str = "soft",
    ):
        """Initialize ensemble predictor.

        Args:
            models: List of fitted predictor models
            weights: Optional weights for each model (default: equal weights)
            voting: Voting strategy - "soft" (probabilities) or "hard" (predictions)
        """
        if not models:
            msg = "At least one model required for ensemble"
            raise ValueError(msg)

        if voting not in ["soft", "hard"]:
            msg = f"Invalid voting strategy: {voting}. Must be 'soft' or 'hard'"
            raise ValueError(msg)

        self.models = models
        self.voting = voting

        # Set weights (default to equal weights)
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                msg = f"Number of weights ({len(weights)}) must match number of models ({len(models)})"
                raise ValueError(msg)
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

        self.logger = logger.bind(
            model="ensemble", num_models=len(models), voting=voting
        )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict probability using ensemble.

        Args:
            features: DataFrame with predictor features

        Returns:
            Array of ensemble probabilities
        """
        if features.empty:
            return np.array([])

        # Get predictions from all models
        predictions = []
        for model in self.models:
            probs = model.predict_proba(features)
            predictions.append(probs)

        # Stack predictions
        predictions_array = np.column_stack(predictions)

        # Compute weighted average
        ensemble_probs = np.average(predictions_array, axis=1, weights=self.weights)

        return ensemble_probs

    def predict(self, features: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Predict race outcomes using ensemble.

        Args:
            features: DataFrame with predictor features
            threshold: Probability threshold for positive prediction (soft voting only)

        Returns:
            DataFrame with predictions and confidence scores
        """
        self.logger.info("predicting", num_samples=len(features))

        if features.empty:
            return pd.DataFrame(
                columns=["driver_id", "predicted_outcome", "confidence"]
            )

        if self.voting == "soft":
            # Use probability-based voting
            probabilities = self.predict_proba(features)
            predicted_outcome = (probabilities >= threshold).astype(int)
            confidence = probabilities * 100
        else:
            # Hard voting - get individual predictions and take majority
            individual_predictions = []
            for model in self.models:
                pred = model.predict(features, threshold=threshold)
                individual_predictions.append(pred["predicted_outcome"].values)

            predictions_array = np.column_stack(individual_predictions)

            # Weighted majority vote
            weighted_votes = np.dot(predictions_array, self.weights)
            predicted_outcome = (weighted_votes >= 0.5).astype(int)

            # Confidence is proportion of weighted votes
            confidence = weighted_votes * 100

        # Create result DataFrame
        result = pd.DataFrame(
            {
                "driver_id": features["driver_id"].values,
                "predicted_outcome": predicted_outcome,
                "confidence": confidence,
            }
        )

        self.logger.info(
            "prediction_complete",
            num_predictions=len(result),
            avg_confidence=result["confidence"].mean(),
            predicted_positive=result["predicted_outcome"].sum(),
        )

        return result

    def get_model_agreement(self, features: pd.DataFrame) -> float:
        """Calculate agreement between models.

        Args:
            features: DataFrame with predictor features

        Returns:
            Agreement score (0-1) indicating consensus
        """
        if features.empty:
            return 0.0

        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(features)
            predictions.append(pred["predicted_outcome"].values)

        predictions_array = np.column_stack(predictions)

        # Calculate agreement (proportion of unanimous decisions)
        all_agree = (predictions_array.sum(axis=1) == 0) | (
            predictions_array.sum(axis=1) == len(self.models)
        )
        agreement = all_agree.mean()

        return float(agreement)

    def save(self, filepath: Path | str) -> None:
        """Save ensemble to disk.

        Args:
            filepath: Path to save the ensemble
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save individual models
        model_dir = filepath.parent / f"{filepath.stem}_models"
        model_dir.mkdir(exist_ok=True)

        model_paths = []
        for i, model in enumerate(self.models):
            model_path = model_dir / f"model_{i}.pkl"
            model.save(model_path)
            model_paths.append(str(model_path))

        # Save ensemble metadata
        ensemble_data = {
            "model_paths": model_paths,
            "weights": self.weights,
            "voting": self.voting,
        }

        with open(filepath, "wb") as f:
            pickle.dump(ensemble_data, f)

        self.logger.info("ensemble_saved", filepath=str(filepath))

    @classmethod
    def load(cls, filepath: Path | str) -> "EnsemblePredictor":
        """Load ensemble from disk.

        Args:
            filepath: Path to the saved ensemble

        Returns:
            Loaded ensemble instance
        """
        filepath = Path(filepath)

        if not filepath.exists():
            msg = f"Ensemble file not found: {filepath}"
            raise FileNotFoundError(msg)

        # Load ensemble metadata
        with open(filepath, "rb") as f:
            ensemble_data = pickle.load(f)

        # Load individual models
        # Import here to avoid circular imports
        from f1_predict.models.lightgbm_model import LightGBMRacePredictor
        from f1_predict.models.logistic import LogisticRacePredictor
        from f1_predict.models.random_forest import RandomForestRacePredictor
        from f1_predict.models.xgboost_model import XGBoostRacePredictor

        models = []
        for model_path in ensemble_data["model_paths"]:
            # Try loading with each model type
            try:
                model = LogisticRacePredictor.load(model_path)
            except Exception:
                try:
                    model = RandomForestRacePredictor.load(model_path)
                except Exception:
                    try:
                        model = XGBoostRacePredictor.load(model_path)
                    except Exception:
                        model = LightGBMRacePredictor.load(model_path)
            models.append(model)

        # Reconstruct ensemble
        ensemble = cls(
            models=models,
            weights=ensemble_data["weights"],
            voting=ensemble_data["voting"],
        )

        logger.info("ensemble_loaded", filepath=str(filepath))

        return ensemble
