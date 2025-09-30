"""Model evaluation and cross-validation for F1 prediction models.

This module provides:
- Cross-validation functionality
- Performance metrics (accuracy, precision, recall, F1)
- Confidence calibration metrics
- Model comparison utilities
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold

logger = structlog.get_logger(__name__)


class ModelEvaluator:
    """Evaluate and compare F1 prediction models.

    Provides comprehensive evaluation metrics:
    - Classification metrics (accuracy, precision, recall, F1)
    - ROC AUC score
    - Confidence calibration
    - Cross-validation support
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """Initialize model evaluator.

        Args:
            n_splits: Number of folds for cross-validation
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.logger = logger.bind(evaluator="model")

    def evaluate(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        y_proba: Optional[pd.Series | np.ndarray] = None,
    ) -> dict[str, float]:
        """Evaluate model predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for AUC)

        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("evaluating_predictions", num_samples=len(y_true))

        metrics: dict[str, float] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        # Add AUC if probabilities provided
        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            except ValueError:
                # Handle case where only one class present
                metrics["roc_auc"] = 0.0

        # Add support counts
        metrics["support"] = float(len(y_true))
        metrics["positive_ratio"] = float(np.mean(y_true))

        self.logger.info("evaluation_complete", **metrics)

        return metrics

    def cross_validate(
        self,
        model: Any,
        features: pd.DataFrame,
        race_results: pd.DataFrame,
    ) -> dict[str, Any]:
        """Perform k-fold cross-validation.

        Args:
            model: Model instance with fit/predict methods
            features: Feature DataFrame
            race_results: Race results DataFrame with positions

        Returns:
            Dictionary with cross-validation results
        """
        self.logger.info(
            "starting_cross_validation",
            n_splits=self.n_splits,
            num_samples=len(features),
        )

        if len(features) < self.n_splits:
            msg = f"Not enough samples ({len(features)}) for {self.n_splits}-fold CV"
            raise ValueError(msg)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        fold_metrics = []
        feature_importance_per_fold = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(features), 1):
            self.logger.debug("processing_fold", fold=fold_idx)

            # Split data
            X_train = features.iloc[train_idx]
            X_test = features.iloc[test_idx]
            y_train = race_results.iloc[train_idx]
            y_test = race_results.iloc[test_idx]

            # Train model
            model.fit(X_train, y_train)

            # Get predictions
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                y_pred = (y_proba >= 0.5).astype(int)
            else:
                predictions = model.predict(X_test)
                if "predicted_outcome" in predictions.columns:
                    y_pred = predictions["predicted_outcome"].values
                    y_proba = predictions["confidence"].values / 100
                else:
                    # Rule-based model with positions
                    # Convert to binary: top 3 = 1, else = 0
                    y_pred = (predictions["predicted_position"] <= 3).astype(int)
                    y_proba = predictions["confidence"].values / 100

            # Prepare true labels
            if hasattr(model, "_prepare_target"):
                y_true = model._prepare_target(y_test)
            else:
                # Assume podium prediction for rule-based
                y_true = (y_test["position"] <= 3).astype(int)

            # Evaluate fold
            fold_metric = self.evaluate(y_true, y_pred, y_proba)
            fold_metric["fold"] = fold_idx
            fold_metrics.append(fold_metric)

            # Get feature importance if available
            if hasattr(model, "get_feature_importance"):
                feature_importance_per_fold.append(model.get_feature_importance())

        # Aggregate results
        metrics_df = pd.DataFrame(fold_metrics)

        results = {
            "mean_metrics": {
                "accuracy": metrics_df["accuracy"].mean(),
                "precision": metrics_df["precision"].mean(),
                "recall": metrics_df["recall"].mean(),
                "f1_score": metrics_df["f1_score"].mean(),
                "roc_auc": metrics_df["roc_auc"].mean() if "roc_auc" in metrics_df else None,
            },
            "std_metrics": {
                "accuracy": metrics_df["accuracy"].std(),
                "precision": metrics_df["precision"].std(),
                "recall": metrics_df["recall"].std(),
                "f1_score": metrics_df["f1_score"].std(),
                "roc_auc": metrics_df["roc_auc"].std() if "roc_auc" in metrics_df else None,
            },
            "fold_metrics": fold_metrics,
            "n_splits": self.n_splits,
            "total_samples": len(features),
        }

        # Add aggregated feature importance if available
        if feature_importance_per_fold:
            # Average feature importance across folds
            all_features = set()
            for imp in feature_importance_per_fold:
                all_features.update(imp.keys())

            avg_importance = {}
            for feature in all_features:
                importances = [imp.get(feature, 0.0) for imp in feature_importance_per_fold]
                avg_importance[feature] = np.mean(importances)

            results["feature_importance"] = avg_importance

        self.logger.info(
            "cross_validation_complete",
            mean_accuracy=results["mean_metrics"]["accuracy"],
            mean_f1=results["mean_metrics"]["f1_score"],
        )

        return results

    def evaluate_confidence_calibration(
        self,
        y_true: pd.Series | np.ndarray,
        y_proba: pd.Series | np.ndarray,
        n_bins: int = 10,
    ) -> dict[str, Any]:
        """Evaluate confidence calibration.

        Measures how well predicted probabilities match actual outcomes.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration curve

        Returns:
            Dictionary with calibration metrics
        """
        self.logger.info("evaluating_calibration", n_bins=n_bins)

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)

        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Calculate calibration metrics per bin
        bin_metrics = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue

            bin_metric = {
                "bin": i,
                "bin_range": f"[{bins[i]:.2f}, {bins[i+1]:.2f})",
                "count": int(mask.sum()),
                "mean_predicted": float(y_proba[mask].mean()),
                "mean_actual": float(y_true[mask].mean()),
            }
            bin_metrics.append(bin_metric)

        # Calculate overall calibration error (Expected Calibration Error)
        ece = 0.0
        for metric in bin_metrics:
            weight = metric["count"] / len(y_true)
            ece += weight * abs(metric["mean_predicted"] - metric["mean_actual"])

        results = {
            "expected_calibration_error": ece,
            "n_bins": n_bins,
            "bin_metrics": bin_metrics,
        }

        self.logger.info("calibration_evaluation_complete", ece=ece)

        return results

    def compare_models(
        self,
        models: dict[str, Any],
        features: pd.DataFrame,
        race_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compare multiple models using cross-validation.

        Args:
            models: Dictionary mapping model names to model instances
            features: Feature DataFrame
            race_results: Race results DataFrame

        Returns:
            DataFrame comparing model performances
        """
        self.logger.info("comparing_models", num_models=len(models))

        comparison_results = []

        for model_name, model in models.items():
            self.logger.info("evaluating_model", model_name=model_name)

            try:
                cv_results = self.cross_validate(model, features, race_results)

                comparison_results.append(
                    {
                        "model": model_name,
                        "accuracy_mean": cv_results["mean_metrics"]["accuracy"],
                        "accuracy_std": cv_results["std_metrics"]["accuracy"],
                        "precision_mean": cv_results["mean_metrics"]["precision"],
                        "precision_std": cv_results["std_metrics"]["precision"],
                        "recall_mean": cv_results["mean_metrics"]["recall"],
                        "recall_std": cv_results["std_metrics"]["recall"],
                        "f1_mean": cv_results["mean_metrics"]["f1_score"],
                        "f1_std": cv_results["std_metrics"]["f1_score"],
                        "roc_auc_mean": cv_results["mean_metrics"].get("roc_auc"),
                        "roc_auc_std": cv_results["std_metrics"].get("roc_auc"),
                    }
                )
            except Exception as e:
                self.logger.error("model_evaluation_failed", model_name=model_name, error=str(e))
                continue

        comparison_df = pd.DataFrame(comparison_results)

        # Sort by F1 score
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values("f1_mean", ascending=False)

        self.logger.info("model_comparison_complete", num_models=len(comparison_df))

        return comparison_df