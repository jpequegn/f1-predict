"""SHAP-based model explainability for F1 race predictions.

This module provides SHAP (SHapley Additive exPlanations) integration for all
model types in the F1 predictor, offering local and global interpretability,
feature importance, and counterfactual analysis.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap
import structlog

logger = structlog.get_logger(__name__)


class SHAPExplainer:
    """SHAP-based explainer for F1 prediction models.

    Provides comprehensive explanations for all model types:
    - RuleBasedPredictor: Simple feature importance
    - LogisticRacePredictor: Linear SHAP explainer
    - RandomForestRacePredictor: Tree SHAP explainer
    - XGBoost/LightGBM models: Tree SHAP explainer

    Features:
    - Local explanations (single prediction)
    - Global explanations (dataset-wide patterns)
    - Counterfactual analysis ("what-if" scenarios)
    - Explanation caching for performance
    """

    def __init__(
        self,
        model: Any,
        model_type: str,
        feature_names: list[str],
        background_data: Optional[pd.DataFrame] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize SHAP explainer for a given model.

        Args:
            model: Trained prediction model
            model_type: Type of model ("rule_based", "logistic", "random_forest",
                        "xgboost", "lightgbm")
            feature_names: List of feature names used by the model
            background_data: Sample data for SHAP background distribution (optional)
            cache_dir: Directory for caching SHAP explanations (optional)

        Raises:
            ValueError: If model_type is not supported
        """
        supported_types = [
            "rule_based",
            "logistic",
            "random_forest",
            "xgboost",
            "lightgbm",
        ]
        if model_type not in supported_types:
            msg = f"Unsupported model type: {model_type}. Must be one of {supported_types}"
            raise ValueError(msg)

        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.background_data = background_data
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.explainer: Optional[shap.Explainer] = None
        self.logger = logger.bind(model_type=model_type)

        # Initialize SHAP explainer
        self._initialize_explainer()

        # Create cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("cache_enabled", cache_dir=str(self.cache_dir))

    def _initialize_explainer(self) -> None:
        """Initialize appropriate SHAP explainer for model type."""
        self.logger.info("initializing_shap_explainer", model_type=self.model_type)

        try:
            if self.model_type == "rule_based":
                # Rule-based models: use feature importance weights directly
                self.explainer = None  # We'll handle this specially
                self.logger.info("rule_based_explainer_initialized")

            elif self.model_type == "logistic":
                # Linear model: use LinearExplainer
                if hasattr(self.model, "model"):
                    sklearn_model = self.model.model
                else:
                    sklearn_model = self.model

                self.explainer = shap.LinearExplainer(
                    sklearn_model, self.background_data
                )
                self.logger.info("linear_explainer_initialized")

            elif self.model_type in ["random_forest", "xgboost", "lightgbm"]:
                # Tree-based models: use TreeExplainer
                if hasattr(self.model, "model"):
                    sklearn_model = self.model.model
                else:
                    sklearn_model = self.model

                self.explainer = shap.TreeExplainer(sklearn_model)
                self.logger.info("tree_explainer_initialized")

        except Exception as e:
            self.logger.error("explainer_initialization_failed", error=str(e))
            raise

    def explain_prediction(
        self,
        features: pd.DataFrame,
        cache_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate SHAP explanation for a single prediction.

        Args:
            features: DataFrame with features for one prediction
            cache_key: Optional key for caching (e.g., "driver_name_race_id")

        Returns:
            Dictionary containing:
                - shap_values: SHAP values for each feature
                - base_value: Expected value (baseline)
                - feature_values: Actual feature values
                - feature_names: Feature names
                - top_features: Top 5 contributing features
        """
        if len(features) != 1:
            msg = f"explain_prediction expects single row, got {len(features)}"
            raise ValueError(msg)

        self.logger.info("explaining_prediction", cache_key=cache_key)

        # Check cache
        if cache_key and self.cache_dir:
            cached = self._load_from_cache(cache_key, "prediction")
            if cached:
                self.logger.info("cache_hit", cache_key=cache_key)
                return cached

        # Prepare features
        X = features[self.feature_names].values

        # Calculate SHAP values
        if self.model_type == "rule_based":
            # For rule-based: use feature importance as proxy for SHAP values
            explanation = self._explain_rule_based(features)
        else:
            # For ML models: use SHAP explainer
            shap_values = self.explainer(X)

            explanation = {
                "shap_values": shap_values.values[0].tolist(),
                "base_value": float(shap_values.base_values[0]),
                "feature_values": X[0].tolist(),
                "feature_names": self.feature_names,
                "model_output": float(shap_values.base_values[0] + shap_values.values[0].sum()),
            }

        # Add top features
        explanation["top_features"] = self._get_top_features(
            explanation["shap_values"],
            explanation["feature_names"],
            explanation["feature_values"],
        )

        # Cache result
        if cache_key and self.cache_dir:
            self._save_to_cache(cache_key, "prediction", explanation)

        self.logger.info("explanation_generated", cache_key=cache_key)
        return explanation

    def explain_dataset(
        self,
        features: pd.DataFrame,
        sample_size: Optional[int] = None,
    ) -> dict[str, Any]:
        """Generate global SHAP explanation for a dataset.

        Args:
            features: DataFrame with features for multiple predictions
            sample_size: Number of samples to use (None for all)

        Returns:
            Dictionary containing:
                - mean_abs_shap: Mean absolute SHAP values per feature
                - feature_importance: Normalized feature importance scores
                - shap_values: SHAP values for all samples
                - feature_names: Feature names
        """
        self.logger.info(
            "explaining_dataset",
            num_samples=len(features),
            sample_size=sample_size,
        )

        # Sample if needed
        if sample_size and len(features) > sample_size:
            features = features.sample(n=sample_size, random_state=42)
            self.logger.info("dataset_sampled", final_size=len(features))

        # Prepare features
        X = features[self.feature_names].values

        # Calculate SHAP values
        if self.model_type == "rule_based":
            # For rule-based: use feature importance directly
            importance = self.model.get_feature_importance()
            feature_importance = np.array([importance.get(f, 0) for f in self.feature_names])

            explanation = {
                "mean_abs_shap": feature_importance.tolist(),
                "feature_importance": feature_importance.tolist(),
                "feature_names": self.feature_names,
            }
        else:
            # For ML models: calculate SHAP values
            shap_values = self.explainer(X)

            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

            # Normalize to get feature importance
            feature_importance = mean_abs_shap / mean_abs_shap.sum()

            explanation = {
                "mean_abs_shap": mean_abs_shap.tolist(),
                "feature_importance": feature_importance.tolist(),
                "shap_values": shap_values.values.tolist(),
                "feature_names": self.feature_names,
            }

        self.logger.info("dataset_explanation_generated")
        return explanation

    def what_if_analysis(
        self,
        base_features: pd.DataFrame,
        feature_changes: dict[str, float],
    ) -> dict[str, Any]:
        """Perform counterfactual "what-if" analysis.

        Args:
            base_features: Original feature values (single row)
            feature_changes: Dictionary of feature changes {feature_name: new_value}

        Returns:
            Dictionary containing:
                - base_prediction: Original prediction explanation
                - modified_prediction: Modified prediction explanation
                - shap_delta: Change in SHAP values
                - feature_delta: Change in feature values
                - prediction_delta: Change in model output
        """
        if len(base_features) != 1:
            msg = f"what_if_analysis expects single row, got {len(base_features)}"
            raise ValueError(msg)

        self.logger.info(
            "performing_what_if_analysis",
            num_changes=len(feature_changes),
        )

        # Get base prediction explanation
        base_explanation = self.explain_prediction(base_features)

        # Create modified features
        modified_features = base_features.copy()
        for feature, value in feature_changes.items():
            if feature not in self.feature_names:
                self.logger.warning("unknown_feature", feature=feature)
                continue
            modified_features[feature] = value

        # Get modified prediction explanation
        modified_explanation = self.explain_prediction(modified_features)

        # Calculate deltas
        shap_delta = [
            mod - base
            for mod, base in zip(
                modified_explanation["shap_values"],
                base_explanation["shap_values"],
                strict=True,
            )
        ]

        feature_delta = {
            feature: modified_features[feature].values[0] - base_features[feature].values[0]
            for feature in self.feature_names
        }

        prediction_delta = (
            modified_explanation["model_output"] - base_explanation["model_output"]
        )

        result = {
            "base_prediction": base_explanation,
            "modified_prediction": modified_explanation,
            "shap_delta": shap_delta,
            "feature_delta": feature_delta,
            "prediction_delta": float(prediction_delta),
            "feature_changes": feature_changes,
        }

        self.logger.info(
            "what_if_analysis_complete",
            prediction_delta=prediction_delta,
        )

        return result

    def _explain_rule_based(self, features: pd.DataFrame) -> dict[str, Any]:
        """Generate explanation for rule-based model.

        Args:
            features: Feature DataFrame (single row)

        Returns:
            Explanation dictionary mimicking SHAP output
        """
        # Get feature importance weights
        importance = self.model.get_feature_importance()

        # Calculate weighted feature contributions
        X = features[self.feature_names].values[0]
        feature_importance = np.array([importance.get(f, 0) for f in self.feature_names])

        # SHAP values as feature_value * importance_weight
        shap_values = X * feature_importance

        # Base value is average of all weighted features
        base_value = shap_values.mean()

        # Adjust SHAP values to be deviations from base
        shap_values = shap_values - base_value

        return {
            "shap_values": shap_values.tolist(),
            "base_value": float(base_value),
            "feature_values": X.tolist(),
            "feature_names": self.feature_names,
            "model_output": float(shap_values.sum() + base_value),
        }

    def _get_top_features(
        self,
        shap_values: list[float],
        feature_names: list[str],
        feature_values: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Get top K features by absolute SHAP value.

        Args:
            shap_values: SHAP values for features
            feature_names: Feature names
            feature_values: Actual feature values
            top_k: Number of top features to return

        Returns:
            List of dictionaries with top features sorted by importance
        """
        # Combine features with their SHAP values
        features = [
            {
                "feature": name,
                "shap_value": float(shap_val),
                "feature_value": float(feat_val),
                "abs_shap": abs(float(shap_val)),
            }
            for name, shap_val, feat_val in zip(
                feature_names, shap_values, feature_values, strict=True
            )
        ]

        # Sort by absolute SHAP value
        features.sort(key=lambda x: x["abs_shap"], reverse=True)

        return features[:top_k]

    def _get_cache_path(self, cache_key: str, explanation_type: str) -> Path:
        """Get cache file path for explanation.

        Args:
            cache_key: Unique key for caching
            explanation_type: Type of explanation ("prediction", "dataset")

        Returns:
            Path to cache file
        """
        # Hash the cache key for filename
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()
        filename = f"{explanation_type}_{key_hash}.json"
        return self.cache_dir / filename

    def _load_from_cache(
        self,
        cache_key: str,
        explanation_type: str,
    ) -> Optional[dict[str, Any]]:
        """Load explanation from cache if available.

        Args:
            cache_key: Cache key
            explanation_type: Type of explanation

        Returns:
            Cached explanation or None
        """
        if not self.cache_dir:
            return None

        cache_path = self._get_cache_path(cache_key, explanation_type)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(
                "cache_load_failed",
                cache_key=cache_key,
                error=str(e),
            )
            return None

    def _save_to_cache(
        self,
        cache_key: str,
        explanation_type: str,
        explanation: dict[str, Any],
    ) -> None:
        """Save explanation to cache.

        Args:
            cache_key: Cache key
            explanation_type: Type of explanation
            explanation: Explanation dictionary to cache
        """
        if not self.cache_dir:
            return

        cache_path = self._get_cache_path(cache_key, explanation_type)

        try:
            with open(cache_path, "w") as f:
                json.dump(explanation, f)
        except Exception as e:
            self.logger.warning(
                "cache_save_failed",
                cache_key=cache_key,
                error=str(e),
            )

    def clear_cache(self) -> None:
        """Clear all cached explanations."""
        if not self.cache_dir or not self.cache_dir.exists():
            self.logger.info("no_cache_to_clear")
            return

        cache_files = list(self.cache_dir.glob("*.json"))
        for cache_file in cache_files:
            cache_file.unlink()

        self.logger.info("cache_cleared", num_files=len(cache_files))
