"""Explainability integration for monitoring system.

Provides SHAP-based explanations for drift detection, performance degradation,
and anomalies with historical tracking and alert enrichment.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FeatureImportance:
    """Individual feature importance at a point in time."""

    feature_name: str
    importance_score: float
    shap_value: float
    percentage: float
    rank: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DriftExplanation:
    """Explanation for detected drift in a feature."""

    feature_name: str
    drift_type: str  # "shift", "scale", "distribution"
    baseline_mean: float
    current_mean: float
    baseline_std: float
    current_std: float
    shap_contribution: float
    contributing_features: list[str]
    confidence: float
    recommendation: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerformanceDegradationAnalysis:
    """Analysis of performance degradation patterns."""

    timestamp: float
    metric_name: str
    baseline_value: float
    current_value: float
    degradation_percent: float
    top_contributing_features: list[FeatureImportance]
    error_patterns: dict[str, Any]
    failure_cohort_size: int
    recommended_actions: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["top_contributing_features"] = [f.to_dict() for f in self.top_contributing_features]
        return data


class ShapExplainabilityMonitor:
    """Monitor SHAP-based explanations for drift and performance issues."""

    def __init__(
        self,
        data_dir: Path | str = "data/monitoring",
        shap_explainer: Optional[Any] = None,
    ):
        """Initialize explainability monitor.

        Args:
            data_dir: Directory for storing explanations
            shap_explainer: SHAPExplainer instance for model explanations
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.shap_explainer = shap_explainer
        self.logger = logger.bind(component="explainability_monitor")

        self.explanations_file = self.data_dir / "explanations.jsonl"
        self.feature_importance_file = self.data_dir / "feature_importance_history.jsonl"
        self.degradation_analysis_file = self.data_dir / "degradation_analysis.jsonl"

        # Historical data
        self.feature_importance_history: dict[str, list[FeatureImportance]] = {}
        self.degradation_analyses: list[PerformanceDegradationAnalysis] = []

        self._load_history()

    def explain_drift(  # noqa: ARG002
        self,
        feature_name: str,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame,
        predictions: np.ndarray,  # noqa: ARG002  # For future use: can analyze prediction patterns
        shap_values: np.ndarray,
    ) -> DriftExplanation:
        """Explain detected drift using SHAP values.

        Args:
            feature_name: Name of drifted feature
            baseline_data: Historical baseline data
            current_data: Current observations
            predictions: Model predictions
            shap_values: SHAP values for predictions

        Returns:
            DriftExplanation with SHAP-based insights
        """
        try:
            # Calculate statistics
            baseline_mean = baseline_data[feature_name].mean() if feature_name in baseline_data.columns else 0.0
            baseline_std = baseline_data[feature_name].std() if feature_name in baseline_data.columns else 1.0
            current_mean = current_data[feature_name].mean() if feature_name in current_data.columns else 0.0
            current_std = current_data[feature_name].std() if feature_name in current_data.columns else 1.0

            # Determine drift type
            drift_type = self._classify_drift(baseline_mean, current_mean, baseline_std, current_std)

            # Get SHAP contribution
            if shap_values is not None and len(shap_values) > 0:
                shap_contribution = float(np.mean(np.abs(shap_values)))
            else:
                shap_contribution = 0.0

            # Find contributing features
            contributing_features = self._find_contributing_features(shap_values, feature_name)

            # Calculate confidence
            confidence = self._calculate_drift_confidence(
                baseline_mean, current_mean, baseline_std, current_std
            )

            # Generate recommendation
            recommendation = self._generate_drift_recommendation(drift_type, shap_contribution)

            explanation = DriftExplanation(
                feature_name=feature_name,
                drift_type=drift_type,
                baseline_mean=baseline_mean,
                current_mean=current_mean,
                baseline_std=baseline_std,
                current_std=current_std,
                shap_contribution=shap_contribution,
                contributing_features=contributing_features,
                confidence=confidence,
                recommendation=recommendation,
            )

            # Store explanation
            self._save_explanation(explanation)

            self.logger.info(
                "drift_explained",
                feature=feature_name,
                drift_type=drift_type,
                confidence=confidence,
            )

            return explanation

        except Exception as e:
            self.logger.error("drift_explanation_failed", feature=feature_name, error=str(e))
            raise

    def analyze_performance_degradation(
        self,
        metric_name: str,
        baseline_value: float,
        current_value: float,
        predictions: pd.DataFrame,
        shap_values: np.ndarray,
        errors: Optional[np.ndarray] = None,
    ) -> PerformanceDegradationAnalysis:
        """Analyze root cause of performance degradation using SHAP.

        Args:
            metric_name: Metric that degraded (accuracy, precision, etc.)
            baseline_value: Historical baseline metric value
            current_value: Current metric value
            predictions: DataFrame with model predictions
            shap_values: SHAP values for predictions
            errors: Array indicating prediction errors (True/False)

        Returns:
            PerformanceDegradationAnalysis with explanations
        """
        try:
            degradation_percent = ((baseline_value - current_value) / baseline_value * 100) if baseline_value != 0 else 0.0

            # Get top contributing features
            top_features = self._get_top_contributing_features(shap_values, predictions)

            # Analyze error patterns
            error_patterns = {}
            failure_cohort_size = 0

            if errors is not None:
                failure_cohort_size = int(np.sum(errors))
                error_patterns = self._analyze_error_cohort(
                    predictions[errors] if failure_cohort_size > 0 else predictions,
                    shap_values[errors] if failure_cohort_size > 0 and len(shap_values) > 0 else shap_values,
                )

            # Generate recommendations
            recommendations = self._generate_degradation_recommendations(
                metric_name, degradation_percent, top_features, error_patterns
            )

            analysis = PerformanceDegradationAnalysis(
                timestamp=time.time(),
                metric_name=metric_name,
                baseline_value=baseline_value,
                current_value=current_value,
                degradation_percent=degradation_percent,
                top_contributing_features=top_features,
                error_patterns=error_patterns,
                failure_cohort_size=failure_cohort_size,
                recommended_actions=recommendations,
            )

            # Store analysis
            self._save_degradation_analysis(analysis)
            self.degradation_analyses.append(analysis)

            self.logger.info(
                "performance_degradation_analyzed",
                metric=metric_name,
                degradation_percent=degradation_percent,
                top_features=[f.feature_name for f in top_features[:3]],
            )

            return analysis

        except Exception as e:
            self.logger.error("degradation_analysis_failed", metric=metric_name, error=str(e))
            raise

    def track_feature_importance(
        self,
        feature_importances: dict[str, float],
        model_version: str,
    ) -> None:
        """Track feature importance over time for trend analysis.

        Args:
            feature_importances: Dict of feature names to importance scores
            model_version: Version of the model
        """
        try:
            total_importance = sum(feature_importances.values())
            timestamp = time.time()

            for idx, (feature_name, importance) in enumerate(
                sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            ):
                percentage = (importance / total_importance * 100) if total_importance > 0 else 0.0

                fi = FeatureImportance(
                    feature_name=feature_name,
                    importance_score=importance,
                    shap_value=importance,  # Approximate with importance
                    percentage=percentage,
                    rank=idx + 1,
                )

                # Store in history
                if feature_name not in self.feature_importance_history:
                    self.feature_importance_history[feature_name] = []

                self.feature_importance_history[feature_name].append(fi)

                # Save to file
                with open(self.feature_importance_file, "a") as f:
                    f.write(
                        json.dumps({
                            "timestamp": timestamp,
                            "feature_name": feature_name,
                            "importance": importance,
                            "percentage": percentage,
                            "rank": idx + 1,
                            "model_version": model_version,
                        })
                        + "\n"
                    )

            self.logger.info(
                "feature_importance_tracked",
                n_features=len(feature_importances),
                model_version=model_version,
            )

        except Exception as e:
            self.logger.error("feature_importance_tracking_failed", error=str(e))

    def get_feature_importance_trend(
        self,
        feature_name: str,
        lookback_days: int = 7,  # noqa: ARG002
    ) -> list[dict]:
        """Get historical feature importance trend.

        Args:
            feature_name: Feature to analyze
            lookback_days: Number of days to look back (reserved for future filtering)

        Returns:
            List of importance values over time
        """
        if feature_name not in self.feature_importance_history:
            return []

        history = self.feature_importance_history[feature_name]
        # Note: cutoff_time reserved for future time-based filtering

        # Filter by time - feature importance items don't have timestamp, so return all
        return [
            {
                "feature": fi.feature_name,
                "importance": fi.importance_score,
                "percentage": fi.percentage,
                "rank": fi.rank,
            }
            for fi in history
        ]

    def get_degradation_analyses(
        self,
        limit: int = 10,
        metric: Optional[str] = None,
    ) -> list[PerformanceDegradationAnalysis]:
        """Get recent performance degradation analyses.

        Args:
            limit: Maximum analyses to return
            metric: Filter by metric name

        Returns:
            List of degradation analyses
        """
        analyses = self.degradation_analyses

        if metric:
            analyses = [a for a in analyses if a.metric_name == metric]

        return sorted(analyses, key=lambda x: x.timestamp, reverse=True)[:limit]

    def _classify_drift(self, baseline_mean: float, current_mean: float, baseline_std: float, current_std: float) -> str:
        """Classify type of drift detected."""
        mean_shift = abs(current_mean - baseline_mean) / (baseline_std + 1e-6)

        if mean_shift > 2.0:
            return "shift"
        if abs(current_std - baseline_std) / (baseline_std + 1e-6) > 0.5:
            return "scale"
        return "distribution"

    def _calculate_drift_confidence(
        self, baseline_mean: float, current_mean: float, baseline_std: float, current_std: float
    ) -> float:
        """Calculate confidence in drift detection."""
        mean_diff = abs(current_mean - baseline_mean)
        std_diff = abs(current_std - baseline_std)

        # Normalize to 0-1
        confidence = min(1.0, mean_diff / (baseline_std + 1e-6) * 0.3 + std_diff / (baseline_std + 1e-6) * 0.7)
        return float(confidence)

    def _find_contributing_features(self, shap_values: np.ndarray, primary_feature: str) -> list[str]:  # noqa: ARG002
        """Find features contributing to drift."""
        if shap_values is None or len(shap_values) == 0:
            return []

        # Get top features by mean absolute SHAP value
        mean_shap = np.mean(np.abs(shap_values), axis=0) if len(shap_values.shape) > 1 else np.abs(shap_values)
        top_indices = np.argsort(mean_shap)[-3:][::-1]

        return [f"feature_{i}" for i in top_indices if i != 0]

    def _generate_drift_recommendation(self, drift_type: str, shap_contribution: float) -> str:  # noqa: ARG002
        """Generate recommendation for drift."""
        if drift_type == "shift":
            return "Mean shift detected. Consider retraining with recent data."
        if drift_type == "scale":
            return "Variance change detected. Verify feature scaling consistency."
        return "Distribution drift detected. Investigate data collection changes."

    def _get_top_contributing_features(
        self, shap_values: np.ndarray, predictions: pd.DataFrame
    ) -> list[FeatureImportance]:
        """Get top contributing features to performance degradation."""
        if shap_values is None or len(shap_values) == 0:
            return []

        # Calculate mean absolute SHAP values
        mean_shap = np.mean(np.abs(shap_values), axis=0) if len(shap_values.shape) > 1 else np.abs(shap_values)

        feature_importances = []
        n_features = min(len(mean_shap), len(predictions.columns))

        for idx in np.argsort(mean_shap)[-n_features:][::-1]:
            if idx < len(predictions.columns):
                feature_name = predictions.columns[idx]
                importance = float(mean_shap[idx])
                percentage = (importance / np.sum(mean_shap) * 100) if np.sum(mean_shap) > 0 else 0.0

                feature_importances.append(
                    FeatureImportance(
                        feature_name=feature_name,
                        importance_score=importance,
                        shap_value=importance,
                        percentage=percentage,
                        rank=len(feature_importances) + 1,
                    )
                )

        return feature_importances[:5]

    def _analyze_error_cohort(self, error_predictions: pd.DataFrame, error_shap: np.ndarray) -> dict:  # noqa: ARG002
        """Analyze patterns in predictions where model failed."""
        patterns = {
            "n_errors": len(error_predictions),
            "error_rate": len(error_predictions) / max(1, len(error_predictions)) * 100,
        }

        if len(error_predictions) > 0:
            patterns["mean_prediction"] = float(error_predictions.mean().mean())
            patterns["std_prediction"] = float(error_predictions.std().mean())

        return patterns

    def _generate_degradation_recommendations(
        self,
        metric_name: str,
        degradation_percent: float,
        top_features: list[FeatureImportance],
        error_patterns: dict,
    ) -> list[str]:
        """Generate recommendations for addressing performance degradation."""
        recommendations = []

        if degradation_percent > 10:
            recommendations.append(f"Critical: {metric_name} degraded {degradation_percent:.1f}%. Investigate immediately.")

        if top_features:
            top_feature_names = [f.feature_name for f in top_features[:3]]
            recommendations.append(
                f"Focus on features: {', '.join(top_feature_names)}. These are driving performance issues."
            )

        if error_patterns.get("n_errors", 0) > 10:
            recommendations.append("High error rate detected. Consider retraining or data quality checks.")

        if not recommendations:
            recommendations.append("Monitor closely. Degradation is within normal range.")

        return recommendations

    def _save_explanation(self, explanation: DriftExplanation) -> None:
        """Save drift explanation to file."""
        try:
            with open(self.explanations_file, "a") as f:
                f.write(json.dumps(explanation.to_dict()) + "\n")
        except Exception as e:
            self.logger.error("save_explanation_failed", error=str(e))

    def _save_degradation_analysis(self, analysis: PerformanceDegradationAnalysis) -> None:
        """Save degradation analysis to file."""
        try:
            with open(self.degradation_analysis_file, "a") as f:
                f.write(json.dumps(analysis.to_dict()) + "\n")
        except Exception as e:
            self.logger.error("save_degradation_analysis_failed", error=str(e))

    def _load_history(self) -> None:  # noqa: C901
        """Load historical data from files.

        Complexity is acceptable for robust data loading with error handling
        for multiple file types and data reconstruction.
        """
        # Load feature importance history
        if self.feature_importance_file.exists():
            try:
                with open(self.feature_importance_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            feature_name = data.get("feature_name", "unknown")
                            if feature_name not in self.feature_importance_history:
                                self.feature_importance_history[feature_name] = []
                            # Store as simple dict for now
                            self.feature_importance_history[feature_name].append(data)
            except Exception as e:
                self.logger.warning("feature_importance_load_failed", error=str(e))

        # Load degradation analyses
        if self.degradation_analysis_file.exists():
            try:
                with open(self.degradation_analysis_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            # Reconstruct PerformanceDegradationAnalysis
                            try:
                                features = [
                                    FeatureImportance(**f) for f in data.get("top_contributing_features", [])
                                ]
                                analysis = PerformanceDegradationAnalysis(
                                    timestamp=data["timestamp"],
                                    metric_name=data["metric_name"],
                                    baseline_value=data["baseline_value"],
                                    current_value=data["current_value"],
                                    degradation_percent=data["degradation_percent"],
                                    top_contributing_features=features,
                                    error_patterns=data.get("error_patterns", {}),
                                    failure_cohort_size=data.get("failure_cohort_size", 0),
                                    recommended_actions=data.get("recommended_actions", []),
                                )
                                self.degradation_analyses.append(analysis)
                            except Exception as e:  # noqa: S112
                                self.logger.debug("degradation_reconstruction_failed", error=str(e))
                                continue
            except Exception as e:
                self.logger.warning("degradation_analysis_load_failed", error=str(e))
