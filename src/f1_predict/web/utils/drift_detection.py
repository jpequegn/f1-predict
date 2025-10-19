"""Data drift detection utilities.

Provides:
- Kolmogorov-Smirnov test for distribution shifts
- Population Stability Index (PSI) calculation
- Feature drift analysis and reporting
- Concept drift detection
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection test."""

    feature_name: str
    drift_detected: bool
    test_statistic: float
    p_value: float
    threshold: float
    drift_type: str  # "ks_test" or "psi"
    baseline_stats: dict[str, Any]
    current_stats: dict[str, Any]
    severity: str  # "low", "medium", "high"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "drift_detected": self.drift_detected,
            "test_statistic": float(self.test_statistic),
            "p_value": float(self.p_value),
            "threshold": float(self.threshold),
            "drift_type": self.drift_type,
            "baseline_stats": self.baseline_stats,
            "current_stats": self.current_stats,
            "severity": self.severity,
        }


class DriftDetector:
    """Detects data drift in input features."""

    def __init__(
        self,
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.2,
        min_samples: int = 30,
    ):
        """Initialize drift detector.

        Args:
            ks_threshold: P-value threshold for KS test (lower = more sensitive)
            psi_threshold: PSI threshold for drift (0.2 = 20% drift)
            min_samples: Minimum samples for drift testing
        """
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.min_samples = min_samples
        self.logger = logger.bind(component="drift_detector")

    def detect_feature_drift_ks_test(
        self,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_name: str,
    ) -> Optional[DriftResult]:
        """Detect feature drift using Kolmogorov-Smirnov test.

        Args:
            baseline_data: Baseline/reference data
            current_data: Current data to test
            feature_name: Name of feature to test

        Returns:
            DriftResult or None if not enough data
        """
        try:
            # Check data availability
            if (
                len(baseline_data) < self.min_samples
                or len(current_data) < self.min_samples
            ):
                self.logger.warning(
                    "insufficient_data_for_ks_test",
                    feature=feature_name,
                    baseline_size=len(baseline_data),
                    current_size=len(current_data),
                )
                return None

            if feature_name not in baseline_data.columns:
                self.logger.warning("feature_not_found", feature=feature_name)
                return None

            # Get feature values
            baseline_vals = baseline_data[feature_name].dropna().values
            current_vals = current_data[feature_name].dropna().values

            if len(baseline_vals) == 0 or len(current_vals) == 0:
                return None

            # Perform KS test
            statistic, p_value = ks_2samp(baseline_vals, current_vals)

            # Determine severity
            if p_value < 0.01:
                severity = "high"
            elif p_value < 0.05:
                severity = "medium"
            else:
                severity = "low"

            drift_detected = p_value < self.ks_threshold

            # Calculate statistics
            baseline_stats = {
                "mean": float(baseline_vals.mean()),
                "std": float(baseline_vals.std()),
                "min": float(baseline_vals.min()),
                "max": float(baseline_vals.max()),
                "median": float(np.median(baseline_vals)),
                "skew": float(pd.Series(baseline_vals).skew()),
            }

            current_stats = {
                "mean": float(current_vals.mean()),
                "std": float(current_vals.std()),
                "min": float(current_vals.min()),
                "max": float(current_vals.max()),
                "median": float(np.median(current_vals)),
                "skew": float(pd.Series(current_vals).skew()),
            }

            result = DriftResult(
                feature_name=feature_name,
                drift_detected=drift_detected,
                test_statistic=statistic,
                p_value=p_value,
                threshold=self.ks_threshold,
                drift_type="ks_test",
                baseline_stats=baseline_stats,
                current_stats=current_stats,
                severity=severity,
            )

            if drift_detected:
                self.logger.warning(
                    "feature_drift_detected",
                    feature=feature_name,
                    p_value=p_value,
                    severity=severity,
                )

            return result
        except Exception as e:
            self.logger.error(
                "error_in_ks_test",
                feature=feature_name,
                error=str(e),
            )
            return None

    def calculate_psi(
        self,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_name: str,
        n_bins: int = 10,
    ) -> Optional[DriftResult]:
        """Calculate Population Stability Index for feature.

        PSI measures how much a variable's distribution has shifted.
        PSI < 0.1 = minimal shift
        PSI 0.1-0.25 = small shift
        PSI > 0.25 = significant shift

        Args:
            baseline_data: Baseline/reference data
            current_data: Current data to test
            feature_name: Name of feature
            n_bins: Number of bins for binning

        Returns:
            DriftResult with PSI value
        """
        try:
            # Check data availability
            if (
                len(baseline_data) < self.min_samples
                or len(current_data) < self.min_samples
            ):
                self.logger.warning(
                    "insufficient_data_for_psi",
                    feature=feature_name,
                    baseline_size=len(baseline_data),
                    current_size=len(current_data),
                )
                return None

            if feature_name not in baseline_data.columns:
                self.logger.warning("feature_not_found", feature=feature_name)
                return None

            # Get feature values
            baseline_vals = baseline_data[feature_name].dropna().values
            current_vals = current_data[feature_name].dropna().values

            if len(baseline_vals) == 0 or len(current_vals) == 0:
                return None

            # Create bins based on baseline data
            min_val = baseline_vals.min()
            max_val = baseline_vals.max()
            bins = np.linspace(min_val, max_val, n_bins + 1)

            # Bin both distributions
            baseline_binned = np.histogram(baseline_vals, bins=bins)[0]
            current_binned = np.histogram(current_vals, bins=bins)[0]

            # Normalize to percentages
            baseline_pct = baseline_binned / baseline_binned.sum()
            current_pct = current_binned / current_binned.sum()

            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            baseline_pct = baseline_pct + epsilon
            current_pct = current_pct + epsilon

            # Calculate PSI
            psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

            # Determine severity
            if psi > 0.25:
                severity = "high"
            elif psi > 0.15:
                severity = "medium"
            else:
                severity = "low"

            drift_detected = psi > self.psi_threshold

            # Statistics
            baseline_stats = {
                "mean": float(baseline_vals.mean()),
                "std": float(baseline_vals.std()),
                "percentile_25": float(np.percentile(baseline_vals, 25)),
                "percentile_50": float(np.percentile(baseline_vals, 50)),
                "percentile_75": float(np.percentile(baseline_vals, 75)),
            }

            current_stats = {
                "mean": float(current_vals.mean()),
                "std": float(current_vals.std()),
                "percentile_25": float(np.percentile(current_vals, 25)),
                "percentile_50": float(np.percentile(current_vals, 50)),
                "percentile_75": float(np.percentile(current_vals, 75)),
            }

            result = DriftResult(
                feature_name=feature_name,
                drift_detected=drift_detected,
                test_statistic=float(psi),
                p_value=0.0,  # PSI doesn't have p-value
                threshold=self.psi_threshold,
                drift_type="psi",
                baseline_stats=baseline_stats,
                current_stats=current_stats,
                severity=severity,
            )

            if drift_detected:
                self.logger.warning(
                    "feature_psi_drift_detected",
                    feature=feature_name,
                    psi=psi,
                    severity=severity,
                )

            return result
        except Exception as e:
            self.logger.error(
                "error_calculating_psi",
                feature=feature_name,
                error=str(e),
            )
            return None

    def detect_dataset_drift(
        self,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame,
        method: str = "psi",
    ) -> dict[str, Any]:
        """Detect drift across all features in dataset.

        Args:
            baseline_data: Baseline/reference data
            current_data: Current data to test
            method: Detection method ("ks_test" or "psi")

        Returns:
            Dictionary with drift detection results
        """
        self.logger.info(
            "detecting_dataset_drift",
            method=method,
            num_features=len(baseline_data.columns),
        )

        results = []
        drifted_features = []

        for feature in baseline_data.columns:
            if feature not in current_data.columns:
                continue

            # Select appropriate method
            if method == "ks_test":
                result = self.detect_feature_drift_ks_test(
                    baseline_data, current_data, feature
                )
            else:  # psi
                result = self.calculate_psi(baseline_data, current_data, feature)

            if result:
                results.append(result.to_dict())
                if result.drift_detected:
                    drifted_features.append(feature)

        # Calculate overall drift metrics
        total_features = len(results)
        drifted_count = len(drifted_features)
        drift_ratio = drifted_count / total_features if total_features > 0 else 0

        dataset_drift = {
            "total_features_checked": total_features,
            "features_with_drift": drifted_count,
            "drift_ratio": drift_ratio,
            "overall_drift_detected": drift_ratio > 0.1,  # >10% features drifted
            "drifted_features": drifted_features,
            "feature_results": results,
        }

        if dataset_drift["overall_drift_detected"]:
            self.logger.warning(
                "overall_dataset_drift_detected",
                drift_ratio=drift_ratio,
                num_drifted=drifted_count,
            )

        return dataset_drift

    def compare_distributions(
        self, data1: np.ndarray, data2: np.ndarray
    ) -> dict[str, float]:
        """Compare two distributions.

        Args:
            data1: First data sample
            data2: Second data sample

        Returns:
            Dictionary with comparison metrics
        """
        try:
            # KS test
            statistic, p_value = ks_2samp(data1, data2)

            # Mean difference
            mean_diff = np.abs(np.mean(data1) - np.mean(data2))

            # Variance ratio
            var_ratio = np.var(data1) / (np.var(data2) + 1e-10)

            return {
                "ks_statistic": float(statistic),
                "ks_p_value": float(p_value),
                "mean_diff": float(mean_diff),
                "variance_ratio": float(var_ratio),
                "distributions_different": p_value < 0.05,
            }
        except Exception as e:
            self.logger.error("error_comparing_distributions", error=str(e))
            return {}
