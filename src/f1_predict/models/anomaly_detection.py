"""Advanced anomaly detection for model monitoring.

Provides multiple anomaly detection methods:
- Isolation Forest for multivariate outliers
- Z-score and IQR for univariate detection
- ARIMA forecast-based detection
- Autoencoder for pattern-based detection
- Adaptive thresholds based on historical data
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import structlog

# Type aliases for cleaner type hints
NDArray = np.ndarray[Any, np.dtype[np.floating[Any]]]

logger = structlog.get_logger(__name__)


class AnomalyType(Enum):
    """Types of anomalies detected."""

    UNIVARIATE = "univariate"  # Single variable anomaly (Z-score, IQR)
    MULTIVARIATE = "multivariate"  # Multiple variable anomaly (Isolation Forest)
    TIMESERIES = "timeseries"  # Time series forecast anomaly (ARIMA)
    PATTERN = "pattern"  # Pattern-based anomaly (Autoencoder)


@dataclass
class AnomalyScore:
    """Result of anomaly detection analysis."""

    timestamp: float
    is_anomaly: bool
    anomaly_type: str
    score: float  # 0.0-1.0, higher = more anomalous
    threshold: float
    confidence: float  # 0.0-1.0 confidence in detection
    features_involved: list[str] = field(default_factory=list)
    explanation: str = ""
    severity: str = "info"  # "info", "warning", "critical"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type,
            "score": round(self.score, 4),
            "threshold": round(self.threshold, 4),
            "confidence": round(self.confidence, 4),
            "features_involved": self.features_involved,
            "explanation": self.explanation,
            "severity": self.severity,
        }


@dataclass
class HistoricalBaseline:
    """Baseline statistics for adaptive thresholding."""

    feature_name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_95: float
    percentile_99: float
    n_samples: int
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "min": round(self.min_val, 6),
            "max": round(self.max_val, 6),
            "p25": round(self.percentile_25, 6),
            "p50": round(self.percentile_50, 6),
            "p75": round(self.percentile_75, 6),
            "p95": round(self.percentile_95, 6),
            "p99": round(self.percentile_99, 6),
            "n_samples": self.n_samples,
            "created_at": self.created_at,
        }


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors."""

    @abstractmethod
    def fit(self, data: pd.DataFrame | NDArray) -> None:
        """Fit the detector to historical data.

        Args:
            data: Historical data for baseline establishment
        """
        pass

    @abstractmethod
    def detect(self, data: pd.DataFrame | NDArray, timestamp: Optional[float] = None) -> AnomalyScore:
        """Detect anomalies in new data.

        Args:
            data: Data to check for anomalies
            timestamp: Timestamp of the observation

        Returns:
            AnomalyScore with detection results
        """
        pass

    @abstractmethod
    def get_threshold(self) -> float:
        """Get the current anomaly threshold."""
        pass


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest-based anomaly detection.

    Effective for multivariate outlier detection. Isolates anomalies by
    randomly selecting features and split values, building isolation trees.
    Anomalies require fewer splits to isolate.
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """Initialize Isolation Forest detector.

        Args:
            contamination: Expected proportion of anomalies (0.0-0.5)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.threshold: float = 0.5
        self.feature_names: list[str] = []
        self.logger = logger.bind(component="isolation_forest_detector")

    def fit(self, data: pd.DataFrame | NDArray) -> None:
        """Fit Isolation Forest to data.

        Args:
            data: Historical data (samples x features)
        """
        try:
            if isinstance(data, pd.DataFrame):
                self.feature_names = data.columns.tolist()
                data_array: NDArray = data.values
            else:
                data_array = np.asarray(data)
                # Handle 1D arrays
                if data_array.ndim == 1:
                    data_array = data_array.reshape(-1, 1)
                self.feature_names = [f"feature_{i}" for i in range(data_array.shape[1])]

            # Standardize data
            self.scaler = StandardScaler()
            data_scaled = self.scaler.fit_transform(data_array)

            # Fit Isolation Forest
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=100,
            )
            self.model.fit(data_scaled)

            # Calculate threshold based on scores
            scores = self.model.score_samples(data_scaled)
            # Normalize scores to 0-1 range (lower scores = more anomalous)
            self.threshold = np.percentile(scores, self.contamination * 100)

            self.logger.info(
                "isolation_forest_fitted",
                n_samples=len(data_array),
                n_features=data_array.shape[1],
                threshold=self.threshold,
            )

        except Exception as e:
            self.logger.error("isolation_forest_fit_failed", error=str(e))
            raise

    def detect(
        self, data: pd.DataFrame | NDArray, timestamp: Optional[float] = None
    ) -> AnomalyScore:
        """Detect anomalies using Isolation Forest.

        Args:
            data: Data to check (samples x features or single sample)
            timestamp: Timestamp of observation

        Returns:
            AnomalyScore with detection results
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        import time

        if timestamp is None:
            timestamp = time.time()

        try:
            # Handle single sample vs batch
            if isinstance(data, pd.DataFrame):
                data_array = data.values
            else:
                data_array = np.asarray(data)

            # Ensure 2D
            if data_array.ndim == 1:
                data_array = data_array.reshape(1, -1)

            # Standardize
            data_scaled = self.scaler.transform(data_array)

            # Get anomaly score (last sample)
            score = self.model.score_samples(data_scaled)[-1]
            prediction = self.model.predict(data_scaled)[-1]

            # Normalize score to 0-1 range
            normalized_score = 1.0 / (1.0 + np.exp(-score))

            is_anomaly = prediction == -1
            confidence = abs(normalized_score - 0.5) * 2  # 0-1 confidence

            # Determine severity
            if normalized_score > 0.8:
                severity = "critical"
            elif normalized_score > 0.6:
                severity = "warning"
            else:
                severity = "info"

            return AnomalyScore(
                timestamp=timestamp,
                is_anomaly=is_anomaly,
                anomaly_type=AnomalyType.MULTIVARIATE.value,
                score=normalized_score,
                threshold=1.0 / (1.0 + np.exp(-self.threshold)),
                confidence=confidence,
                features_involved=self.feature_names,
                explanation=f"Isolation Forest anomaly score: {normalized_score:.4f}",
                severity=severity,
            )

        except Exception as e:
            self.logger.error("isolation_forest_detection_failed", error=str(e))
            raise

    def get_threshold(self) -> float:
        """Get anomaly threshold."""
        return self.threshold


class ZScoreDetector(AnomalyDetector):
    """Z-score based univariate anomaly detection.

    Detects values that are statistically far from the mean.
    Points beyond ±3 sigma are typically considered anomalies.
    """

    def __init__(self, threshold_sigma: float = 3.0):
        """Initialize Z-score detector.

        Args:
            threshold_sigma: Number of standard deviations for threshold
        """
        self.threshold_sigma = threshold_sigma
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self.threshold: float = 0.0
        self.feature_name: str = ""
        self.logger = logger.bind(component="zscore_detector")
        self._fitted = False

    def fit(self, data: pd.DataFrame | NDArray) -> None:
        """Calculate mean and std from data.

        Args:
            data: Historical data values
        """
        try:
            if isinstance(data, pd.DataFrame):
                if len(data.columns) > 1:
                    self.logger.warning("multiple_features_provided, using first")
                self.feature_name = str(data.columns[0])
                values: NDArray = data.iloc[:, 0].values
            else:
                values = np.asarray(data).flatten()
                self.feature_name = "value"

            mean_val = float(np.mean(values))
            std_val = float(np.std(values))

            if std_val == 0:
                std_val = 1.0  # Avoid division by zero

            self.mean = mean_val
            self.std = std_val
            self.threshold = self.threshold_sigma * std_val
            self._fitted = True

            self.logger.info(
                "zscore_detector_fitted",
                mean=self.mean,
                std=self.std,
                threshold_sigma=self.threshold_sigma,
            )

        except Exception as e:
            self.logger.error("zscore_fit_failed", error=str(e))
            raise

    def detect(
        self, data: pd.DataFrame | NDArray, timestamp: Optional[float] = None
    ) -> AnomalyScore:
        """Detect anomalies using Z-score.

        Args:
            data: Value(s) to check
            timestamp: Timestamp of observation

        Returns:
            AnomalyScore with detection results
        """
        if self.mean is None or self.std is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        import time

        if timestamp is None:
            timestamp = time.time()

        try:
            # Extract value
            if isinstance(data, pd.DataFrame):
                value = data.iloc[-1, 0] if len(data) > 0 else data.iloc[0, 0]
            elif isinstance(data, (list, np.ndarray)):
                value = float(np.asarray(data).flat[-1])
            else:
                value = float(data)

            # Calculate Z-score
            z_score = abs((value - self.mean) / self.std)

            # Normalize to 0-1 range
            normalized_score = min(1.0, z_score / self.threshold_sigma)

            is_anomaly = z_score > self.threshold_sigma
            confidence = min(1.0, z_score / self.threshold_sigma)

            # Determine severity
            if z_score > 5:
                severity = "critical"
            elif z_score > 3:
                severity = "warning"
            else:
                severity = "info"

            return AnomalyScore(
                timestamp=timestamp,
                is_anomaly=is_anomaly,
                anomaly_type=AnomalyType.UNIVARIATE.value,
                score=normalized_score,
                threshold=1.0,
                confidence=confidence,
                features_involved=[self.feature_name],
                explanation=f"Z-score: {z_score:.2f}σ (threshold: {self.threshold_sigma}σ)",
                severity=severity,
            )

        except Exception as e:
            self.logger.error("zscore_detection_failed", error=str(e))
            raise

    def get_threshold(self) -> float:
        """Get anomaly threshold."""
        return self.threshold


class IQRDetector(AnomalyDetector):
    """Interquartile Range (IQR) based anomaly detection.

    Detects values outside 1.5 * IQR bounds. More robust to outliers
    than Z-score for skewed distributions.
    """

    def __init__(self, iqr_multiplier: float = 1.5):
        """Initialize IQR detector.

        Args:
            iqr_multiplier: IQR multiplier for bounds (typically 1.5)
        """
        self.iqr_multiplier = iqr_multiplier
        self.q1: Optional[float] = None
        self.q3: Optional[float] = None
        self.iqr: Optional[float] = None
        self.lower_bound: float = 0.0
        self.upper_bound: float = 0.0
        self.feature_name: str = ""
        self.logger = logger.bind(component="iqr_detector")
        self._fitted = False

    def fit(self, data: pd.DataFrame | NDArray) -> None:
        """Calculate quartiles from data.

        Args:
            data: Historical data values
        """
        try:
            if isinstance(data, pd.DataFrame):
                if len(data.columns) > 1:
                    self.logger.warning("multiple_features_provided, using first")
                self.feature_name = str(data.columns[0])
                values: NDArray = data.iloc[:, 0].values
            else:
                values = np.asarray(data).flatten()
                self.feature_name = "value"

            q1_val = float(np.percentile(values, 25))
            q3_val = float(np.percentile(values, 75))
            iqr_val = q3_val - q1_val

            self.q1 = q1_val
            self.q3 = q3_val
            self.iqr = iqr_val
            self.lower_bound = q1_val - self.iqr_multiplier * iqr_val
            self.upper_bound = q3_val + self.iqr_multiplier * iqr_val
            self._fitted = True

            self.logger.info(
                "iqr_detector_fitted",
                q1=self.q1,
                q3=self.q3,
                iqr=self.iqr,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
            )

        except Exception as e:
            self.logger.error("iqr_fit_failed", error=str(e))
            raise

    def detect(  # noqa: PLR0912
        self, data: pd.DataFrame | NDArray, timestamp: Optional[float] = None
    ) -> AnomalyScore:
        """Detect anomalies using IQR method.

        Args:
            data: Value(s) to check
            timestamp: Timestamp of observation

        Returns:
            AnomalyScore with detection results
        """
        if self.q1 is None or self.q3 is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        import time

        if timestamp is None:
            timestamp = time.time()

        try:
            # Extract value
            if isinstance(data, pd.DataFrame):
                value = data.iloc[-1, 0] if len(data) > 0 else data.iloc[0, 0]
            elif isinstance(data, (list, np.ndarray)):
                value = float(np.asarray(data).flat[-1])
            else:
                value = float(data)

            is_anomaly = value < self.lower_bound or value > self.upper_bound

            # Calculate normalized anomaly score
            if value < self.lower_bound:
                distance = self.lower_bound - value
                normalized_score = min(1.0, distance / abs(self.lower_bound) if self.lower_bound != 0 else 1.0)
            elif value > self.upper_bound:
                distance = value - self.upper_bound
                normalized_score = min(1.0, distance / abs(self.upper_bound) if self.upper_bound != 0 else 1.0)
            else:
                normalized_score = 0.0

            confidence = normalized_score

            # Determine severity
            severity = "critical" if (is_anomaly and normalized_score > 0.7) else (
                "warning" if is_anomaly else "info"
            )

            return AnomalyScore(
                timestamp=timestamp,
                is_anomaly=is_anomaly,
                anomaly_type=AnomalyType.UNIVARIATE.value,
                score=normalized_score,
                threshold=1.0,
                confidence=confidence,
                features_involved=[self.feature_name],
                explanation=f"IQR bounds: [{self.lower_bound:.4f}, {self.upper_bound:.4f}], value: {value:.4f}",
                severity=severity,
            )

        except Exception as e:
            self.logger.error("iqr_detection_failed", error=str(e))
            raise

    def get_threshold(self) -> float:
        """Get anomaly threshold (upper bound)."""
        return self.upper_bound


class ARIMAForecastDetector(AnomalyDetector):
    """ARIMA forecast-based anomaly detection.

    Compares actual values to ARIMA forecasts and flags large deviations.
    Effective for time series with trend and seasonality.
    """

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1), deviation_threshold: float = 2.0):
        """Initialize ARIMA detector.

        Args:
            order: ARIMA order (p, d, q)
            deviation_threshold: Number of forecast std devs for anomaly
        """
        self.order = order
        self.deviation_threshold = deviation_threshold
        self.model: Optional[Any] = None  # ARIMA model
        self.residuals: Optional[NDArray] = None
        self.residual_std: float = 1.0
        self.feature_name: str = ""
        self.logger = logger.bind(component="arima_forecast_detector")

    def fit(self, data: pd.DataFrame | NDArray) -> None:
        """Fit ARIMA model to data.

        Args:
            data: Historical time series data
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA

            if isinstance(data, pd.DataFrame):
                if len(data.columns) > 1:
                    self.logger.warning("multiple_features_provided, using first")
                self.feature_name = str(data.columns[0])
                values: NDArray = data.iloc[:, 0].values
            else:
                values = np.asarray(data).flatten()
                self.feature_name = "value"

            # Fit ARIMA model
            self.model = ARIMA(values, order=self.order)
            results = self.model.fit()

            # Store residuals and their std
            residuals_data = results.resid
            self.residuals = residuals_data.values if hasattr(residuals_data, 'values') else residuals_data
            self.residual_std = float(np.std(self.residuals))

            if self.residual_std == 0:
                self.residual_std = 1.0

            self.logger.info(
                "arima_detector_fitted",
                order=self.order,
                residual_std=self.residual_std,
            )

        except ImportError as e:
            self.logger.error("statsmodels_required", feature="ARIMA detection")
            raise ValueError("statsmodels required for ARIMA detection") from e
        except Exception as e:
            self.logger.error("arima_fit_failed", error=str(e))
            raise

    def detect(
        self, data: pd.DataFrame | NDArray, timestamp: Optional[float] = None
    ) -> AnomalyScore:
        """Detect anomalies using ARIMA forecast deviation.

        Args:
            data: Time series data with new observation
            timestamp: Timestamp of observation

        Returns:
            AnomalyScore with detection results
        """
        if self.model is None or self.residual_std is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        import time

        if timestamp is None:
            timestamp = time.time()

        try:
            from statsmodels.tsa.arima.model import ARIMA

            if isinstance(data, pd.DataFrame):
                values = data.iloc[:, 0].values if len(data.columns) > 0 else data.values[:, 0]
            else:
                values = np.asarray(data).flatten()

            actual_value = values[-1]

            # Refit model with all data
            model = ARIMA(values, order=self.order)
            results = model.fit()

            # Get one-step ahead forecast
            forecast = results.get_forecast(steps=1)
            predicted_mean = forecast.predicted_mean
            predicted_value = predicted_mean.values[0] if hasattr(predicted_mean, 'values') else predicted_mean[0]

            # Calculate deviation from forecast
            deviation = abs(actual_value - predicted_value)
            confidence_interval_width = 2 * self.residual_std * self.deviation_threshold

            # Normalize to 0-1
            normalized_score = min(1.0, deviation / confidence_interval_width if confidence_interval_width > 0 else 1.0)

            is_anomaly = deviation > confidence_interval_width

            # Determine severity
            if normalized_score > 0.9:
                severity = "critical"
            elif normalized_score > 0.6:
                severity = "warning"
            else:
                severity = "info"

            return AnomalyScore(
                timestamp=timestamp,
                is_anomaly=is_anomaly,
                anomaly_type=AnomalyType.TIMESERIES.value,
                score=normalized_score,
                threshold=1.0,
                confidence=normalized_score,
                features_involved=[self.feature_name],
                explanation=(
                    f"ARIMA forecast: {predicted_value:.4f}, actual: {actual_value:.4f}, "
                    f"deviation: {deviation:.4f} (threshold: {confidence_interval_width:.4f})"
                ),
                severity=severity,
            )

        except Exception as e:
            self.logger.error("arima_detection_failed", error=str(e))
            raise

    def get_threshold(self) -> float:
        """Get anomaly threshold."""
        return self.deviation_threshold * self.residual_std
