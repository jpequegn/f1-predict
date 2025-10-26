"""Multivariate anomaly detection using Isolation Forest."""

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import structlog

logger = structlog.get_logger(__name__)

# Constants for validation
MAX_CONTAMINATION = 0.5
MIN_SAMPLES = 2


class MultivariateAnalyzer:
    """Sophisticated multivariate anomaly detection using Isolation Forest.

    This analyzer runs asynchronously post-storage (~45s) to detect complex
    multivariate patterns that univariate methods miss. Uses scikit-learn's
    IsolationForest algorithm with StandardScaler preprocessing.

    Key Features:
        - Detects multivariate outliers across multiple features simultaneously
        - Handles missing values by filling with column means
        - Normalizes anomaly scores to 0-1 range for consistency
        - Automatically scales features using StandardScaler

    Attributes:
        contamination: Expected proportion of outliers (0.0-0.5, default: 0.1)
        n_estimators: Number of isolation trees (default: 100)
        random_state: Random seed for reproducibility (default: 42)
        is_fitted: Whether model has been trained on data

    Example:
        >>> analyzer = MultivariateAnalyzer(contamination=0.1)
        >>> analyzer.fit(training_data)
        >>> results = analyzer.detect(test_data)
        >>> anomalies = results[results['anomaly_flag']]
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        """Initialize multivariate anomaly analyzer.

        Args:
            contamination: Expected proportion of outliers in dataset (0.0-0.5).
                Higher values = more aggressive detection. Default: 0.1 (10%)
            n_estimators: Number of isolation trees in the forest. More trees
                = better accuracy but slower. Default: 100
            random_state: Random seed for reproducibility. Default: 42

        Raises:
            ValueError: If contamination not in (0, 0.5] range
        """
        if not 0 < contamination <= MAX_CONTAMINATION:
            raise ValueError(
                f"contamination must be in range (0, {MAX_CONTAMINATION}]"
            )

        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        # Initialize components (created during fit)
        self._model: IsolationForest | None = None
        self._scaler: StandardScaler | None = None
        self._feature_columns: list[str] = []

        self.logger = logger.bind(
            component="multivariate_analyzer",
            contamination=contamination,
            n_estimators=n_estimators,
        )

    @property
    def is_fitted(self) -> bool:
        """Check if analyzer has been fitted to data.

        Returns:
            True if fit() has been called and model is ready
        """
        return self._model is not None and self._scaler is not None

    def fit(self, df: pd.DataFrame) -> None:
        """Train Isolation Forest model on historical data.

        Fits the model to learn normal patterns. Should be called on clean
        historical data before using detect(). Handles missing values by
        filling with column means.

        Args:
            df: Training DataFrame with numeric features

        Raises:
            ValueError: If DataFrame is empty or has < 2 samples

        Note:
            - Only numeric columns are used for training
            - Missing values are filled with column means
            - Model and scaler are stored for later use in detect()
        """
        if df.empty:
            raise ValueError("Cannot fit on empty DataFrame")

        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ValueError("DataFrame has no numeric columns to analyze")

        if len(numeric_df) < MIN_SAMPLES:
            raise ValueError(
                f"Isolation Forest requires at least {MIN_SAMPLES} samples, "
                f"got {len(numeric_df)}"
            )

        # Store feature columns for later
        self._feature_columns = numeric_df.columns.tolist()

        # Handle missing values by filling with mean
        numeric_df_filled = numeric_df.fillna(numeric_df.mean())

        # Initialize and fit scaler
        self._scaler = StandardScaler()
        scaled_data = self._scaler.fit_transform(numeric_df_filled)

        # Initialize and fit Isolation Forest
        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
        )

        self._model.fit(scaled_data)

        self.logger.info(
            "isolation_forest_fitted",
            n_samples=len(df),
            n_features=len(self._feature_columns),
            features=self._feature_columns,
        )

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect multivariate anomalies in data using trained model.

        Adds the following columns to the DataFrame:
            - anomaly_flag: Boolean indicating if row is anomalous
            - anomaly_score: Numeric score (0-1) indicating anomaly severity
            - anomaly_method: Always "isolation_forest" for this detector

        Args:
            df: Input DataFrame to analyze (must have same features as training data)

        Returns:
            DataFrame with anomaly detection columns added

        Raises:
            ValueError: If model not fitted (call fit() first)

        Note:
            - Missing values are filled with column means
            - Scores are normalized to 0-1 range (higher = more anomalous)
            - Only numeric columns used in training are analyzed
        """
        if not self.is_fitted:
            raise ValueError(
                "MultivariateAnalyzer must be fitted before calling detect(). "
                "Call fit() first."
            )

        # Type guards: ensure model and scaler are initialized
        assert self._model is not None, "Model should be fitted"
        assert self._scaler is not None, "Scaler should be fitted"

        # Create copy to avoid modifying input
        result = df.copy()

        # Initialize anomaly columns
        result["anomaly_flag"] = False
        result["anomaly_score"] = 0.0
        result["anomaly_method"] = ""

        # Get numeric columns that were used in training
        try:
            numeric_df = df[self._feature_columns]
        except KeyError as e:
            missing_cols = set(self._feature_columns) - set(df.columns)
            raise ValueError(
                f"DataFrame missing columns used in training: {missing_cols}"
            ) from e

        # Handle missing values
        numeric_df_filled = numeric_df.fillna(numeric_df.mean())

        # Scale features
        scaled_data = self._scaler.transform(numeric_df_filled)

        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = self._model.predict(scaled_data)
        result["anomaly_flag"] = predictions == -1

        # Get anomaly scores (more negative = more anomalous)
        raw_scores = self._model.score_samples(scaled_data)

        # Normalize scores to 0-1 range
        # Isolation Forest scores are negative, more negative = more anomalous
        # We normalize so that 0 = most normal, 1 = most anomalous
        normalized_scores = self._normalize_scores(raw_scores)
        result["anomaly_score"] = normalized_scores

        # Set method name for anomalies
        result.loc[result["anomaly_flag"], "anomaly_method"] = "isolation_forest"

        self.logger.debug(
            "detection_complete",
            total_rows=len(result),
            anomalies_detected=result["anomaly_flag"].sum(),
            features_analyzed=len(self._feature_columns or []),
        )

        return result

    def _normalize_scores(
        self, scores: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Normalize Isolation Forest scores to 0-1 range.

        Isolation Forest returns negative scores where more negative values
        indicate stronger anomalies. We transform these to 0-1 range where:
        - 0 = most normal (least anomalous)
        - 1 = most anomalous

        Args:
            scores: Raw Isolation Forest scores (negative values)

        Returns:
            Normalized scores in 0-1 range

        Note:
            Uses min-max normalization: (score - min) / (max - min)
            Handles edge case where all scores are identical
        """
        # Handle edge case: all scores identical
        if len(scores) == 0:
            return np.array([], dtype=np.float64)

        min_score = scores.min()
        max_score = scores.max()

        # If all scores are the same, return zeros
        if max_score == min_score:
            return np.zeros_like(scores, dtype=np.float64)

        # Normalize to 0-1 range
        # More negative scores (anomalies) will be closer to 1
        normalized = (scores - min_score) / (max_score - min_score)

        # Invert so that anomalies (low scores) get high normalized values
        return np.asarray(1.0 - normalized, dtype=np.float64)
