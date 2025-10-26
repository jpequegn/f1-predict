"""Univariate anomaly detection using Z-score and IQR methods."""


import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class UnivariateDetector:
    """Fast univariate anomaly detection using Z-score and IQR methods.

    This detector runs synchronously during data collection (~60ms) to provide
    immediate feedback on potential data quality issues.

    Methods:
        - Z-score: Flags values >3σ from mean (configurable threshold)
        - IQR: Flags values outside [Q1-1.5×IQR, Q3+1.5×IQR]

    Attributes:
        z_score_threshold: Number of standard deviations for Z-score
            method (default: 3.0)
        use_iqr: Whether to use IQR method in addition to Z-score
            (default: True)
        enabled: Whether detector is active (default: True)
    """

    def __init__(
        self,
        z_score_threshold: float = 3.0,
        use_iqr: bool = True,
        enabled: bool = True,
    ):
        """Initialize univariate anomaly detector.

        Args:
            z_score_threshold: Z-score threshold for anomaly detection (default: 3.0)
            use_iqr: Whether to use IQR method (default: True)
            enabled: Whether detector is active (default: True)
        """
        self.z_score_threshold = z_score_threshold
        self.use_iqr = use_iqr
        self.enabled = enabled
        self.logger = logger.bind(
            component="univariate_detector",
            z_threshold=z_score_threshold,
            use_iqr=use_iqr,
        )

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in DataFrame using Z-score and/or IQR methods.

        Adds the following columns to the DataFrame:
            - anomaly_flag: Boolean indicating if row is anomalous
            - anomaly_score: Numeric score (0-1) indicating anomaly severity
            - anomaly_method: Comma-separated list of methods that flagged this row
            - anomaly_confidence: Confidence score (0-1) for the anomaly detection

        Args:
            df: Input DataFrame with numeric columns to analyze

        Returns:
            DataFrame with anomaly detection columns added

        Note:
            - Returns input DataFrame unchanged if detector is disabled
            - Handles missing values by ignoring them in calculations
            - Skips non-numeric columns automatically
            - Returns empty DataFrame if input is empty
        """
        if not self.enabled:
            return df

        if df.empty:
            return df

        # Create a copy to avoid modifying input
        result = df.copy()

        # Initialize anomaly columns
        result["anomaly_flag"] = False
        result["anomaly_score"] = 0.0
        result["anomaly_method"] = ""
        result["anomaly_confidence"] = 0.0

        # Get numeric columns only
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

        # Remove anomaly columns from analysis
        numeric_cols = [
            col
            for col in numeric_cols
            if col not in ["anomaly_flag", "anomaly_score", "anomaly_confidence"]
        ]

        if not numeric_cols:
            self.logger.debug("no_numeric_columns_to_analyze")
            return result

        # Track anomalies per row across all columns
        anomaly_flags = pd.DataFrame(False, index=result.index, columns=numeric_cols)
        anomaly_scores = pd.DataFrame(0.0, index=result.index, columns=numeric_cols)
        anomaly_methods = pd.DataFrame("", index=result.index, columns=numeric_cols)

        # Minimum samples required for statistical analysis
        min_samples = 2

        for col in numeric_cols:
            col_data = result[col]

            # Skip if insufficient data
            if len(col_data.dropna()) < min_samples:
                continue

            # Z-score detection
            z_flags, z_scores = self._detect_z_score(col_data)
            anomaly_flags[col] |= z_flags
            anomaly_scores[col] = np.maximum(anomaly_scores[col], z_scores)

            # Add method name where flagged
            z_method_mask = z_flags & (anomaly_methods[col] == "")
            anomaly_methods.loc[z_method_mask, col] = "z_score"

            # IQR detection
            if self.use_iqr:
                iqr_flags, iqr_scores = self._detect_iqr(col_data)
                anomaly_flags[col] |= iqr_flags
                anomaly_scores[col] = np.maximum(anomaly_scores[col], iqr_scores)

                # Add IQR to method name
                iqr_only_mask = iqr_flags & ~z_flags
                anomaly_methods.loc[iqr_only_mask, col] = "iqr"

                both_mask = iqr_flags & z_flags
                anomaly_methods.loc[both_mask, col] = "z_score,iqr"

        # Aggregate across columns: a row is anomalous if ANY column is anomalous
        result["anomaly_flag"] = anomaly_flags.any(axis=1)

        # Max anomaly score across all columns
        result["anomaly_score"] = anomaly_scores.max(axis=1)

        # Combine methods from all flagged columns
        result["anomaly_method"] = anomaly_methods.apply(
            lambda row: ",".join(sorted(set(filter(None, row)))), axis=1
        )

        # Confidence based on anomaly score (normalize to 0-1)
        result["anomaly_confidence"] = self._calculate_confidence(
            result["anomaly_score"]
        )

        self.logger.debug(
            "detection_complete",
            total_rows=len(result),
            anomalies_detected=result["anomaly_flag"].sum(),
            columns_analyzed=len(numeric_cols),
        )

        return result

    def _detect_z_score(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Detect anomalies using Z-score method.

        Args:
            series: Numeric series to analyze

        Returns:
            Tuple of (flags, scores) where:
                - flags: Boolean series indicating anomalies
                - scores: Numeric series with anomaly scores (0-1)
        """
        # Calculate Z-scores, ignoring NaN values
        mean = series.mean()
        std = series.std()

        # Handle zero variance case
        if std == 0 or pd.isna(std):
            return pd.Series(False, index=series.index), pd.Series(
                0.0, index=series.index
            )

        z_scores = np.abs((series - mean) / std)

        # Flag values beyond threshold
        flags = z_scores > self.z_score_threshold

        # Normalize scores to 0-1 range (cap at 10 sigma)
        normalized_scores = np.minimum(z_scores / 10.0, 1.0)

        return flags, normalized_scores

    def _detect_iqr(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Detect anomalies using IQR method.

        Args:
            series: Numeric series to analyze

        Returns:
            Tuple of (flags, scores) where:
                - flags: Boolean series indicating anomalies
                - scores: Numeric series with anomaly scores (0-1)
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        # Handle zero IQR case (all values in same range)
        if iqr == 0 or pd.isna(iqr):
            return pd.Series(False, index=series.index), pd.Series(
                0.0, index=series.index
            )

        # Standard IQR bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Flag values outside bounds
        flags = (series < lower_bound) | (series > upper_bound)

        # Calculate scores based on distance from bounds
        scores = pd.Series(0.0, index=series.index)

        # Lower outliers
        lower_mask = series < lower_bound
        if lower_mask.any():
            distance = (lower_bound - series[lower_mask]) / iqr
            scores[lower_mask] = np.minimum(distance / 3.0, 1.0)  # Normalize to 0-1

        # Upper outliers
        upper_mask = series > upper_bound
        if upper_mask.any():
            distance = (series[upper_mask] - upper_bound) / iqr
            scores[upper_mask] = np.minimum(distance / 3.0, 1.0)  # Normalize to 0-1

        return flags, scores

    def _calculate_confidence(self, scores: pd.Series) -> pd.Series:
        """Calculate confidence scores for anomaly detection.

        Higher anomaly scores get higher confidence. Confidence is the anomaly
        score itself, already normalized to 0-1.

        Args:
            scores: Anomaly scores (0-1)

        Returns:
            Confidence scores (0-1)
        """
        # Confidence is directly proportional to anomaly score
        # Already normalized to 0-1 range
        return scores
