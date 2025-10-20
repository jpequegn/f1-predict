"""Temporal feature engineering for time series models.

This module provides tools for creating time series features like rolling statistics,
exponential moving averages, lag features, and momentum indicators.
"""

from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class TemporalFeatureEngineer:
    """Engineer temporal features from historical performance data."""

    def __init__(self, lookback_windows: list[int] | None = None):
        """Initialize temporal feature engineer.

        Args:
            lookback_windows: Sizes of rolling windows (default: [3, 5, 10])
        """
        self.lookback_windows = lookback_windows or [3, 5, 10]
        self.logger = logger.bind(component="temporal_features")

    def create_rolling_statistics(
        self,
        series: pd.Series,
        windows: list[int] | None = None,
        include_std: bool = True,
    ) -> pd.DataFrame:
        """Create rolling mean and standard deviation features.

        Args:
            series: Time series of values
            windows: Lookback window sizes (default: self.lookback_windows)
            include_std: Whether to include standard deviation

        Returns:
            DataFrame with rolling statistics
        """
        windows = windows or self.lookback_windows
        features = pd.DataFrame(index=series.index)

        for w in windows:
            features[f"rolling_mean_{w}"] = series.rolling(window=w, min_periods=1).mean()
            if include_std:
                features[f"rolling_std_{w}"] = series.rolling(window=w, min_periods=1).std()

        return features

    def create_exponential_moving_average(
        self, series: pd.Series, spans: list[int] | None = None
    ) -> pd.DataFrame:
        """Create exponential moving average (EMA) features.

        Args:
            series: Time series of values
            spans: EMA span values (default: [3, 5, 10])

        Returns:
            DataFrame with EMA features
        """
        spans = spans or self.lookback_windows
        features = pd.DataFrame(index=series.index)

        for span in spans:
            features[f"ema_{span}"] = series.ewm(span=span, adjust=False).mean()

        return features

    def create_lag_features(
        self, series: pd.Series, lags: list[int] | None = None
    ) -> pd.DataFrame:
        """Create lag features (previous values).

        Args:
            series: Time series of values
            lags: Lag amounts in periods (default: [1, 2, 3, 5])

        Returns:
            DataFrame with lagged values
        """
        lags = lags or [1, 2, 3, 5]
        features = pd.DataFrame(index=series.index)

        for lag in lags:
            features[f"lag_{lag}"] = series.shift(lag)

        return features

    def create_momentum_features(self, series: pd.Series) -> pd.DataFrame:
        """Create momentum indicators (rate of change).

        Args:
            series: Time series of values

        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=series.index)

        # Rate of change
        for period in [1, 3, 5]:
            features[f"roc_{period}"] = series.pct_change(periods=period)

        # Momentum (absolute change)
        for period in [3, 5]:
            features[f"momentum_{period}"] = series.diff(periods=period)

        # Acceleration (change in momentum)
        features["acceleration"] = features["momentum_3"].diff()

        return features

    def create_trend_features(self, series: pd.Series) -> pd.DataFrame:
        """Create trend indicators.

        Args:
            series: Time series of values

        Returns:
            DataFrame with trend features
        """
        features = pd.DataFrame(index=series.index)

        # Linear regression slope over windows
        for window in [3, 5, 10]:
            slopes = []
            for i in range(len(series)):
                if i < window:
                    slopes.append(np.nan)
                else:
                    window_data = series.iloc[i - window : i].values
                    x = np.arange(len(window_data))
                    slope = np.polyfit(x, window_data, 1)[0]
                    slopes.append(slope)
            features[f"trend_slope_{window}"] = slopes

        # Higher/Lower than MA
        for window in [5, 10]:
            ma = series.rolling(window=window, min_periods=1).mean()
            features[f"above_ma_{window}"] = (series > ma).astype(int)

        return features

    def create_seasonality_features(
        self, dates: pd.DatetimeIndex, season_length: int = 23
    ) -> pd.DataFrame:
        """Create seasonality indicators (for F1, season_length ~23 races).

        Args:
            dates: DateTime index of observations
            season_length: Typical season length (default: 23 races)

        Returns:
            DataFrame with seasonality features
        """
        features = pd.DataFrame(index=dates)

        # Season progress (0 = start, 1 = end)
        race_number = np.arange(len(dates)) % season_length + 1
        features["season_progress"] = race_number / season_length

        # Season phase (early/mid/late)
        features["phase_early"] = (race_number <= 7).astype(int)
        features["phase_mid"] = ((race_number > 7) & (race_number <= 17)).astype(int)
        features["phase_late"] = (race_number > 17).astype(int)

        # Harmonic features for cyclical patterns
        features["sin_season"] = np.sin(2 * np.pi * race_number / season_length)
        features["cos_season"] = np.cos(2 * np.pi * race_number / season_length)

        return features

    def prepare_multivariate_time_series(
        self,
        driver_data: pd.DataFrame,
        target_column: str,
        look_back: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data for neural network training.

        Creates sequences of (X, y) pairs for supervised learning.

        Args:
            driver_data: Time-ordered DataFrame with features
            target_column: Column to predict
            look_back: Sequence length (window of past observations)

        Returns:
            Tuple of (X sequences, y targets) as numpy arrays
        """
        data = driver_data.values
        X, y = [], []

        for i in range(len(data) - look_back):
            X.append(data[i : i + look_back])
            y.append(data[i + look_back, driver_data.columns.get_loc(target_column)])

        return np.array(X), np.array(y)

    def identify_form_trajectory(
        self, performance_series: pd.Series, window: int = 5
    ) -> dict[str, Any]:
        """Identify if performance is improving or declining.

        Args:
            performance_series: Time series of performance values
            window: Window size for trend analysis

        Returns:
            Dictionary with trajectory info (improving/declining, strength)
        """
        if len(performance_series) < window:
            return {"trajectory": "insufficient_data", "strength": 0.0}

        recent = performance_series.iloc[-window:].values
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]

        # Calculate strength (R-squared)
        predicted = np.polyfit(x, recent, 1)[0] * x + np.polyfit(x, recent, 1)[1]
        ss_res = np.sum((recent - predicted) ** 2)
        ss_tot = np.sum((recent - np.mean(recent)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        trajectory = "improving" if slope > 0 else "declining"

        return {
            "trajectory": trajectory,
            "slope": float(slope),
            "strength": float(abs(slope)),
            "consistency": float(r_squared),
        }

    def detect_inflection_points(
        self, series: pd.Series, threshold: float = 0.5
    ) -> list[int]:
        """Detect significant changes in direction (inflection points).

        Args:
            series: Time series of values
            threshold: Acceleration threshold to detect inflection

        Returns:
            List of indices where significant direction changes occur
        """
        if len(series) < 3:
            return []

        # Calculate second derivative (acceleration)
        first_diff = series.diff()
        second_diff = first_diff.diff()

        # Detect sign changes in acceleration
        inflection_points = []
        for i in range(1, len(second_diff)):
            if (
                not np.isnan(second_diff.iloc[i])
                and not np.isnan(second_diff.iloc[i - 1])
                and second_diff.iloc[i] * second_diff.iloc[i - 1] < 0
                and abs(second_diff.iloc[i]) > threshold
            ):
                inflection_points.append(i)

        return inflection_points
