"""Base classes for time series forecasting models.

This module provides abstract interfaces for time series models that forecast
driver/team performance and championship standings.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class TimeSeriesPredictor(ABC):
    """Abstract base class for time series forecasting models."""

    @abstractmethod
    def fit(
        self, historical_data: pd.DataFrame, target_column: str
    ) -> "TimeSeriesPredictor":
        """Fit the time series model on historical data.

        Args:
            historical_data: Time-ordered DataFrame with target values
            target_column: Name of column to forecast

        Returns:
            Fitted model instance (self)
        """
        pass

    @abstractmethod
    def predict(
        self, steps_ahead: int = 3, confidence_level: float = 0.95
    ) -> dict[str, Any]:
        """Generate forecasts for future periods.

        Args:
            steps_ahead: Number of periods to forecast ahead
            confidence_level: Confidence level for prediction intervals (0.95 = 95% CI)

        Returns:
            Dictionary with 'forecast', 'lower_bound', 'upper_bound', 'confidence'
        """
        pass

    @abstractmethod
    def get_trend(self) -> dict[str, Any]:
        """Extract trend information from the model.

        Returns:
            Dictionary with trend metrics (direction, strength, change_points)
        """
        pass


class MultiVariateTimeSeriesPredictor(TimeSeriesPredictor):
    """Base for multivariate time series models with external features."""

    @abstractmethod
    def fit_with_exogenous(
        self,
        historical_data: pd.DataFrame,
        target_column: str,
        exogenous_features: list[str],
    ) -> "MultiVariateTimeSeriesPredictor":
        """Fit model with external (exogenous) variables.

        Args:
            historical_data: Time-ordered DataFrame
            target_column: Column to forecast
            exogenous_features: External variables to use

        Returns:
            Fitted model instance
        """
        pass

    @abstractmethod
    def predict_with_exogenous(
        self,
        future_exogenous: pd.DataFrame,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """Forecast using external features.

        Args:
            future_exogenous: Future values of exogenous variables
            confidence_level: Confidence level for intervals

        Returns:
            Forecast dictionary
        """
        pass


class NeuralTimeSeriesPredictor(TimeSeriesPredictor):
    """Base for deep learning time series models (LSTM, GRU)."""

    @abstractmethod
    def set_sequence_length(self, length: int) -> "NeuralTimeSeriesPredictor":
        """Set the lookback window size for sequences.

        Args:
            length: Number of past time steps to use for prediction

        Returns:
            Model instance for chaining
        """
        pass

    @abstractmethod
    def get_attention_weights(self) -> np.ndarray | None:
        """Get attention weights if model uses attention mechanism.

        Returns:
            Attention weight matrix or None if not applicable
        """
        pass
