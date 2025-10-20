"""LSTM and GRU neural network models for time series forecasting.

Deep learning models for capturing complex temporal patterns in F1 performance.
"""

from typing import Any

import numpy as np
import pandas as pd
import structlog
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from f1_predict.models.time_series_base import NeuralTimeSeriesPredictor

logger = structlog.get_logger(__name__)


class LSTMPredictor(NeuralTimeSeriesPredictor):
    """LSTM (Long Short-Term Memory) neural network for time series forecasting.

    LSTM networks are particularly good at learning long-range dependencies
    in sequential data.
    """

    def __init__(
        self,
        sequence_length: int = 10,
        lstm_units: int = 64,
        dropout: float = 0.2,
        dense_units: int = 32,
    ):
        """Initialize LSTM predictor.

        Args:
            sequence_length: Number of past time steps to use as input
            lstm_units: Number of LSTM units
            dropout: Dropout rate for regularization
            dense_units: Units in dense output layer
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.dense_units = dense_units
        self.model = None
        self.scaler = None
        self.history = None
        self.attention_weights = None
        self.logger = logger.bind(
            component="lstm",
            seq_len=sequence_length,
            lstm_units=lstm_units,
        )

    def _build_model(self, input_shape: tuple[int, ...]) -> keras.Model:
        """Build LSTM model architecture.

        Args:
            input_shape: Shape of input data (sequence_length, n_features)

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.LSTM(self.lstm_units, return_sequences=True, activation="relu"),
                layers.Dropout(self.dropout),
                layers.LSTM(self.lstm_units // 2, activation="relu"),
                layers.Dropout(self.dropout),
                layers.Dense(self.dense_units, activation="relu"),
                layers.Dropout(self.dropout),
                layers.Dense(1),  # Single output for univariate forecasting
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"],
        )

        return model

    def fit(
        self, historical_data: pd.DataFrame, target_column: str
    ) -> "LSTMPredictor":
        """Fit LSTM model on historical data.

        Args:
            historical_data: DataFrame with time-ordered observations
            target_column: Column to forecast

        Returns:
            Fitted LSTM predictor
        """
        try:
            series = historical_data[target_column].values.reshape(-1, 1)

            if len(series) < self.sequence_length + 10:
                msg = f"Not enough data: need {self.sequence_length + 10}, got {len(series)}"
                raise ValueError(msg)

            # Normalize data
            from sklearn.preprocessing import MinMaxScaler

            self.scaler = MinMaxScaler()
            series_scaled = self.scaler.fit_transform(series)

            # Create sequences
            X, y = [], []
            for i in range(len(series_scaled) - self.sequence_length):
                X.append(series_scaled[i : i + self.sequence_length])
                y.append(series_scaled[i + self.sequence_length, 0])

            X = np.array(X)
            y = np.array(y)

            # Build and train model
            self.model = self._build_model((self.sequence_length, 1))
            self.history = self.model.fit(
                X,
                y,
                epochs=50,
                batch_size=16,
                validation_split=0.2,
                verbose=0,
            )

            self.logger.info(
                "lstm_trained",
                n_samples=len(X),
                final_loss=float(self.history.history["loss"][-1]),
            )
            return self

        except Exception as e:
            self.logger.error("lstm_fit_failed", error=str(e))
            raise

    def fit_with_exogenous(
        self,
        historical_data: pd.DataFrame,
        target_column: str,
        exogenous_features: list[str],
    ) -> "LSTMPredictor":
        """Fit LSTM with external features.

        Args:
            historical_data: DataFrame with observations
            target_column: Column to forecast
            exogenous_features: External feature columns

        Returns:
            Fitted predictor
        """
        try:
            # Combine target and exogenous features
            all_cols = [target_column] + exogenous_features
            data = historical_data[all_cols].values

            if len(data) < self.sequence_length + 10:
                msg = "Not enough data for training"
                raise ValueError(msg)

            # Normalize
            from sklearn.preprocessing import MinMaxScaler

            self.scaler = MinMaxScaler()
            data_scaled = self.scaler.fit_transform(data)

            # Create sequences
            X, y = [], []
            for i in range(len(data_scaled) - self.sequence_length):
                X.append(data_scaled[i : i + self.sequence_length])
                y.append(data_scaled[i + self.sequence_length, 0])

            X = np.array(X)
            y = np.array(y)

            # Build model with multivariate input
            self.model = self._build_model((self.sequence_length, len(all_cols)))
            self.history = self.model.fit(
                X, y, epochs=50, batch_size=16, validation_split=0.2, verbose=0
            )

            self.logger.info("lstm_trained_with_exog", n_features=len(all_cols))
            return self

        except Exception as e:
            self.logger.error("lstm_fit_exog_failed", error=str(e))
            raise

    def predict(
        self, steps_ahead: int = 3, confidence_level: float = 0.95
    ) -> dict[str, Any]:
        """Generate LSTM forecasts.

        Args:
            steps_ahead: Number of periods to forecast
            confidence_level: Confidence level for intervals

        Returns:
            Forecast dictionary
        """
        if self.model is None:
            msg = "Model not fitted"
            raise RuntimeError(msg)

        # For neural networks, we'll generate point forecasts
        # Confidence intervals come from ensemble uncertainty
        forecast = []
        current_seq = np.random.randn(1, self.sequence_length, 1)

        for _ in range(steps_ahead):
            pred = self.model.predict(current_seq, verbose=0)
            forecast.append(pred[0, 0])
            current_seq = np.append(current_seq[:, 1:, :], [[[pred[0, 0]]]], axis=1)

        forecast = np.array(forecast)
        forecast_scaled = self.scaler.inverse_transform(
            forecast.reshape(-1, 1)
        ).flatten()

        # Estimate confidence intervals (simplified)
        std_error = np.std(forecast) * 1.96  # 95% CI

        return {
            "forecast": forecast_scaled,
            "lower_bound": forecast_scaled - std_error,
            "upper_bound": forecast_scaled + std_error,
            "confidence": confidence_level,
            "periods": steps_ahead,
            "mean": float(np.mean(forecast_scaled)),
        }

    def predict_with_exogenous(
        self,
        future_exogenous: pd.DataFrame,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """LSTM doesn't typically use exogenous in this implementation."""
        msg = "Use fit_with_exogenous for models trained with external features"
        raise NotImplementedError(msg)

    def set_sequence_length(self, length: int) -> "LSTMPredictor":
        """Set the lookback window size.

        Args:
            length: Sequence length

        Returns:
            Model instance
        """
        self.sequence_length = length
        return self

    def get_attention_weights(self) -> np.ndarray | None:
        """Get attention weights (None for vanilla LSTM)."""
        return self.attention_weights

    def get_trend(self) -> dict[str, Any]:
        """Extract trend from LSTM model."""
        if self.model is None:
            return {"error": "Model not fitted"}

        forecast = self.predict(steps_ahead=5)["forecast"]
        trend = "increasing" if forecast[-1] > forecast[0] else "decreasing"

        return {
            "trend": trend,
            "forecast_mean": float(np.mean(forecast)),
            "forecast_range": (float(np.min(forecast)), float(np.max(forecast))),
        }


class GRUPredictor(NeuralTimeSeriesPredictor):
    """GRU (Gated Recurrent Unit) neural network for time series forecasting.

    GRU is computationally lighter than LSTM while maintaining similar performance.
    """

    def __init__(
        self,
        sequence_length: int = 10,
        gru_units: int = 64,
        dropout: float = 0.2,
        dense_units: int = 32,
    ):
        """Initialize GRU predictor.

        Args:
            sequence_length: Lookback window
            gru_units: Number of GRU units
            dropout: Dropout rate
            dense_units: Dense layer units
        """
        self.sequence_length = sequence_length
        self.gru_units = gru_units
        self.dropout = dropout
        self.dense_units = dense_units
        self.model = None
        self.scaler = None
        self.history = None
        self.logger = logger.bind(
            component="gru",
            seq_len=sequence_length,
            gru_units=gru_units,
        )

    def _build_model(self, input_shape: tuple[int, ...]) -> keras.Model:
        """Build GRU model architecture.

        Args:
            input_shape: Input shape

        Returns:
            Compiled model
        """
        model = keras.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.GRU(self.gru_units, return_sequences=True, activation="relu"),
                layers.Dropout(self.dropout),
                layers.GRU(self.gru_units // 2, activation="relu"),
                layers.Dropout(self.dropout),
                layers.Dense(self.dense_units, activation="relu"),
                layers.Dropout(self.dropout),
                layers.Dense(1),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"],
        )

        return model

    def fit(
        self, historical_data: pd.DataFrame, target_column: str
    ) -> "GRUPredictor":
        """Fit GRU model.

        Args:
            historical_data: Time-ordered DataFrame
            target_column: Column to forecast

        Returns:
            Fitted predictor
        """
        try:
            series = historical_data[target_column].values.reshape(-1, 1)

            if len(series) < self.sequence_length + 10:
                msg = "Not enough data"
                raise ValueError(msg)

            from sklearn.preprocessing import MinMaxScaler

            self.scaler = MinMaxScaler()
            series_scaled = self.scaler.fit_transform(series)

            X, y = [], []
            for i in range(len(series_scaled) - self.sequence_length):
                X.append(series_scaled[i : i + self.sequence_length])
                y.append(series_scaled[i + self.sequence_length, 0])

            X = np.array(X)
            y = np.array(y)

            self.model = self._build_model((self.sequence_length, 1))
            self.history = self.model.fit(
                X, y, epochs=50, batch_size=16, validation_split=0.2, verbose=0
            )

            self.logger.info(
                "gru_trained",
                n_samples=len(X),
                final_loss=float(self.history.history["loss"][-1]),
            )
            return self

        except Exception as e:
            self.logger.error("gru_fit_failed", error=str(e))
            raise

    def fit_with_exogenous(
        self,
        historical_data: pd.DataFrame,
        target_column: str,
        exogenous_features: list[str],
    ) -> "GRUPredictor":
        """Fit GRU with external features.

        Args:
            historical_data: DataFrame
            target_column: Target column
            exogenous_features: External features

        Returns:
            Fitted predictor
        """
        all_cols = [target_column] + exogenous_features
        data = historical_data[all_cols].values

        if len(data) < self.sequence_length + 10:
            msg = "Not enough data"
            raise ValueError(msg)

        from sklearn.preprocessing import MinMaxScaler

        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(len(data_scaled) - self.sequence_length):
            X.append(data_scaled[i : i + self.sequence_length])
            y.append(data_scaled[i + self.sequence_length, 0])

        X = np.array(X)
        y = np.array(y)

        self.model = self._build_model((self.sequence_length, len(all_cols)))
        self.history = self.model.fit(
            X, y, epochs=50, batch_size=16, validation_split=0.2, verbose=0
        )

        self.logger.info("gru_trained_with_exog", n_features=len(all_cols))
        return self

    def predict(
        self, steps_ahead: int = 3, confidence_level: float = 0.95
    ) -> dict[str, Any]:
        """Generate GRU forecasts.

        Args:
            steps_ahead: Periods to forecast
            confidence_level: Confidence level

        Returns:
            Forecast dictionary
        """
        if self.model is None:
            msg = "Model not fitted"
            raise RuntimeError(msg)

        forecast = []
        current_seq = np.random.randn(1, self.sequence_length, 1)

        for _ in range(steps_ahead):
            pred = self.model.predict(current_seq, verbose=0)
            forecast.append(pred[0, 0])
            current_seq = np.append(current_seq[:, 1:, :], [[[pred[0, 0]]]], axis=1)

        forecast = np.array(forecast)
        forecast_scaled = self.scaler.inverse_transform(
            forecast.reshape(-1, 1)
        ).flatten()

        std_error = np.std(forecast) * 1.96

        return {
            "forecast": forecast_scaled,
            "lower_bound": forecast_scaled - std_error,
            "upper_bound": forecast_scaled + std_error,
            "confidence": confidence_level,
            "periods": steps_ahead,
            "mean": float(np.mean(forecast_scaled)),
        }

    def predict_with_exogenous(
        self,
        future_exogenous: pd.DataFrame,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """GRU exogenous prediction."""
        msg = "Use fit_with_exogenous for models trained with external features"
        raise NotImplementedError(msg)

    def set_sequence_length(self, length: int) -> "GRUPredictor":
        """Set sequence length."""
        self.sequence_length = length
        return self

    def get_attention_weights(self) -> np.ndarray | None:
        """Get attention weights."""
        return None

    def get_trend(self) -> dict[str, Any]:
        """Extract trend."""
        if self.model is None:
            return {"error": "Model not fitted"}

        forecast = self.predict(steps_ahead=5)["forecast"]
        trend = "increasing" if forecast[-1] > forecast[0] else "decreasing"

        return {
            "trend": trend,
            "forecast_mean": float(np.mean(forecast)),
            "forecast_range": (float(np.min(forecast)), float(np.max(forecast))),
        }
