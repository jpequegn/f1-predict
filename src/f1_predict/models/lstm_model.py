"""LSTM and GRU neural network models for time series forecasting.

Deep learning models for capturing complex temporal patterns in F1 performance.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import structlog
import torch
from torch import nn

from f1_predict.models.time_series_base import NeuralTimeSeriesPredictor

logger = structlog.get_logger(__name__)


class LSTMNet(nn.Module):
    """PyTorch LSTM neural network module."""

    def __init__(self, input_size: int, lstm_units: int, dropout: float, dense_units: int):
        """Initialize LSTM network.

        Args:
            input_size: Number of input features
            lstm_units: Number of LSTM units
            dropout: Dropout rate
            dense_units: Units in dense layer
        """
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, lstm_units, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units // 2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.dense = nn.Linear(lstm_units // 2, dense_units)
        self.dropout3 = nn.Dropout(dropout)
        self.output = nn.Linear(dense_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output predictions
        """
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        # Take last output
        last_output = lstm_out2[:, -1, :]
        dense_out = torch.relu(self.dense(last_output))
        dense_out = self.dropout3(dense_out)
        output = self.output(dense_out)
        return output


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger.bind(
            component="lstm",
            seq_len=sequence_length,
            lstm_units=lstm_units,
        )

    def _build_model(self, input_size: int) -> LSTMNet:
        """Build LSTM model architecture.

        Args:
            input_size: Number of input features

        Returns:
            LSTM network
        """
        return LSTMNet(
            input_size=input_size,
            lstm_units=self.lstm_units,
            dropout=self.dropout,
            dense_units=self.dense_units,
        ).to(self.device)

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
            self.scaler = MinMaxScaler()
            series_scaled = self.scaler.fit_transform(series)

            # Create sequences
            X, y = [], []
            for i in range(len(series_scaled) - self.sequence_length):
                X.append(series_scaled[i : i + self.sequence_length])
                y.append(series_scaled[i + self.sequence_length, 0])

            X = np.array(X)
            y = np.array(y)

            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)

            # Build and train model
            self.model = self._build_model(1)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # Training loop
            epochs = 50
            batch_size = 16
            losses = []

            for _epoch in range(epochs):
                epoch_loss = 0
                for i in range(0, len(X_tensor), batch_size):
                    batch_X = X_tensor[i : i + batch_size]
                    batch_y = y_tensor[i : i + batch_size]

                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                losses.append(epoch_loss / (len(X_tensor) // batch_size + 1))

            self.history = {"loss": losses}

            self.logger.info(
                "lstm_trained",
                n_samples=len(X),
                final_loss=float(losses[-1]),
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
            self.scaler = MinMaxScaler()
            data_scaled = self.scaler.fit_transform(data)

            # Create sequences
            X, y = [], []
            for i in range(len(data_scaled) - self.sequence_length):
                X.append(data_scaled[i : i + self.sequence_length])
                y.append(data_scaled[i + self.sequence_length, 0])

            X = np.array(X)
            y = np.array(y)

            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)

            # Build model with multivariate input
            self.model = self._build_model(len(all_cols))
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # Training loop
            epochs = 50
            batch_size = 16
            losses = []

            for _epoch in range(epochs):
                epoch_loss = 0
                for i in range(0, len(X_tensor), batch_size):
                    batch_X = X_tensor[i : i + batch_size]
                    batch_y = y_tensor[i : i + batch_size]

                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                losses.append(epoch_loss / (len(X_tensor) // batch_size + 1))

            self.history = {"loss": losses}

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

        self.model.eval()
        with torch.no_grad():
            forecast = []
            current_seq = torch.randn(1, self.sequence_length, 1).to(self.device)

            for _ in range(steps_ahead):
                pred = self.model(current_seq)
                forecast.append(pred.item())
                # Shift sequence and add new prediction
                current_seq = torch.cat(
                    [current_seq[:, 1:, :], pred.reshape(1, 1, 1)], dim=1
                )

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


class GRUNet(nn.Module):
    """PyTorch GRU neural network module."""

    def __init__(self, input_size: int, gru_units: int, dropout: float, dense_units: int):
        """Initialize GRU network.

        Args:
            input_size: Number of input features
            gru_units: Number of GRU units
            dropout: Dropout rate
            dense_units: Units in dense layer
        """
        super().__init__()
        self.gru1 = nn.GRU(input_size, gru_units, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(gru_units, gru_units // 2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.dense = nn.Linear(gru_units // 2, dense_units)
        self.dropout3 = nn.Dropout(dropout)
        self.output = nn.Linear(dense_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output predictions
        """
        gru_out1, _ = self.gru1(x)
        gru_out1 = self.dropout1(gru_out1)
        gru_out2, _ = self.gru2(gru_out1)
        gru_out2 = self.dropout2(gru_out2)
        # Take last output
        last_output = gru_out2[:, -1, :]
        dense_out = torch.relu(self.dense(last_output))
        dense_out = self.dropout3(dense_out)
        output = self.output(dense_out)
        return output


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger.bind(
            component="gru",
            seq_len=sequence_length,
            gru_units=gru_units,
        )

    def _build_model(self, input_size: int) -> GRUNet:
        """Build GRU model architecture.

        Args:
            input_size: Number of input features

        Returns:
            GRU network
        """
        return GRUNet(
            input_size=input_size,
            gru_units=self.gru_units,
            dropout=self.dropout,
            dense_units=self.dense_units,
        ).to(self.device)

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

            self.scaler = MinMaxScaler()
            series_scaled = self.scaler.fit_transform(series)

            X, y = [], []
            for i in range(len(series_scaled) - self.sequence_length):
                X.append(series_scaled[i : i + self.sequence_length])
                y.append(series_scaled[i + self.sequence_length, 0])

            X = np.array(X)
            y = np.array(y)

            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)

            self.model = self._build_model(1)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # Training loop
            epochs = 50
            batch_size = 16
            losses = []

            for _epoch in range(epochs):
                epoch_loss = 0
                for i in range(0, len(X_tensor), batch_size):
                    batch_X = X_tensor[i : i + batch_size]
                    batch_y = y_tensor[i : i + batch_size]

                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                losses.append(epoch_loss / (len(X_tensor) // batch_size + 1))

            self.history = {"loss": losses}

            self.logger.info(
                "gru_trained",
                n_samples=len(X),
                final_loss=float(losses[-1]),
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

        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(len(data_scaled) - self.sequence_length):
            X.append(data_scaled[i : i + self.sequence_length])
            y.append(data_scaled[i + self.sequence_length, 0])

        X = np.array(X)
        y = np.array(y)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)

        self.model = self._build_model(len(all_cols))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training loop
        epochs = 50
        batch_size = 16
        losses = []

        for _epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i : i + batch_size]
                batch_y = y_tensor[i : i + batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            losses.append(epoch_loss / (len(X_tensor) // batch_size + 1))

        self.history = {"loss": losses}

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

        self.model.eval()
        with torch.no_grad():
            forecast = []
            current_seq = torch.randn(1, self.sequence_length, 1).to(self.device)

            for _ in range(steps_ahead):
                pred = self.model(current_seq)
                forecast.append(pred.item())
                # Shift sequence and add new prediction
                current_seq = torch.cat(
                    [current_seq[:, 1:, :], pred.reshape(1, 1, 1)], dim=1
                )

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
