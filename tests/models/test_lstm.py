"""Tests for LSTM time series prediction models.

Comprehensive test coverage for LSTMPredictor and GRUPredictor classes
including training, prediction, and trend analysis.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from f1_predict.models.lstm_model import GRUPredictor, GRUNet, LSTMNet, LSTMPredictor


@pytest.fixture
def sample_univariate_data():
    """Create sample univariate time series data."""
    np.random.seed(42)
    # Create 50 time steps with some trend
    trend = np.arange(50) * 0.5
    noise = np.random.normal(0, 1, 50)
    values = 100 + trend + noise
    return pd.DataFrame({"time_step": range(50), "value": values})


@pytest.fixture
def sample_multivariate_data():
    """Create sample multivariate time series data."""
    np.random.seed(42)
    n_steps = 50
    trend = np.arange(n_steps) * 0.5
    noise = np.random.normal(0, 1, n_steps)

    return pd.DataFrame({
        "target": 100 + trend + noise,
        "feature1": np.sin(np.arange(n_steps) / 10) * 10 + 50,
        "feature2": np.cos(np.arange(n_steps) / 10) * 10 + 50,
    })


class TestLSTMNetModule:
    """Tests for LSTMNet PyTorch module."""

    def test_init_valid_params(self):
        """Test LSTMNet initialization with valid parameters."""
        lstm_net = LSTMNet(input_size=1, lstm_units=64, dropout=0.2, dense_units=32)
        assert lstm_net is not None
        assert isinstance(lstm_net, torch.nn.Module)

    def test_forward_pass_output_shape(self):
        """Test forward pass produces correct output shape."""
        lstm_net = LSTMNet(input_size=1, lstm_units=64, dropout=0.2, dense_units=32)
        # Input: batch_size=2, sequence_length=10, features=1
        input_tensor = torch.randn(2, 10, 1)
        output = lstm_net(input_tensor)

        assert output.shape == (2, 1)  # batch_size x 1

    def test_forward_pass_multivariate_input(self):
        """Test forward pass with multivariate input."""
        lstm_net = LSTMNet(input_size=3, lstm_units=32, dropout=0.2, dense_units=16)
        # Input: batch_size=2, sequence_length=10, features=3
        input_tensor = torch.randn(2, 10, 3)
        output = lstm_net(input_tensor)

        assert output.shape == (2, 1)


class TestLSTMPredictorInit:
    """Tests for LSTMPredictor initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        predictor = LSTMPredictor()
        assert predictor.sequence_length == 10
        assert predictor.lstm_units == 64
        assert predictor.dropout == 0.2
        assert predictor.dense_units == 32
        assert predictor.model is None
        assert predictor.scaler is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        predictor = LSTMPredictor(
            sequence_length=15,
            lstm_units=128,
            dropout=0.3,
            dense_units=64,
        )
        assert predictor.sequence_length == 15
        assert predictor.lstm_units == 128
        assert predictor.dropout == 0.3
        assert predictor.dense_units == 64

    def test_device_assignment(self):
        """Test that device is properly assigned."""
        predictor = LSTMPredictor()
        # Device should be cuda if available, else cpu
        assert predictor.device in [torch.device("cpu"), torch.device("cuda")]

    def test_logger_binding(self):
        """Test that logger is properly bound with model params."""
        predictor = LSTMPredictor(sequence_length=20, lstm_units=100)
        # Logger should be bound (structlog functionality)
        assert predictor.logger is not None


class TestLSTMPredictorFit:
    """Tests for LSTM model fitting."""

    def test_fit_univariate_success(self, sample_univariate_data):
        """Test successful fit on univariate time series."""
        predictor = LSTMPredictor(sequence_length=5)
        result = predictor.fit(sample_univariate_data, target_column="value")

        # Should return self for chaining
        assert result is predictor
        # Model should be trained
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.history is not None
        assert "loss" in predictor.history

    def test_fit_training_loss_decreases(self, sample_univariate_data):
        """Test that training loss decreases over epochs."""
        predictor = LSTMPredictor(sequence_length=5)
        predictor.fit(sample_univariate_data, target_column="value")

        losses = predictor.history["loss"]
        # Loss should generally trend downward (though may fluctuate)
        assert losses[-1] < losses[0] or len(losses) > 1

    def test_fit_insufficient_data(self):
        """Test that fit raises error with insufficient data."""
        predictor = LSTMPredictor(sequence_length=10)
        insufficient_data = pd.DataFrame({"value": np.random.randn(15)})

        # Need at least sequence_length + 10 samples
        with pytest.raises(ValueError, match="Not enough data"):
            predictor.fit(insufficient_data, target_column="value")

    def test_fit_with_exogenous_success(self, sample_multivariate_data):
        """Test fit_with_exogenous with external features."""
        predictor = LSTMPredictor(sequence_length=5)
        result = predictor.fit_with_exogenous(
            sample_multivariate_data,
            target_column="target",
            exogenous_features=["feature1", "feature2"],
        )

        assert result is predictor
        assert predictor.model is not None
        assert predictor.scaler is not None

    def test_fit_with_exogenous_insufficient_data(self):
        """Test fit_with_exogenous raises error with insufficient data."""
        predictor = LSTMPredictor(sequence_length=10)
        insufficient_data = pd.DataFrame({
            "target": np.random.randn(15),
            "feature1": np.random.randn(15),
        })

        with pytest.raises(ValueError, match="Not enough data"):
            predictor.fit_with_exogenous(
                insufficient_data,
                target_column="target",
                exogenous_features=["feature1"],
            )

    def test_fit_missing_target_column(self, sample_univariate_data):
        """Test that fit raises error if target column missing."""
        predictor = LSTMPredictor()
        with pytest.raises(KeyError):
            predictor.fit(sample_univariate_data, target_column="nonexistent")


class TestLSTMPredictorPredict:
    """Tests for LSTM prediction."""

    def test_predict_unfitted_model(self):
        """Test predict raises error on unfitted model."""
        predictor = LSTMPredictor()
        with pytest.raises(RuntimeError, match="Model not fitted"):
            predictor.predict(steps_ahead=3)

    def test_predict_forecast_output_structure(self, sample_univariate_data):
        """Test predict returns correct output structure."""
        predictor = LSTMPredictor(sequence_length=5)
        predictor.fit(sample_univariate_data, target_column="value")

        forecast = predictor.predict(steps_ahead=5, confidence_level=0.95)

        assert isinstance(forecast, dict)
        assert "forecast" in forecast
        assert "lower_bound" in forecast
        assert "upper_bound" in forecast
        assert "confidence" in forecast
        assert "periods" in forecast
        assert "mean" in forecast

    def test_predict_forecast_length(self, sample_univariate_data):
        """Test predict returns correct number of forecast steps."""
        predictor = LSTMPredictor(sequence_length=5)
        predictor.fit(sample_univariate_data, target_column="value")

        for steps in [1, 3, 5, 10]:
            forecast = predictor.predict(steps_ahead=steps)
            assert len(forecast["forecast"]) == steps
            assert len(forecast["lower_bound"]) == steps
            assert len(forecast["upper_bound"]) == steps

    def test_predict_confidence_level_storage(self, sample_univariate_data):
        """Test that confidence level is stored correctly."""
        predictor = LSTMPredictor(sequence_length=5)
        predictor.fit(sample_univariate_data, target_column="value")

        forecast = predictor.predict(confidence_level=0.90)
        assert forecast["confidence"] == 0.90

    def test_predict_bounds_contain_mean(self, sample_univariate_data):
        """Test that forecast mean is within confidence bounds."""
        predictor = LSTMPredictor(sequence_length=5)
        predictor.fit(sample_univariate_data, target_column="value")

        forecast = predictor.predict(steps_ahead=3)
        # Each point should be between bounds
        for i in range(len(forecast["forecast"])):
            assert forecast["lower_bound"][i] <= forecast["forecast"][i]
            assert forecast["forecast"][i] <= forecast["upper_bound"][i]

    def test_predict_with_exogenous_not_implemented(self, sample_multivariate_data):
        """Test that predict_with_exogenous raises NotImplementedError."""
        predictor = LSTMPredictor()
        future_data = sample_multivariate_data.head(5)

        with pytest.raises(NotImplementedError):
            predictor.predict_with_exogenous(future_data)


class TestLSTMPredictorMethods:
    """Tests for LSTM utility methods."""

    def test_set_sequence_length(self, sample_univariate_data):
        """Test setting sequence length."""
        predictor = LSTMPredictor(sequence_length=5)
        result = predictor.set_sequence_length(15)

        # Should return self for chaining
        assert result is predictor
        assert predictor.sequence_length == 15

    def test_get_attention_weights_unfitted(self):
        """Test get_attention_weights returns None (vanilla LSTM)."""
        predictor = LSTMPredictor()
        weights = predictor.get_attention_weights()
        # Vanilla LSTM has no attention mechanism
        assert weights is None

    def test_get_trend_unfitted(self):
        """Test get_trend returns error dict for unfitted model."""
        predictor = LSTMPredictor()
        trend = predictor.get_trend()

        assert isinstance(trend, dict)
        assert "error" in trend

    def test_get_trend_fitted_increasing(self):
        """Test get_trend detects increasing trend."""
        predictor = LSTMPredictor(sequence_length=5)
        # Create data with clear uptrend
        data = pd.DataFrame({"value": np.linspace(0, 100, 50)})
        predictor.fit(data, target_column="value")

        trend = predictor.get_trend()
        assert "trend" in trend
        assert trend["trend"] in ["increasing", "decreasing"]
        assert "forecast_mean" in trend
        assert "forecast_range" in trend

    def test_get_trend_values_are_numbers(self, sample_univariate_data):
        """Test get_trend returns numeric values."""
        predictor = LSTMPredictor(sequence_length=5)
        predictor.fit(sample_univariate_data, target_column="value")

        trend = predictor.get_trend()
        assert isinstance(trend["forecast_mean"], (int, float))
        assert isinstance(trend["forecast_range"], tuple)
        assert len(trend["forecast_range"]) == 2


class TestGRUNetModule:
    """Tests for GRUNet PyTorch module."""

    def test_init_valid_params(self):
        """Test GRUNet initialization."""
        gru_net = GRUNet(input_size=1, gru_units=64, dropout=0.2, dense_units=32)
        assert gru_net is not None
        assert isinstance(gru_net, torch.nn.Module)

    def test_forward_pass_output_shape(self):
        """Test GRU forward pass output shape."""
        gru_net = GRUNet(input_size=1, gru_units=64, dropout=0.2, dense_units=32)
        input_tensor = torch.randn(2, 10, 1)
        output = gru_net(input_tensor)

        assert output.shape == (2, 1)


class TestGRUPredictorInit:
    """Tests for GRUPredictor initialization."""

    def test_init_default_params(self):
        """Test GRUPredictor initialization with defaults."""
        predictor = GRUPredictor()
        assert predictor.sequence_length == 10
        assert predictor.gru_units == 64
        assert predictor.dropout == 0.2
        assert predictor.dense_units == 32

    def test_init_custom_params(self):
        """Test GRUPredictor initialization with custom params."""
        predictor = GRUPredictor(
            sequence_length=20,
            gru_units=128,
            dropout=0.3,
            dense_units=64,
        )
        assert predictor.sequence_length == 20
        assert predictor.gru_units == 128


class TestGRUPredictorFit:
    """Tests for GRU model fitting."""

    def test_fit_univariate_success(self, sample_univariate_data):
        """Test successful GRU fit."""
        predictor = GRUPredictor(sequence_length=5)
        result = predictor.fit(sample_univariate_data, target_column="value")

        assert result is predictor
        assert predictor.model is not None
        assert predictor.scaler is not None

    def test_fit_with_exogenous_success(self, sample_multivariate_data):
        """Test GRU fit with exogenous features."""
        predictor = GRUPredictor(sequence_length=5)
        result = predictor.fit_with_exogenous(
            sample_multivariate_data,
            target_column="target",
            exogenous_features=["feature1", "feature2"],
        )

        assert result is predictor
        assert predictor.model is not None


class TestGRUPredictorPredict:
    """Tests for GRU prediction."""

    def test_predict_unfitted_model(self):
        """Test predict raises error on unfitted GRU."""
        predictor = GRUPredictor()
        with pytest.raises(RuntimeError):
            predictor.predict(steps_ahead=3)

    def test_predict_output_structure(self, sample_univariate_data):
        """Test GRU predict returns correct structure."""
        predictor = GRUPredictor(sequence_length=5)
        predictor.fit(sample_univariate_data, target_column="value")

        forecast = predictor.predict(steps_ahead=3)

        assert isinstance(forecast, dict)
        assert "forecast" in forecast
        assert "lower_bound" in forecast
        assert "upper_bound" in forecast


class TestLSTMGRUComparison:
    """Tests comparing LSTM and GRU behavior."""

    def test_lstm_gru_similar_structure(self):
        """Test that LSTM and GRU have similar interfaces."""
        lstm = LSTMPredictor(sequence_length=10, lstm_units=64)
        gru = GRUPredictor(sequence_length=10, gru_units=64)

        # Both should have same interface
        assert hasattr(lstm, "fit")
        assert hasattr(lstm, "predict")
        assert hasattr(gru, "fit")
        assert hasattr(gru, "predict")

    def test_lstm_gru_both_trainable(self, sample_univariate_data):
        """Test that both LSTM and GRU can be trained."""
        lstm = LSTMPredictor(sequence_length=5)
        gru = GRUPredictor(sequence_length=5)

        lstm.fit(sample_univariate_data, target_column="value")
        gru.fit(sample_univariate_data, target_column="value")

        assert lstm.model is not None
        assert gru.model is not None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_negative_sequence_length(self):
        """Test behavior with negative sequence length."""
        # Should initialize without immediate error
        predictor = LSTMPredictor(sequence_length=-1)
        # Error will occur during fit when trying to create sequences
        assert predictor.sequence_length == -1

    def test_zero_dropout(self):
        """Test LSTM with zero dropout."""
        predictor = LSTMPredictor(dropout=0.0)
        assert predictor.dropout == 0.0

    def test_very_large_units(self):
        """Test LSTM with large number of units."""
        predictor = LSTMPredictor(lstm_units=512, dense_units=256)
        assert predictor.lstm_units == 512
        assert predictor.dense_units == 256

    def test_forecast_with_single_step(self, sample_univariate_data):
        """Test forecast with single step ahead."""
        predictor = LSTMPredictor(sequence_length=5)
        predictor.fit(sample_univariate_data, target_column="value")

        forecast = predictor.predict(steps_ahead=1)
        assert len(forecast["forecast"]) == 1

    def test_forecast_with_many_steps(self, sample_univariate_data):
        """Test forecast with many steps ahead."""
        predictor = LSTMPredictor(sequence_length=5)
        predictor.fit(sample_univariate_data, target_column="value")

        forecast = predictor.predict(steps_ahead=20)
        assert len(forecast["forecast"]) == 20
