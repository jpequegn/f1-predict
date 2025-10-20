"""Tests for time series forecasting models."""

import numpy as np
import pandas as pd
import pytest

from f1_predict.models.arima_model import ARIMAPredictor, SARIMAPredictor
from f1_predict.models.championship_predictor import ChampionshipPredictor
from f1_predict.models.lstm_model import GRUPredictor, LSTMPredictor
from f1_predict.models.momentum_analyzer import MomentumAnalyzer
from f1_predict.models.temporal_features import TemporalFeatureEngineer


@pytest.fixture
def sample_time_series():
    """Create sample time series data."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=50, freq="W")
    values = np.cumsum(np.random.randn(50)) + 100
    return pd.DataFrame({"date": dates, "performance": values})


@pytest.fixture
def driver_standings():
    """Create sample driver standings."""
    return pd.DataFrame(
        {
            "driver_name": ["Hamilton", "Verstappen", "Leclerc"],
            "total_points": [100, 95, 85],
        }
    )


class TestTemporalFeatureEngineer:
    """Test temporal feature engineering."""

    def test_rolling_statistics(self, sample_time_series):
        """Test rolling statistics creation."""
        engineer = TemporalFeatureEngineer()
        features = engineer.create_rolling_statistics(
            sample_time_series["performance"], windows=[3, 5]
        )

        assert "rolling_mean_3" in features.columns
        assert "rolling_std_3" in features.columns
        assert len(features) == len(sample_time_series)

    def test_exponential_moving_average(self, sample_time_series):
        """Test EMA creation."""
        engineer = TemporalFeatureEngineer()
        ema = engineer.create_exponential_moving_average(
            sample_time_series["performance"], spans=[3, 5]
        )

        assert "ema_3" in ema.columns
        assert "ema_5" in ema.columns
        assert len(ema) == len(sample_time_series)

    def test_lag_features(self, sample_time_series):
        """Test lag feature creation."""
        engineer = TemporalFeatureEngineer()
        lags = engineer.create_lag_features(
            sample_time_series["performance"], lags=[1, 2, 3]
        )

        assert "lag_1" in lags.columns
        assert "lag_2" in lags.columns
        assert pd.isna(lags["lag_1"].iloc[0])

    def test_momentum_features(self, sample_time_series):
        """Test momentum indicator creation."""
        engineer = TemporalFeatureEngineer()
        momentum = engineer.create_momentum_features(sample_time_series["performance"])

        assert "roc_1" in momentum.columns
        assert "momentum_3" in momentum.columns
        assert "acceleration" in momentum.columns

    def test_form_trajectory(self, sample_time_series):
        """Test form trajectory identification."""
        engineer = TemporalFeatureEngineer()
        trajectory = engineer.identify_form_trajectory(
            sample_time_series["performance"], window=5
        )

        assert "trajectory" in trajectory
        assert "slope" in trajectory
        assert "strength" in trajectory

    def test_inflection_points(self, sample_time_series):
        """Test inflection point detection."""
        engineer = TemporalFeatureEngineer()
        points = engineer.detect_inflection_points(sample_time_series["performance"])

        assert isinstance(points, list)


class TestARIMAPredictor:
    """Test ARIMA predictor."""

    def test_initialization(self):
        """Test ARIMA initialization."""
        arima = ARIMAPredictor(order=(1, 1, 1))
        assert arima.order == (1, 1, 1)
        assert arima.fitted_model is None

    def test_fit_and_predict(self, sample_time_series):
        """Test ARIMA fit and predict."""
        arima = ARIMAPredictor(order=(1, 1, 1))
        arima.fit(sample_time_series, "performance")

        assert arima.fitted_model is not None
        assert arima.historical_data is not None

        forecast = arima.predict(steps_ahead=3)
        assert "forecast" in forecast
        assert "lower_bound" in forecast
        assert "upper_bound" in forecast
        assert len(forecast["forecast"]) == 3

    def test_get_trend(self, sample_time_series):
        """Test trend extraction."""
        arima = ARIMAPredictor()
        arima.fit(sample_time_series, "performance")

        trend = arima.get_trend()
        assert "trend" in trend
        assert "slope" in trend


class TestSARIMAPredictor:
    """Test SARIMA predictor."""

    def test_initialization(self):
        """Test SARIMA initialization."""
        sarima = SARIMAPredictor(order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
        assert sarima.order == (1, 1, 1)
        assert sarima.seasonal_order == (1, 1, 1, 4)

    def test_fit_and_predict(self, sample_time_series):
        """Test SARIMA fit and predict."""
        sarima = SARIMAPredictor()
        sarima.fit(sample_time_series, "performance")

        assert sarima.fitted_model is not None

        forecast = sarima.predict(steps_ahead=3)
        assert "forecast" in forecast
        assert len(forecast["forecast"]) == 3

    def test_seasonal_decomposition(self, sample_time_series):
        """Test seasonal information extraction."""
        sarima = SARIMAPredictor()
        sarima.fit(sample_time_series, "performance")

        trend = sarima.get_trend()
        assert "seasonal_period" in trend


class TestLSTMPredictor:
    """Test LSTM predictor."""

    def test_initialization(self):
        """Test LSTM initialization."""
        lstm = LSTMPredictor(sequence_length=10, lstm_units=64)
        assert lstm.sequence_length == 10
        assert lstm.lstm_units == 64
        assert lstm.model is None

    def test_fit_and_predict(self, sample_time_series):
        """Test LSTM fit and predict."""
        lstm = LSTMPredictor(sequence_length=5, lstm_units=32)
        lstm.fit(sample_time_series, "performance")

        assert lstm.model is not None
        assert lstm.scaler is not None

        forecast = lstm.predict(steps_ahead=3)
        assert "forecast" in forecast
        assert len(forecast["forecast"]) == 3

    def test_set_sequence_length(self):
        """Test sequence length setter."""
        lstm = LSTMPredictor(sequence_length=10)
        lstm.set_sequence_length(15)
        assert lstm.sequence_length == 15

    def test_attention_weights(self):
        """Test attention weights (should be None for vanilla LSTM)."""
        lstm = LSTMPredictor()
        assert lstm.get_attention_weights() is None


class TestGRUPredictor:
    """Test GRU predictor."""

    def test_initialization(self):
        """Test GRU initialization."""
        gru = GRUPredictor(sequence_length=10, gru_units=64)
        assert gru.sequence_length == 10
        assert gru.gru_units == 64

    def test_fit_and_predict(self, sample_time_series):
        """Test GRU fit and predict."""
        gru = GRUPredictor(sequence_length=5, gru_units=32)
        gru.fit(sample_time_series, "performance")

        assert gru.model is not None

        forecast = gru.predict(steps_ahead=3)
        assert "forecast" in forecast
        assert len(forecast["forecast"]) == 3


class TestChampionshipPredictor:
    """Test championship predictor."""

    def test_initialization(self):
        """Test championship predictor initialization."""
        predictor = ChampionshipPredictor()
        assert predictor.models == {}

    def test_points_needed_calculation(self):
        """Test points needed calculation."""
        predictor = ChampionshipPredictor()

        result = predictor.estimate_points_needed(
            driver_name="Hamilton",
            current_points=50,
            leader_points=100,
            races_remaining=5,
            max_points_per_race=25,
        )

        assert "gap_to_leader" in result
        assert "races_remaining" in result
        assert "catchup_probability" in result
        assert result["gap_to_leader"] == 50
        assert result["feasible"] is True

    def test_points_trajectory_with_model(self, driver_standings):
        """Test points trajectory prediction."""
        # Create mock models
        models = {}
        for driver in driver_standings["driver_name"]:
            mock_model = type("MockModel", (), {})()
            mock_model.predict = lambda steps_ahead=3: {
                "forecast": np.array([1.5, 1.4, 1.3])
            }
            models[driver] = mock_model

        predictor = ChampionshipPredictor(time_series_models=models)
        trajectory = predictor.predict_points_trajectory("Hamilton", races_ahead=3)

        assert "driver" in trajectory
        assert "total_projected" in trajectory
        assert "average_per_race" in trajectory


class TestMomentumAnalyzer:
    """Test momentum analyzer."""

    def test_initialization(self):
        """Test momentum analyzer initialization."""
        analyzer = MomentumAnalyzer(min_races_for_form=3)
        assert analyzer.min_races_for_form == 3

    def test_recent_form_calculation(self):
        """Test recent form scoring."""
        analyzer = MomentumAnalyzer()
        performance = pd.Series([0.60, 0.65, 0.70, 0.72, 0.75])

        form = analyzer.calculate_recent_form(performance)

        assert "form_score" in form
        assert "trend" in form
        assert "consistency" in form
        assert form["form_score"] is not None

    def test_hot_cold_streak_detection(self):
        """Test streak detection."""
        analyzer = MomentumAnalyzer()
        performance = pd.Series([0.2, 0.3, 0.8, 0.9, 0.85, 0.1, 0.2, 0.15])

        streaks = analyzer.detect_hot_cold_streaks(performance, threshold=0.5, min_streak_length=2)

        assert "current_streak" in streaks
        assert "all_streaks" in streaks

    def test_inflection_point_detection(self):
        """Test inflection point identification."""
        analyzer = MomentumAnalyzer()
        performance = pd.Series([1, 2, 3, 2, 1, 0.5, 0.5, 1, 2, 3])

        inflections = analyzer.identify_performance_inflection_points(performance)

        assert isinstance(inflections, list)

    def test_circuit_type_analysis(self):
        """Test circuit type performance analysis."""
        analyzer = MomentumAnalyzer()
        data = pd.DataFrame(
            {
                "circuit_type": ["street", "oval", "street", "oval", "street"],
                "performance": [0.75, 0.85, 0.78, 0.88, 0.76],
            }
        )

        analysis = analyzer.analyze_circuit_type_performance(data, "circuit_type", "performance")

        assert "street" in analysis
        assert "oval" in analysis
        assert analysis["street"]["n_races"] == 3

    def test_development_rate_estimation(self):
        """Test team development rate."""
        analyzer = MomentumAnalyzer()
        performance = pd.Series([0.5, 0.55, 0.6, 0.68, 0.75, 0.80])

        rate = analyzer.estimate_team_development_rate(performance, window=2)

        assert "average_development_rate" in rate
        assert "recent_rate" in rate
        assert "trend" in rate

    def test_momentum_comparison(self):
        """Test momentum comparison between drivers."""
        analyzer = MomentumAnalyzer()
        driver1 = pd.Series([0.5, 0.6, 0.7, 0.8, 0.85])
        driver2 = pd.Series([0.7, 0.68, 0.65, 0.62, 0.60])

        comparison = analyzer.compare_momentum(driver1, driver2, recent_races=5)

        assert "driver1_momentum" in comparison
        assert "driver2_momentum" in comparison
        assert "leader" in comparison


class TestIntegration:
    """Integration tests for time series system."""

    def test_end_to_end_forecasting(self, sample_time_series):
        """Test complete forecasting pipeline."""
        # Create features
        engineer = TemporalFeatureEngineer()
        features = engineer.create_rolling_statistics(sample_time_series["performance"])

        # Fit ARIMA
        arima = ARIMAPredictor()
        arima.fit(sample_time_series, "performance")

        # Get forecast
        forecast = arima.predict(steps_ahead=5)

        assert len(forecast["forecast"]) == 5
        assert all(np.isfinite(forecast["forecast"]))

    def test_multiple_models_ensemble(self, sample_time_series):
        """Test using multiple time series models together."""
        arima = ARIMAPredictor()
        arima.fit(sample_time_series, "performance")

        sarima = SARIMAPredictor()
        sarima.fit(sample_time_series, "performance")

        arima_forecast = arima.predict(steps_ahead=3)["forecast"]
        sarima_forecast = sarima.predict(steps_ahead=3)["forecast"]

        # Average forecasts
        ensemble_forecast = (arima_forecast + sarima_forecast) / 2

        assert len(ensemble_forecast) == 3
        assert all(np.isfinite(ensemble_forecast))
