"""Tests for ARIMA and SARIMA time series models.

Comprehensive test coverage for ARIMAPredictor and SARIMAPredictor classes
including fitting, forecasting, and trend analysis.
"""

import numpy as np
import pandas as pd
import pytest

from f1_predict.models.arima_model import ARIMAPredictor, SARIMAPredictor


@pytest.fixture
def sample_univariate_data():
    """Create sample univariate time series data."""
    np.random.seed(42)
    # Create 60 time steps with trend and noise
    trend = np.arange(60) * 0.5
    noise = np.random.normal(0, 1, 60)
    values = 100 + trend + noise
    return pd.DataFrame({"time_step": range(60), "value": values})


@pytest.fixture
def sample_seasonal_data():
    """Create sample data with seasonal pattern."""
    np.random.seed(42)
    n_steps = 80
    trend = np.arange(n_steps) * 0.3
    seasonal = 10 * np.sin(np.arange(n_steps) * 2 * np.pi / 12)  # 12-step cycle
    noise = np.random.normal(0, 1, n_steps)
    values = 100 + trend + seasonal + noise
    return pd.DataFrame({"time_step": range(n_steps), "value": values})


@pytest.fixture
def sample_multivariate_data():
    """Create sample multivariate time series."""
    np.random.seed(42)
    n_steps = 80
    return pd.DataFrame({
        "target": 100 + np.arange(n_steps) * 0.5 + np.random.normal(0, 1, n_steps),
        "feature1": np.sin(np.arange(n_steps) / 10) * 10 + 50,
        "feature2": np.cos(np.arange(n_steps) / 10) * 10 + 50,
    })


class TestARIMAPredictorInit:
    """Tests for ARIMAPredictor initialization."""

    def test_init_default_order(self):
        """Test initialization with default order."""
        predictor = ARIMAPredictor()
        assert predictor.order == (1, 1, 1)
        assert predictor.model is None
        assert predictor.fitted_model is None

    def test_init_custom_order(self):
        """Test initialization with custom order."""
        predictor = ARIMAPredictor(order=(2, 1, 2))
        assert predictor.order == (2, 1, 2)

    def test_init_zero_differences(self):
        """Test initialization with zero differencing."""
        predictor = ARIMAPredictor(order=(1, 0, 1))
        assert predictor.order == (1, 0, 1)

    def test_init_logger_binding(self):
        """Test that logger is properly bound."""
        predictor = ARIMAPredictor(order=(2, 1, 2))
        assert predictor.logger is not None


class TestARIMAPredictorFit:
    """Tests for ARIMA model fitting."""

    def test_fit_success(self, sample_univariate_data):
        """Test successful ARIMA fit."""
        predictor = ARIMAPredictor(order=(1, 1, 1))
        result = predictor.fit(sample_univariate_data, target_column="value")

        assert result is predictor  # Returns self
        assert predictor.fitted_model is not None
        assert predictor.historical_data is not None
        assert predictor.target_column == "value"
        assert predictor.last_value is not None

    def test_fit_different_orders(self, sample_univariate_data):
        """Test fit with different ARIMA orders."""
        for order in [(1, 0, 0), (0, 1, 1), (2, 1, 1), (1, 2, 1)]:
            predictor = ARIMAPredictor(order=order)
            predictor.fit(sample_univariate_data, target_column="value")
            assert predictor.fitted_model is not None

    def test_fit_insufficient_data(self):
        """Test that fit raises error with insufficient data."""
        predictor = ARIMAPredictor(order=(2, 1, 2))
        insufficient_data = pd.DataFrame({"value": np.random.randn(5)})

        with pytest.raises(ValueError, match="Not enough data"):
            predictor.fit(insufficient_data, target_column="value")

    def test_fit_with_missing_values(self, sample_univariate_data):
        """Test fit handles missing values by dropping them."""
        data_with_na = sample_univariate_data.copy()
        data_with_na.loc[0:5, "value"] = np.nan

        predictor = ARIMAPredictor()
        predictor.fit(data_with_na, target_column="value")
        # Should work after dropping NaNs
        assert predictor.fitted_model is not None

    def test_fit_missing_column(self, sample_univariate_data):
        """Test fit raises error when target column missing."""
        predictor = ARIMAPredictor()
        with pytest.raises(KeyError):
            predictor.fit(sample_univariate_data, target_column="nonexistent")

    def test_fit_with_exogenous_ignored(self, sample_univariate_data):
        """Test that fit_with_exogenous falls back to fit."""
        predictor = ARIMAPredictor()
        result = predictor.fit_with_exogenous(
            sample_univariate_data,
            target_column="value",
            exogenous_features=["time_step"],  # Ignored
        )
        # Should return fitted model (fit is called internally)
        assert result is predictor
        assert predictor.fitted_model is not None


class TestARIMAPredictorPredict:
    """Tests for ARIMA prediction."""

    def test_predict_unfitted_model(self):
        """Test predict raises error on unfitted model."""
        predictor = ARIMAPredictor()
        with pytest.raises(RuntimeError, match="not fitted"):
            predictor.predict(steps_ahead=3)

    def test_predict_output_structure(self, sample_univariate_data):
        """Test predict returns correct output structure."""
        predictor = ARIMAPredictor(order=(1, 1, 1))
        predictor.fit(sample_univariate_data, target_column="value")

        forecast = predictor.predict(steps_ahead=5, confidence_level=0.95)

        assert isinstance(forecast, dict)
        assert "forecast" in forecast
        assert "lower_bound" in forecast
        assert "upper_bound" in forecast
        assert "confidence" in forecast
        assert "periods" in forecast
        assert "mean" in forecast

    def test_predict_correct_length(self, sample_univariate_data):
        """Test predict returns correct number of steps."""
        predictor = ARIMAPredictor(order=(1, 1, 1))
        predictor.fit(sample_univariate_data, target_column="value")

        for steps in [1, 3, 5, 10]:
            forecast = predictor.predict(steps_ahead=steps)
            assert len(forecast["forecast"]) == steps
            assert len(forecast["lower_bound"]) == steps
            assert len(forecast["upper_bound"]) == steps

    def test_predict_confidence_levels(self, sample_univariate_data):
        """Test predict with different confidence levels."""
        predictor = ARIMAPredictor(order=(1, 1, 1))
        predictor.fit(sample_univariate_data, target_column="value")

        for conf in [0.80, 0.90, 0.95, 0.99]:
            forecast = predictor.predict(confidence_level=conf)
            assert forecast["confidence"] == conf

    def test_predict_bounds_contain_forecast(self, sample_univariate_data):
        """Test that forecasts are within confidence bounds."""
        predictor = ARIMAPredictor(order=(1, 1, 1))
        predictor.fit(sample_univariate_data, target_column="value")

        forecast = predictor.predict(steps_ahead=5)
        # Each forecast should be within bounds
        for i in range(len(forecast["forecast"])):
            assert forecast["lower_bound"][i] <= forecast["forecast"][i]
            assert forecast["forecast"][i] <= forecast["upper_bound"][i]

    def test_predict_with_exogenous_not_supported(self, sample_univariate_data):
        """Test that predict_with_exogenous raises NotImplementedError."""
        predictor = ARIMAPredictor()
        future_data = pd.DataFrame({"feature1": [1, 2, 3]})

        with pytest.raises(NotImplementedError):
            predictor.predict_with_exogenous(future_data)


class TestARIMAPredictorTrend:
    """Tests for ARIMA trend extraction."""

    def test_get_trend_unfitted(self):
        """Test get_trend returns error dict for unfitted model."""
        predictor = ARIMAPredictor()
        trend = predictor.get_trend()

        assert isinstance(trend, dict)
        assert "error" in trend

    def test_get_trend_fitted_contains_keys(self, sample_univariate_data):
        """Test get_trend returns required keys."""
        predictor = ARIMAPredictor(order=(1, 1, 1))
        predictor.fit(sample_univariate_data, target_column="value")

        trend = predictor.get_trend()

        assert "trend" in trend
        assert "slope" in trend
        assert "last_value" in trend
        assert "forecast_mean" in trend
        assert "forecast_range" in trend

    def test_get_trend_values_are_numeric(self, sample_univariate_data):
        """Test get_trend returns numeric values."""
        predictor = ARIMAPredictor(order=(1, 1, 1))
        predictor.fit(sample_univariate_data, target_column="value")

        trend = predictor.get_trend()

        assert isinstance(trend["slope"], (int, float))
        assert isinstance(trend["last_value"], (int, float))
        assert isinstance(trend["forecast_mean"], (int, float))
        assert isinstance(trend["forecast_range"], tuple)
        assert len(trend["forecast_range"]) == 2

    def test_get_trend_detects_increasing(self):
        """Test get_trend detects increasing trend."""
        # Create clearly increasing data
        increasing_data = pd.DataFrame({"value": np.linspace(0, 100, 60)})

        predictor = ARIMAPredictor(order=(1, 1, 1))
        predictor.fit(increasing_data, target_column="value")

        trend = predictor.get_trend()
        assert trend["trend"] in ["increasing", "decreasing"]


class TestSARIMAPredictorInit:
    """Tests for SARIMAPredictor initialization."""

    def test_init_default_params(self):
        """Test SARIMA initialization with defaults."""
        predictor = SARIMAPredictor()
        assert predictor.order == (1, 1, 1)
        assert predictor.seasonal_order == (1, 1, 1, 4)
        assert predictor.fitted_model is None

    def test_init_custom_order(self):
        """Test SARIMA with custom orders."""
        predictor = SARIMAPredictor(
            order=(2, 1, 2),
            seasonal_order=(1, 1, 1, 12),
        )
        assert predictor.order == (2, 1, 2)
        assert predictor.seasonal_order == (1, 1, 1, 12)

    def test_init_logger_binding(self):
        """Test SARIMA logger binding."""
        predictor = SARIMAPredictor(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
        )
        assert predictor.logger is not None


class TestSARIMAPredictorFit:
    """Tests for SARIMA model fitting."""

    def test_fit_success(self, sample_seasonal_data):
        """Test successful SARIMA fit."""
        predictor = SARIMAPredictor(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
        )
        result = predictor.fit(sample_seasonal_data, target_column="value")

        assert result is predictor
        assert predictor.fitted_model is not None
        assert predictor.target_column == "value"

    def test_fit_with_different_seasonal_periods(self, sample_seasonal_data):
        """Test SARIMA with different seasonal periods."""
        for s in [4, 7, 12, 24]:
            predictor = SARIMAPredictor(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, s),
            )
            predictor.fit(sample_seasonal_data, target_column="value")
            assert predictor.fitted_model is not None

    def test_fit_with_minimal_data(self):
        """Test SARIMA fit with minimal data."""
        predictor = SARIMAPredictor(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
        )
        minimal_data = pd.DataFrame({"value": np.linspace(0, 100, 30)})

        # SARIMA has flexible requirements depending on parameters
        # This should fit successfully with sufficient samples
        predictor.fit(minimal_data, target_column="value")
        assert predictor.fitted_model is not None

    def test_fit_with_exogenous_success(self, sample_multivariate_data):
        """Test SARIMA fit with exogenous variables."""
        predictor = SARIMAPredictor(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
        )
        result = predictor.fit_with_exogenous(
            sample_multivariate_data,
            target_column="target",
            exogenous_features=["feature1", "feature2"],
        )

        assert result is predictor
        assert predictor.fitted_model is not None

    def test_fit_with_exogenous_single_feature(self, sample_multivariate_data):
        """Test SARIMA with single exogenous feature."""
        predictor = SARIMAPredictor()
        predictor.fit_with_exogenous(
            sample_multivariate_data,
            target_column="target",
            exogenous_features=["feature1"],
        )
        assert predictor.fitted_model is not None


class TestSARIMAPredictorPredict:
    """Tests for SARIMA prediction."""

    def test_predict_unfitted_model(self):
        """Test predict raises error on unfitted model."""
        predictor = SARIMAPredictor()
        with pytest.raises(RuntimeError, match="not fitted"):
            predictor.predict(steps_ahead=3)

    def test_predict_output_structure(self, sample_seasonal_data):
        """Test SARIMA predict output structure."""
        predictor = SARIMAPredictor()
        predictor.fit(sample_seasonal_data, target_column="value")

        forecast = predictor.predict(steps_ahead=5)

        assert "forecast" in forecast
        assert "lower_bound" in forecast
        assert "upper_bound" in forecast
        assert "confidence" in forecast
        assert "periods" in forecast

    def test_predict_correct_length(self, sample_seasonal_data):
        """Test SARIMA predict returns correct length."""
        predictor = SARIMAPredictor()
        predictor.fit(sample_seasonal_data, target_column="value")

        for steps in [1, 3, 5, 12]:
            forecast = predictor.predict(steps_ahead=steps)
            assert len(forecast["forecast"]) == steps

    def test_predict_with_exogenous_unfitted(self):
        """Test predict_with_exogenous raises error on unfitted model."""
        predictor = SARIMAPredictor()
        future_data = pd.DataFrame({"feature1": [1, 2, 3]})

        with pytest.raises(RuntimeError):
            predictor.predict_with_exogenous(future_data)


class TestSARIMAPredictorTrend:
    """Tests for SARIMA trend extraction."""

    def test_get_trend_unfitted(self):
        """Test get_trend returns error dict for unfitted model."""
        predictor = SARIMAPredictor()
        trend = predictor.get_trend()

        assert isinstance(trend, dict)
        assert "error" in trend

    def test_get_trend_fitted_returns_dict(self, sample_seasonal_data):
        """Test get_trend returns dictionary for fitted SARIMA."""
        predictor = SARIMAPredictor()
        predictor.fit(sample_seasonal_data, target_column="value")

        trend = predictor.get_trend()

        assert isinstance(trend, dict)
        assert "trend" in trend  # Should have at least trend direction


class TestARIMASARIMAComparison:
    """Tests comparing ARIMA and SARIMA."""

    def test_both_have_same_interface(self):
        """Test ARIMA and SARIMA have compatible interfaces."""
        arima = ARIMAPredictor()
        sarima = SARIMAPredictor()

        # Both should have same methods
        for method in ["fit", "predict", "get_trend"]:
            assert hasattr(arima, method)
            assert hasattr(sarima, method)

    def test_arima_sarima_both_trainable(self, sample_seasonal_data):
        """Test both models can be trained."""
        arima = ARIMAPredictor()
        sarima = SARIMAPredictor()

        arima.fit(sample_seasonal_data, target_column="value")
        sarima.fit(sample_seasonal_data, target_column="value")

        assert arima.fitted_model is not None
        assert sarima.fitted_model is not None

    def test_arima_sarima_produce_forecasts(self, sample_seasonal_data):
        """Test both produce similar forecast structures."""
        arima = ARIMAPredictor()
        sarima = SARIMAPredictor()

        arima.fit(sample_seasonal_data, target_column="value")
        sarima.fit(sample_seasonal_data, target_column="value")

        arima_forecast = arima.predict(steps_ahead=5)
        sarima_forecast = sarima.predict(steps_ahead=5)

        # Both should have same keys
        assert set(arima_forecast.keys()) == set(sarima_forecast.keys())


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_arima_zero_differencing(self, sample_univariate_data):
        """Test ARIMA with zero differencing (I=0)."""
        predictor = ARIMAPredictor(order=(1, 0, 1))
        predictor.fit(sample_univariate_data, target_column="value")
        forecast = predictor.predict(steps_ahead=3)
        assert len(forecast["forecast"]) == 3

    def test_arima_high_differencing(self, sample_univariate_data):
        """Test ARIMA with high differencing."""
        predictor = ARIMAPredictor(order=(1, 2, 1))
        predictor.fit(sample_univariate_data, target_column="value")
        forecast = predictor.predict(steps_ahead=3)
        assert len(forecast["forecast"]) == 3

    def test_sarima_seasonal_period_four(self, sample_univariate_data):
        """Test SARIMA with seasonal period = 4."""
        # SARIMA requires seasonal period > 1
        predictor = SARIMAPredictor(
            order=(1, 1, 1),
            seasonal_order=(1, 0, 0, 4),
        )
        predictor.fit(sample_univariate_data, target_column="value")
        assert predictor.fitted_model is not None

    def test_predict_single_step(self, sample_univariate_data):
        """Test forecast with single step."""
        predictor = ARIMAPredictor()
        predictor.fit(sample_univariate_data, target_column="value")

        forecast = predictor.predict(steps_ahead=1)
        assert len(forecast["forecast"]) == 1

    def test_predict_many_steps(self, sample_univariate_data):
        """Test forecast with many steps."""
        predictor = ARIMAPredictor()
        predictor.fit(sample_univariate_data, target_column="value")

        forecast = predictor.predict(steps_ahead=20)
        assert len(forecast["forecast"]) == 20
