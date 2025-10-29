"""ARIMA/SARIMA time series models for F1 performance forecasting.

ARIMA (AutoRegressive Integrated Moving Average) and SARIMA (Seasonal ARIMA)
models for forecasting driver performance trends and championship positions.
"""

from typing import Any

import numpy as np
import pandas as pd
import structlog
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from f1_predict.models.time_series_base import MultiVariateTimeSeriesPredictor

logger = structlog.get_logger(__name__)


class ARIMAPredictor(MultiVariateTimeSeriesPredictor):
    """ARIMA time series forecaster for F1 driver/team performance.

    ARIMA(p,d,q) where:
    - p: Number of autoregressive lags
    - d: Degree of differencing for stationarity
    - q: Number of moving average terms
    """

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1)):
        """Initialize ARIMA predictor.

        Args:
            order: (p, d, q) parameters for ARIMA model
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.historical_data = None
        self.target_column = None
        self.last_value = None
        self.logger = logger.bind(component="arima", order=order)

    def fit(
        self, historical_data: pd.DataFrame, target_column: str
    ) -> "ARIMAPredictor":
        """Fit ARIMA model on historical time series.

        Args:
            historical_data: DataFrame with time-ordered observations
            target_column: Column name to forecast

        Returns:
            Fitted ARIMA predictor
        """
        try:
            series = historical_data[target_column].dropna()

            if len(series) < max(self.order) + 5:
                msg = f"Not enough data: need {max(self.order) + 5}, got {len(series)}"
                raise ValueError(msg)

            self.model = ARIMA(series, order=self.order)
            self.fitted_model = self.model.fit()
            self.historical_data = series
            self.target_column = target_column
            self.last_value = series.iloc[-1]

            self.logger.info(
                "arima_fitted",
                n_obs=len(series),
                aic=float(self.fitted_model.aic),
                bic=float(self.fitted_model.bic),
            )
            return self

        except Exception as e:
            self.logger.error("arima_fit_failed", error=str(e))
            raise

    def fit_with_exogenous(
        self,
        historical_data: pd.DataFrame,
        target_column: str,
        exogenous_features: list[str],
    ) -> "ARIMAPredictor":
        """Fit ARIMA with external variables (not typically used).

        For exogenous variables, SARIMAX is recommended instead.

        Args:
            historical_data: DataFrame with observations
            target_column: Column to forecast
            exogenous_features: External feature columns

        Returns:
            Fitted predictor
        """
        # ARIMA doesn't support exogenous, use SARIMAX instead
        self.logger.warning("arima_no_exog", use_sarimax=True)
        return self.fit(historical_data, target_column)

    def predict(
        self, steps_ahead: int = 3, confidence_level: float = 0.95
    ) -> dict[str, Any]:
        """Generate ARIMA forecasts with confidence intervals.

        Args:
            steps_ahead: Number of periods to forecast
            confidence_level: Confidence level for intervals (0.95 = 95% CI)

        Returns:
            Dictionary with forecast, lower/upper bounds, confidence
        """
        if self.fitted_model is None:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        try:
            alpha = 1 - confidence_level
            forecast_result = self.fitted_model.get_forecast(steps=steps_ahead)
            forecast_mean = forecast_result.predicted_mean.values
            conf_df = forecast_result.conf_int(alpha=alpha)

            return {
                "forecast": forecast_mean,
                "lower_bound": conf_df.iloc[:, 0].values,
                "upper_bound": conf_df.iloc[:, 1].values,
                "confidence": confidence_level,
                "periods": steps_ahead,
                "mean": float(forecast_mean.mean()),
            }

        except Exception as e:
            self.logger.error("arima_predict_failed", error=str(e))
            raise

    def predict_with_exogenous(
        self,
        future_exogenous: pd.DataFrame,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """ARIMA doesn't support exogenous prediction."""
        msg = "ARIMA doesn't support exogenous variables. Use SARIMAX instead."
        raise NotImplementedError(msg)

    def get_trend(self) -> dict[str, Any]:
        """Extract trend information from ARIMA model.

        Returns:
            Dictionary with trend direction, strength, and components
        """
        if self.fitted_model is None:
            return {"error": "Model not fitted"}

        # Get recent forecast to determine trend
        forecast = self.fitted_model.get_forecast(steps=5).predicted_mean
        trend_direction = "increasing" if forecast.iloc[-1] > self.last_value else "decreasing"

        # Calculate trend strength
        x = np.arange(len(forecast))
        y = forecast.values
        slope = np.polyfit(x, y, 1)[0]

        return {
            "trend": trend_direction,
            "slope": float(slope),
            "last_value": float(self.last_value),
            "forecast_mean": float(forecast.mean()),
            "forecast_range": (float(forecast.min()), float(forecast.max())),
        }


class SARIMAPredictor(MultiVariateTimeSeriesPredictor):
    """SARIMA time series forecaster with seasonal components.

    SARIMA(p,d,q)(P,D,Q,s) for seasonal time series patterns.
    Useful for F1 with periodic patterns within seasons.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 4),
    ):
        """Initialize SARIMA predictor.

        Args:
            order: (p, d, q) for non-seasonal component
            seasonal_order: (P, D, Q, s) for seasonal component, s=season length
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.historical_data = None
        self.target_column = None
        self.logger = logger.bind(
            component="sarima", order=order, seasonal_order=seasonal_order
        )

    def fit(
        self, historical_data: pd.DataFrame, target_column: str
    ) -> "SARIMAPredictor":
        """Fit SARIMA model with seasonal components.

        Args:
            historical_data: Time-ordered DataFrame
            target_column: Column to forecast

        Returns:
            Fitted SARIMA predictor
        """
        try:
            series = historical_data[target_column].dropna()
            min_required = max(self.order) + max(self.seasonal_order[:3]) * self.seasonal_order[3]

            if len(series) < min_required + 5:
                msg = f"Not enough data: need {min_required + 5}, got {len(series)}"
                raise ValueError(msg)

            self.model = SARIMAX(
                series, order=self.order, seasonal_order=self.seasonal_order
            )
            self.fitted_model = self.model.fit(disp=False)
            self.historical_data = series
            self.target_column = target_column

            self.logger.info(
                "sarima_fitted",
                n_obs=len(series),
                aic=float(self.fitted_model.aic),
            )
            return self

        except Exception as e:
            self.logger.error("sarima_fit_failed", error=str(e))
            raise

    def fit_with_exogenous(
        self,
        historical_data: pd.DataFrame,
        target_column: str,
        exogenous_features: list[str],
    ) -> "SARIMAPredictor":
        """Fit SARIMA with exogenous variables.

        Args:
            historical_data: DataFrame with observations
            target_column: Column to forecast
            exogenous_features: External feature columns

        Returns:
            Fitted predictor
        """
        try:
            endog = historical_data[target_column].dropna()
            exog = historical_data[exogenous_features].loc[endog.index]

            self.model = SARIMAX(
                endog, exog=exog, order=self.order, seasonal_order=self.seasonal_order
            )
            self.fitted_model = self.model.fit(disp=False)
            self.historical_data = endog
            self.target_column = target_column

            self.logger.info("sarima_fitted_with_exog", n_features=len(exogenous_features))
            return self

        except Exception as e:
            self.logger.error("sarima_fit_with_exog_failed", error=str(e))
            raise

    def predict(
        self, steps_ahead: int = 3, confidence_level: float = 0.95
    ) -> dict[str, Any]:
        """Generate SARIMA forecasts.

        Args:
            steps_ahead: Periods to forecast
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with forecast and bounds
        """
        if self.fitted_model is None:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        try:
            alpha = 1 - confidence_level
            forecast_result = self.fitted_model.get_forecast(steps=steps_ahead)
            forecast_mean = forecast_result.predicted_mean.values
            conf_df = forecast_result.conf_int(alpha=alpha)

            return {
                "forecast": forecast_mean,
                "lower_bound": conf_df.iloc[:, 0].values,
                "upper_bound": conf_df.iloc[:, 1].values,
                "confidence": confidence_level,
                "periods": steps_ahead,
                "mean": float(forecast_mean.mean()),
            }

        except Exception as e:
            self.logger.error("sarima_predict_failed", error=str(e))
            raise

    def predict_with_exogenous(
        self,
        future_exogenous: pd.DataFrame,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """Generate SARIMA forecasts with future exogenous variables.

        Args:
            future_exogenous: Future external variables
            confidence_level: Confidence level

        Returns:
            Forecast dictionary
        """
        if self.fitted_model is None:
            msg = "Model not fitted."
            raise RuntimeError(msg)

        try:
            alpha = 1 - confidence_level
            steps = len(future_exogenous)
            forecast_result = self.fitted_model.get_forecast(
                steps=steps, exog=future_exogenous
            )
            forecast_df = forecast_result.conf_int(alpha=alpha)

            return {
                "forecast": forecast_df.iloc[:, 0].values,
                "lower_bound": forecast_df.iloc[:, 1].values,
                "upper_bound": forecast_df.iloc[:, 2].values,
                "confidence": confidence_level,
                "periods": steps,
            }

        except Exception as e:
            self.logger.error("sarima_predict_exog_failed", error=str(e))
            raise

    def get_trend(self) -> dict[str, Any]:
        """Extract trend and seasonality information.

        Returns:
            Dictionary with decomposition results
        """
        if self.fitted_model is None:
            return {"error": "Model not fitted"}

        # Get forecast
        forecast = self.fitted_model.get_forecast(steps=5).predicted_mean

        return {
            "trend": "increasing" if forecast.iloc[-1] > self.historical_data.iloc[-1] else "decreasing",
            "seasonal_period": self.seasonal_order[3],
            "forecast_mean": float(forecast.mean()),
            "aic": float(self.fitted_model.aic),
            "bic": float(self.fitted_model.bic),
        }
