"""Tests for weather API client."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from f1_predict.api.weather import WeatherAPIClient
from f1_predict.data.external_models import WeatherCondition


class TestWeatherAPIClient:
    """Test WeatherAPIClient."""

    @patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test_key"})
    def test_client_initialization(self):
        """Test client initialization with API key."""
        client = WeatherAPIClient()
        assert client.api_key == "test_key"
        assert client.base_url == "https://api.openweathermap.org/data/2.5"

    def test_client_initialization_no_key(self):
        """Test client initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                WeatherAPIClient()

    @patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test_key"})
    def test_map_weather_condition_clear(self):
        """Test mapping clear weather condition."""
        client = WeatherAPIClient()

        # Code 800 = clear sky
        condition = client._map_weather_condition(800, "clear sky")
        assert condition == WeatherCondition.CLEAR

    @patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test_key"})
    def test_map_weather_condition_rain(self):
        """Test mapping rain conditions."""
        client = WeatherAPIClient()

        # Light rain
        assert client._map_weather_condition(500, "light rain") == WeatherCondition.LIGHT_RAIN

        # Moderate rain
        assert client._map_weather_condition(501, "moderate rain") == WeatherCondition.RAIN

        # Heavy rain
        assert client._map_weather_condition(502, "heavy rain") == WeatherCondition.RAIN

    @patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test_key"})
    def test_map_weather_condition_clouds(self):
        """Test mapping cloud conditions."""
        client = WeatherAPIClient()

        assert client._map_weather_condition(801, "few clouds") == WeatherCondition.PARTLY_CLOUDY
        assert client._map_weather_condition(803, "broken clouds") == WeatherCondition.CLOUDY
        assert client._map_weather_condition(804, "overcast clouds") == WeatherCondition.OVERCAST

    @patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test_key"})
    def test_get_current_weather(self):
        """Test getting current weather."""
        client = WeatherAPIClient()

        # Mock API response
        mock_response = {
            "main": {
                "temp": 25.5,
                "humidity": 60.0,
                "pressure": 1013.0,
            },
            "weather": [{"id": 800, "description": "clear sky"}],
            "wind": {"speed": 3.5, "deg": 180},
        }

        with patch.object(client, "get", return_value=mock_response):
            weather = client.get_current_weather(
                lat=26.0325,
                lon=50.5106,
                circuit_id="bahrain",
                season="2024",
                round_num="1",
            )

            assert weather.air_temperature == 25.5
            assert weather.humidity == 60.0
            assert weather.condition == WeatherCondition.CLEAR
            assert weather.wind_speed == 3.5 * 3.6  # Converted to km/h
            assert weather.circuit_id == "bahrain"

    @patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test_key"})
    def test_get_historical_weather_fallback(self):
        """Test historical weather with API error fallback."""
        client = WeatherAPIClient()

        # Mock API error
        with patch.object(client, "get", side_effect=Exception("API error")):
            weather = client.get_historical_weather(
                lat=26.0325,
                lon=50.5106,
                timestamp=int(datetime(2024, 3, 2).timestamp()),
                circuit_id="bahrain",
                season="2024",
                round_num="1",
            )

            # Should return fallback data
            assert weather.source == "fallback"
            assert weather.air_temperature == 20.0  # Default
            assert weather.condition == WeatherCondition.PARTLY_CLOUDY

    @patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test_key"})
    def test_get_forecast(self):
        """Test getting weather forecast."""
        client = WeatherAPIClient()

        # Mock forecast response
        mock_response = {
            "list": [
                {
                    "dt": int(datetime(2024, 3, 2, 15, 0).timestamp()),
                    "main": {"temp": 26.0, "humidity": 55.0, "pressure": 1012.0},
                    "weather": [{"id": 801, "description": "few clouds"}],
                    "wind": {"speed": 4.0, "deg": 90},
                },
                {
                    "dt": int(datetime(2024, 3, 2, 18, 0).timestamp()),
                    "main": {"temp": 24.0, "humidity": 58.0, "pressure": 1013.0},
                    "weather": [{"id": 800, "description": "clear sky"}],
                    "wind": {"speed": 3.0, "deg": 100},
                },
            ]
        }

        with patch.object(client, "get", return_value=mock_response):
            forecasts = client.get_forecast(
                lat=26.0325,
                lon=50.5106,
                circuit_id="bahrain",
                season="2024",
                round_num="1",
            )

            assert len(forecasts) == 2
            assert forecasts[0].air_temperature == 26.0
            assert forecasts[0].condition == WeatherCondition.PARTLY_CLOUDY
            assert forecasts[1].air_temperature == 24.0
            assert forecasts[1].condition == WeatherCondition.CLEAR

    @patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test_key"})
    def test_create_fallback_weather(self):
        """Test creating fallback weather data."""
        client = WeatherAPIClient()

        weather = client._create_fallback_weather(
            session_date=datetime(2024, 3, 2, 15, 0),
            circuit_id="test",
            season="2024",
            round_num="1",
        )

        assert weather.source == "fallback"
        assert weather.air_temperature == 20.0
        assert weather.humidity == 50.0
        assert weather.condition == WeatherCondition.PARTLY_CLOUDY
