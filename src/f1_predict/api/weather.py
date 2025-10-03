"""OpenWeatherMap API client for historical and current weather data."""

import os
from datetime import datetime
from typing import Optional

from f1_predict.api.base import BaseAPIClient
from f1_predict.data.external_models import WeatherCondition, WeatherData


class WeatherAPIClient(BaseAPIClient):
    """Client for OpenWeatherMap API to fetch weather data."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize the weather API client.

        Args:
            api_key: OpenWeatherMap API key (defaults to OPENWEATHER_API_KEY env var)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenWeatherMap API key is required. Set OPENWEATHER_API_KEY "
                "environment variable or pass api_key parameter."
            )

        # OpenWeatherMap API base URL
        base_url = "https://api.openweathermap.org/data/2.5"

        super().__init__(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            rate_limit_requests=60,  # OpenWeatherMap free tier: 60 calls/minute
            rate_limit_window=60.0,
        )

        self.logger.info("WeatherAPIClient initialized with API key")

    def _map_weather_condition(self, owm_code: int, description: str) -> WeatherCondition:
        """Map OpenWeatherMap weather code to F1 weather condition.

        Args:
            owm_code: OpenWeatherMap weather condition code
            description: Weather description text

        Returns:
            Mapped weather condition
        """
        # OpenWeatherMap condition codes:
        # 2xx: Thunderstorm
        # 3xx: Drizzle
        # 5xx: Rain
        # 6xx: Snow
        # 7xx: Atmosphere (fog, mist, etc.)
        # 800: Clear
        # 80x: Clouds

        if owm_code == 800:
            return WeatherCondition.CLEAR
        elif 801 <= owm_code <= 802:
            return WeatherCondition.PARTLY_CLOUDY
        elif owm_code == 803:
            return WeatherCondition.CLOUDY
        elif owm_code == 804:
            return WeatherCondition.OVERCAST
        elif 300 <= owm_code < 400 or owm_code == 500:
            return WeatherCondition.LIGHT_RAIN
        elif 501 <= owm_code <= 504 or owm_code == 520:
            return WeatherCondition.RAIN
        elif owm_code >= 521 or (500 <= owm_code < 600 and owm_code not in [500, 501, 502, 503, 504, 520]):
            return WeatherCondition.HEAVY_RAIN
        elif 200 <= owm_code < 300:
            return WeatherCondition.THUNDERSTORM
        elif 700 <= owm_code < 800:
            return WeatherCondition.FOG
        else:
            # Default to cloudy for unknown conditions
            return WeatherCondition.CLOUDY

    def get_current_weather(
        self,
        lat: float,
        lon: float,
        circuit_id: str,
        season: str,
        round_num: str,
    ) -> WeatherData:
        """Get current weather conditions for a location.

        Args:
            lat: Latitude
            lon: Longitude
            circuit_id: Circuit identifier
            season: Season year
            round_num: Round number

        Returns:
            Weather data for the location
        """
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",  # Celsius
        }

        response = self.get("weather", params=params)

        # Extract weather data
        main_data = response.get("main", {})
        weather_info = response.get("weather", [{}])[0]
        wind_data = response.get("wind", {})

        return WeatherData(
            session_date=datetime.now(),
            circuit_id=circuit_id,
            season=season,
            round=round_num,
            air_temperature=main_data.get("temp", 20.0),
            track_temperature=None,  # Not available from OpenWeatherMap
            condition=self._map_weather_condition(
                weather_info.get("id", 800),
                weather_info.get("description", "clear"),
            ),
            precipitation_mm=response.get("rain", {}).get("1h", 0.0),
            humidity=main_data.get("humidity", 50.0),
            wind_speed=wind_data.get("speed", 0.0) * 3.6,  # Convert m/s to km/h
            wind_direction=wind_data.get("deg"),
            pressure=main_data.get("pressure"),
            source="openweathermap",
        )

    def get_historical_weather(
        self,
        lat: float,
        lon: float,
        timestamp: int,
        circuit_id: str,
        season: str,
        round_num: str,
    ) -> WeatherData:
        """Get historical weather data for a specific time.

        Note: This requires a paid OpenWeatherMap subscription.
        For historical data, consider using the One Call API with historical endpoint.

        Args:
            lat: Latitude
            lon: Longitude
            timestamp: Unix timestamp for the date/time
            circuit_id: Circuit identifier
            season: Season year
            round_num: Round number

        Returns:
            Historical weather data
        """
        # This endpoint requires a paid subscription
        # Using the Historical Weather API (formerly Time Machine)
        params = {
            "lat": lat,
            "lon": lon,
            "dt": timestamp,
            "appid": self.api_key,
            "units": "metric",
        }

        try:
            response = self.get("onecall/timemachine", params=params)

            # Extract current conditions from historical data
            current = response.get("current", {})
            weather_info = current.get("weather", [{}])[0]
            wind_speed = current.get("wind_speed", 0.0) * 3.6  # m/s to km/h

            return WeatherData(
                session_date=datetime.fromtimestamp(timestamp),
                circuit_id=circuit_id,
                season=season,
                round=round_num,
                air_temperature=current.get("temp", 20.0),
                track_temperature=None,
                condition=self._map_weather_condition(
                    weather_info.get("id", 800),
                    weather_info.get("description", "clear"),
                ),
                precipitation_mm=current.get("rain", {}).get("1h", 0.0),
                humidity=current.get("humidity", 50.0),
                wind_speed=wind_speed,
                wind_direction=current.get("wind_deg"),
                pressure=current.get("pressure"),
                source="openweathermap_historical",
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to fetch historical weather data: {e}. "
                "Historical weather requires OpenWeatherMap paid subscription."
            )
            # Return a fallback weather data with defaults
            return self._create_fallback_weather(
                datetime.fromtimestamp(timestamp),
                circuit_id,
                season,
                round_num,
            )

    def _create_fallback_weather(
        self,
        session_date: datetime,
        circuit_id: str,
        season: str,
        round_num: str,
    ) -> WeatherData:
        """Create fallback weather data when API is unavailable.

        Args:
            session_date: Session date
            circuit_id: Circuit identifier
            season: Season year
            round_num: Round number

        Returns:
            Default weather data
        """
        return WeatherData(
            session_date=session_date,
            circuit_id=circuit_id,
            season=season,
            round=round_num,
            air_temperature=20.0,  # Reasonable default
            track_temperature=None,
            condition=WeatherCondition.PARTLY_CLOUDY,
            precipitation_mm=0.0,
            humidity=50.0,
            wind_speed=10.0,
            wind_direction=None,
            pressure=1013.0,
            source="fallback",
        )

    def get_forecast(
        self,
        lat: float,
        lon: float,
        circuit_id: str,
        season: str,
        round_num: str,
    ) -> list[WeatherData]:
        """Get 5-day weather forecast.

        Args:
            lat: Latitude
            lon: Longitude
            circuit_id: Circuit identifier
            season: Season year
            round_num: Round number

        Returns:
            List of weather forecasts
        """
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
        }

        response = self.get("forecast", params=params)

        forecasts = []
        for item in response.get("list", []):
            main_data = item.get("main", {})
            weather_info = item.get("weather", [{}])[0]
            wind_data = item.get("wind", {})

            forecast = WeatherData(
                session_date=datetime.fromtimestamp(item.get("dt", 0)),
                circuit_id=circuit_id,
                season=season,
                round=round_num,
                air_temperature=main_data.get("temp", 20.0),
                track_temperature=None,
                condition=self._map_weather_condition(
                    weather_info.get("id", 800),
                    weather_info.get("description", "clear"),
                ),
                precipitation_mm=item.get("rain", {}).get("3h", 0.0),
                humidity=main_data.get("humidity", 50.0),
                wind_speed=wind_data.get("speed", 0.0) * 3.6,
                wind_direction=wind_data.get("deg"),
                pressure=main_data.get("pressure"),
                source="openweathermap_forecast",
            )
            forecasts.append(forecast)

        return forecasts
