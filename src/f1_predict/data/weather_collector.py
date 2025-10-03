"""Weather data collection and enrichment for F1 races."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from f1_predict.api.weather import WeatherAPIClient
from f1_predict.data.external_models import CircuitWeatherHistory, WeatherData


class WeatherDataCollector:
    """Collects weather data for F1 races and builds historical patterns."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the weather data collector.

        Args:
            data_dir: Base directory for storing data files
            api_key: OpenWeatherMap API key (optional, uses env var if not provided)
        """
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.external_dir = self.data_dir / "external" / "weather"
        self.external_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.client = WeatherAPIClient(api_key=api_key)
            self.api_available = True
        except ValueError as e:
            self.logger.warning(f"Weather API not configured: {e}")
            self.api_available = False
            self.client = None

    def collect_race_weather(
        self,
        circuit_id: str,
        season: str,
        round_num: str,
        lat: float,
        lon: float,
        race_date: datetime,
    ) -> Optional[WeatherData]:
        """Collect weather data for a race.

        Args:
            circuit_id: Circuit identifier
            season: Season year
            round_num: Round number
            lat: Circuit latitude
            lon: Circuit longitude
            race_date: Race date and time

        Returns:
            Weather data or None if API unavailable
        """
        if not self.api_available:
            self.logger.warning("Weather API not available, skipping weather collection")
            return None

        try:
            # For historical races, try to get historical data
            timestamp = int(race_date.timestamp())
            weather = self.client.get_historical_weather(
                lat=lat,
                lon=lon,
                timestamp=timestamp,
                circuit_id=circuit_id,
                season=season,
                round_num=round_num,
            )

            self.logger.info(
                f"Collected weather data for {season} round {round_num}: "
                f"{weather.condition.value}, {weather.air_temperature}°C"
            )

            return weather

        except Exception as e:
            self.logger.error(f"Failed to collect weather data: {e}")
            return None

    def collect_session_weather(
        self,
        circuit_id: str,
        season: str,
        round_num: str,
        lat: float,
        lon: float,
        session_dates: dict[str, datetime],
    ) -> dict[str, WeatherData]:
        """Collect weather for all race weekend sessions.

        Args:
            circuit_id: Circuit identifier
            season: Season year
            round_num: Round number
            lat: Circuit latitude
            lon: Circuit longitude
            session_dates: Dictionary mapping session names to dates

        Returns:
            Dictionary mapping session names to weather data
        """
        weather_data = {}

        for session_name, session_date in session_dates.items():
            try:
                weather = self.collect_race_weather(
                    circuit_id=circuit_id,
                    season=season,
                    round_num=round_num,
                    lat=lat,
                    lon=lon,
                    race_date=session_date,
                )

                if weather:
                    weather_data[session_name] = weather

            except Exception as e:
                self.logger.warning(f"Failed to collect weather for {session_name}: {e}")
                continue

        return weather_data

    def build_historical_weather_patterns(
        self,
        circuit_id: str,
        typical_race_month: int,
        historical_data: list[WeatherData],
    ) -> CircuitWeatherHistory:
        """Build historical weather pattern from collected data.

        Args:
            circuit_id: Circuit identifier
            typical_race_month: Month when races typically occur
            historical_data: List of historical weather data points

        Returns:
            Aggregated historical weather patterns
        """
        if not historical_data:
            # Return defaults if no data available
            return CircuitWeatherHistory(
                circuit_id=circuit_id,
                month=typical_race_month,
                average_air_temp=20.0,
                average_track_temp=None,
                average_humidity=50.0,
                rain_probability=0.3,
                average_rainfall_mm=0.0,
                average_wind_speed=10.0,
                years_of_data=0,
            )

        # Calculate averages
        air_temps = [w.air_temperature for w in historical_data]
        track_temps = [w.track_temperature for w in historical_data if w.track_temperature is not None]
        humidities = [w.humidity for w in historical_data]
        rainfalls = [w.precipitation_mm for w in historical_data]
        wind_speeds = [w.wind_speed for w in historical_data]

        rain_count = sum(1 for w in historical_data if w.precipitation_mm > 0)

        return CircuitWeatherHistory(
            circuit_id=circuit_id,
            month=typical_race_month,
            average_air_temp=sum(air_temps) / len(air_temps),
            average_track_temp=sum(track_temps) / len(track_temps) if track_temps else None,
            average_humidity=sum(humidities) / len(humidities),
            rain_probability=rain_count / len(historical_data),
            average_rainfall_mm=sum(rainfalls) / len(rainfalls),
            average_wind_speed=sum(wind_speeds) / len(wind_speeds),
            years_of_data=len(set(w.season for w in historical_data)),
        )

    def save_weather_data(
        self,
        weather_data: list[WeatherData],
        filename: str = "race_weather_2020_2024.json",
    ) -> Path:
        """Save weather data to file.

        Args:
            weather_data: List of weather data to save
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_file = self.external_dir / filename

        # Convert to dict for JSON serialization
        data_dicts = [w.model_dump(mode="json") for w in weather_data]

        with open(output_file, "w") as f:
            json.dump(data_dicts, f, indent=2, default=str)

        self.logger.info(f"Saved {len(weather_data)} weather records to {output_file}")
        return output_file

    def load_weather_data(
        self,
        filename: str = "race_weather_2020_2024.json",
    ) -> list[WeatherData]:
        """Load weather data from file.

        Args:
            filename: Input filename

        Returns:
            List of weather data
        """
        input_file = self.external_dir / filename

        if not input_file.exists():
            self.logger.warning(f"Weather data file not found: {input_file}")
            return []

        with open(input_file) as f:
            data_dicts = json.load(f)

        weather_data = [WeatherData(**d) for d in data_dicts]
        self.logger.info(f"Loaded {len(weather_data)} weather records from {input_file}")
        return weather_data

    def get_weather_summary(self) -> dict:
        """Get summary of collected weather data.

        Returns:
            Dictionary with weather data summary
        """
        summary = {
            "api_available": self.api_available,
            "data_directory": str(self.external_dir),
            "files": [],
        }

        for file_path in self.external_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)

                summary["files"].append({
                    "name": file_path.name,
                    "records": len(data),
                    "size_bytes": file_path.stat().st_size,
                })
            except Exception as e:
                self.logger.warning(f"Failed to read {file_path}: {e}")

        return summary

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.client:
            self.client.close()
