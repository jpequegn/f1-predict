"""Pydantic models for external F1 data sources.

This module defines data models for external data that impacts race outcomes,
including weather conditions, track characteristics, and tire strategy data.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class WeatherCondition(str, Enum):
    """Weather condition types."""

    CLEAR = "clear"
    PARTLY_CLOUDY = "partly_cloudy"
    CLOUDY = "cloudy"
    OVERCAST = "overcast"
    LIGHT_RAIN = "light_rain"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"
    FOG = "fog"


class TireCompound(str, Enum):
    """F1 tire compound types."""

    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"
    INTERMEDIATE = "intermediate"
    WET = "wet"

    # Historical compounds
    HYPERSOFT = "hypersoft"
    ULTRASOFT = "ultrasoft"
    SUPERSOFT = "supersoft"
    SUPERHARD = "superhard"


class TrackType(str, Enum):
    """Circuit type classification."""

    STREET = "street"
    PERMANENT = "permanent"
    SEMI_PERMANENT = "semi_permanent"


class DownforceLevel(str, Enum):
    """Downforce level required for circuit."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class WeatherData(BaseModel):
    """Weather conditions for a race session."""

    session_date: datetime
    circuit_id: str
    season: str
    round: str

    # Temperature data
    air_temperature: float = Field(..., description="Air temperature in Celsius")
    track_temperature: Optional[float] = Field(None, description="Track surface temperature in Celsius")

    # Precipitation
    condition: WeatherCondition
    precipitation_mm: float = Field(default=0.0, description="Precipitation in mm")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")

    # Wind
    wind_speed: float = Field(..., ge=0, description="Wind speed in km/h")
    wind_direction: Optional[int] = Field(None, ge=0, le=360, description="Wind direction in degrees")

    # Atmospheric pressure
    pressure: Optional[float] = Field(None, description="Atmospheric pressure in hPa")

    # Data source metadata
    source: str = Field(default="openweathermap", description="Weather data source")
    timestamp: datetime = Field(default_factory=datetime.now, description="Data collection timestamp")

    @field_validator("session_date", mode="before")
    @classmethod
    def parse_datetime(cls, v) -> datetime:
        """Parse datetime string to datetime object."""
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class TrackCharacteristics(BaseModel):
    """Physical and technical characteristics of an F1 circuit."""

    circuit_id: str
    circuit_name: str

    # Track layout
    length_km: float = Field(..., gt=0, description="Circuit length in kilometers")
    number_of_corners: int = Field(..., gt=0, description="Total number of corners")
    number_of_drs_zones: int = Field(default=0, ge=0, description="Number of DRS activation zones")

    # Track type and characteristics
    track_type: TrackType
    downforce_level: DownforceLevel
    overtaking_difficulty: int = Field(..., ge=1, le=10, description="Overtaking difficulty (1=easy, 10=very hard)")

    # Surface characteristics
    surface_roughness: Optional[float] = Field(None, ge=0, le=10, description="Surface roughness index")
    asphalt_age: Optional[int] = Field(None, ge=0, description="Years since last resurfacing")
    grip_level: Optional[int] = Field(None, ge=1, le=10, description="Typical grip level (1=low, 10=high)")

    # Safety and limits
    average_safety_car_probability: float = Field(..., ge=0, le=1, description="Historical safety car probability")
    track_limits_severity: int = Field(default=5, ge=1, le=10, description="Track limits enforcement strictness")

    # Performance characteristics
    average_lap_time_seconds: Optional[float] = Field(None, gt=0, description="Average lap time in seconds")
    top_speed_km_h: Optional[float] = Field(None, gt=0, description="Typical top speed in km/h")

    # Power unit stress
    power_unit_stress: int = Field(default=5, ge=1, le=10, description="Power unit stress level (1=low, 10=extreme)")
    brake_stress: int = Field(default=5, ge=1, le=10, description="Brake stress level (1=low, 10=extreme)")
    tire_stress: int = Field(default=5, ge=1, le=10, description="Tire stress level (1=low, 10=extreme)")


class TireStintData(BaseModel):
    """Tire stint information from a race."""

    session_type: str = Field(..., description="Session type (race, qualifying, practice)")
    season: str
    round: str
    driver_id: str

    # Tire information
    compound: TireCompound
    stint_number: int = Field(..., ge=1, description="Stint number in the race")
    starting_lap: int = Field(..., ge=1, description="First lap of the stint")
    ending_lap: int = Field(..., ge=1, description="Last lap of the stint")
    laps_completed: int = Field(..., ge=0, description="Number of laps on this set")

    # Performance
    average_lap_time: Optional[float] = Field(None, description="Average lap time in seconds")
    fastest_lap_time: Optional[float] = Field(None, description="Fastest lap time in seconds")
    degradation_rate: Optional[float] = Field(None, description="Lap time degradation per lap in seconds")

    # Stint outcome
    stint_end_reason: Optional[str] = Field(None, description="Reason for ending stint (pit_stop, race_end, dnf)")

    @field_validator("laps_completed", mode="before")
    @classmethod
    def calculate_laps(cls, v, info) -> int:
        """Calculate laps if not provided."""
        if v is None and "ending_lap" in info.data and "starting_lap" in info.data:
            return info.data["ending_lap"] - info.data["starting_lap"] + 1
        return v


class PitStopStrategy(BaseModel):
    """Pit stop strategy data for a race."""

    season: str
    round: str
    driver_id: str
    constructor_id: str

    # Strategy overview
    total_pit_stops: int = Field(..., ge=0, description="Total number of pit stops")
    planned_stops: int = Field(..., ge=0, description="Planned number of pit stops")

    # Tire strategy
    starting_compound: TireCompound
    tire_sequence: list[TireCompound] = Field(default_factory=list, description="Sequence of tire compounds used")

    # Pit stop performance
    pit_stop_laps: list[int] = Field(default_factory=list, description="Laps on which pit stops occurred")
    pit_stop_durations: list[float] = Field(default_factory=list, description="Pit stop durations in seconds")
    average_pit_duration: Optional[float] = Field(None, description="Average pit stop duration")

    # Strategy outcome
    strategy_effectiveness: Optional[float] = Field(None, ge=0, le=10, description="Strategy effectiveness rating")
    positions_gained_lost: Optional[int] = Field(None, description="Net positions gained/lost due to strategy")


class CircuitWeatherHistory(BaseModel):
    """Historical weather patterns for a circuit."""

    circuit_id: str
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")

    # Historical averages
    average_air_temp: float
    average_track_temp: Optional[float] = None
    average_humidity: float = Field(..., ge=0, le=100)

    # Rain statistics
    rain_probability: float = Field(..., ge=0, le=1, description="Historical probability of rain")
    average_rainfall_mm: float = Field(default=0.0, ge=0)

    # Wind
    average_wind_speed: float = Field(..., ge=0)

    # Sample size
    years_of_data: int = Field(..., gt=0, description="Number of years in historical data")


class EnrichedRaceData(BaseModel):
    """Race data enriched with external sources."""

    # Core race information
    season: str
    round: str
    circuit_id: str

    # Weather data
    race_weather: Optional[WeatherData] = None
    qualifying_weather: Optional[WeatherData] = None
    practice_weather: Optional[list[WeatherData]] = Field(default_factory=list)

    # Track characteristics
    track_characteristics: Optional[TrackCharacteristics] = None

    # Historical weather
    historical_weather: Optional[CircuitWeatherHistory] = None

    # Tire and strategy data
    tire_strategies: list[PitStopStrategy] = Field(default_factory=list)
    tire_stints: list[TireStintData] = Field(default_factory=list)

    # Data completeness
    data_completeness_score: float = Field(default=0.0, ge=0, le=1, description="Percentage of external data available")
