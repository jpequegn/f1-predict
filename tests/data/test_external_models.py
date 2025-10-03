"""Tests for external data models."""

from datetime import datetime

import pytest

from f1_predict.data.external_models import (
    CircuitWeatherHistory,
    DownforceLevel,
    EnrichedRaceData,
    PitStopStrategy,
    TireCompound,
    TireStintData,
    TrackCharacteristics,
    TrackType,
    WeatherCondition,
    WeatherData,
)


class TestWeatherData:
    """Test WeatherData model."""

    def test_weather_data_creation(self):
        """Test creating weather data instance."""
        weather = WeatherData(
            session_date=datetime(2024, 3, 2, 15, 0),
            circuit_id="bahrain",
            season="2024",
            round="1",
            air_temperature=28.5,
            track_temperature=42.0,
            condition=WeatherCondition.CLEAR,
            precipitation_mm=0.0,
            humidity=35.0,
            wind_speed=12.5,
            wind_direction=180,
            pressure=1013.0,
        )

        assert weather.circuit_id == "bahrain"
        assert weather.air_temperature == 28.5
        assert weather.condition == WeatherCondition.CLEAR
        assert weather.humidity == 35.0

    def test_weather_data_validation(self):
        """Test weather data validation."""
        # Humidity must be 0-100
        with pytest.raises(ValueError):
            WeatherData(
                session_date=datetime.now(),
                circuit_id="test",
                season="2024",
                round="1",
                air_temperature=20.0,
                condition=WeatherCondition.CLEAR,
                precipitation_mm=0.0,
                humidity=150.0,  # Invalid
                wind_speed=10.0,
            )

    def test_weather_condition_enum(self):
        """Test weather condition enum values."""
        assert WeatherCondition.CLEAR.value == "clear"
        assert WeatherCondition.RAIN.value == "rain"
        assert WeatherCondition.HEAVY_RAIN.value == "heavy_rain"


class TestTrackCharacteristics:
    """Test TrackCharacteristics model."""

    def test_track_creation(self):
        """Test creating track characteristics."""
        track = TrackCharacteristics(
            circuit_id="monaco",
            circuit_name="Circuit de Monaco",
            length_km=3.337,
            number_of_corners=19,
            number_of_drs_zones=1,
            track_type=TrackType.STREET,
            downforce_level=DownforceLevel.VERY_HIGH,
            overtaking_difficulty=10,
            surface_roughness=8.0,
            asphalt_age=5,
            grip_level=5,
            average_safety_car_probability=0.75,
            track_limits_severity=10,
            average_lap_time_seconds=72.0,
            top_speed_km_h=290.0,
            power_unit_stress=4,
            brake_stress=6,
            tire_stress=4,
        )

        assert track.circuit_id == "monaco"
        assert track.track_type == TrackType.STREET
        assert track.downforce_level == DownforceLevel.VERY_HIGH
        assert track.overtaking_difficulty == 10

    def test_track_validation(self):
        """Test track characteristics validation."""
        # Overtaking difficulty must be 1-10
        with pytest.raises(ValueError):
            TrackCharacteristics(
                circuit_id="test",
                circuit_name="Test Circuit",
                length_km=5.0,
                number_of_corners=15,
                track_type=TrackType.PERMANENT,
                downforce_level=DownforceLevel.MEDIUM,
                overtaking_difficulty=15,  # Invalid
                average_safety_car_probability=0.3,
            )


class TestTireData:
    """Test tire-related models."""

    def test_tire_compound_enum(self):
        """Test tire compound enum."""
        assert TireCompound.SOFT.value == "soft"
        assert TireCompound.MEDIUM.value == "medium"
        assert TireCompound.HARD.value == "hard"
        assert TireCompound.INTERMEDIATE.value == "intermediate"

    def test_tire_stint_creation(self):
        """Test creating tire stint data."""
        stint = TireStintData(
            session_type="race",
            season="2024",
            round="1",
            driver_id="max_verstappen",
            compound=TireCompound.SOFT,
            stint_number=1,
            starting_lap=1,
            ending_lap=15,
            laps_completed=15,
            average_lap_time=92.5,
            fastest_lap_time=91.2,
            degradation_rate=0.05,
            stint_end_reason="pit_stop",
        )

        assert stint.compound == TireCompound.SOFT
        assert stint.laps_completed == 15
        assert stint.degradation_rate == 0.05

    def test_pit_stop_strategy_creation(self):
        """Test creating pit stop strategy."""
        strategy = PitStopStrategy(
            season="2024",
            round="1",
            driver_id="max_verstappen",
            constructor_id="red_bull",
            total_pit_stops=2,
            planned_stops=2,
            starting_compound=TireCompound.MEDIUM,
            tire_sequence=[TireCompound.MEDIUM, TireCompound.HARD, TireCompound.SOFT],
            pit_stop_laps=[15, 35],
            pit_stop_durations=[2.3, 2.5],
            average_pit_duration=2.4,
            strategy_effectiveness=8.5,
            positions_gained_lost=2,
        )

        assert strategy.total_pit_stops == 2
        assert len(strategy.tire_sequence) == 3
        assert strategy.average_pit_duration == 2.4


class TestCircuitWeatherHistory:
    """Test circuit weather history model."""

    def test_weather_history_creation(self):
        """Test creating weather history."""
        history = CircuitWeatherHistory(
            circuit_id="silverstone",
            month=7,
            average_air_temp=18.5,
            average_track_temp=25.0,
            average_humidity=65.0,
            rain_probability=0.35,
            average_rainfall_mm=2.5,
            average_wind_speed=15.0,
            years_of_data=10,
        )

        assert history.circuit_id == "silverstone"
        assert history.month == 7
        assert history.rain_probability == 0.35
        assert history.years_of_data == 10

    def test_weather_history_validation(self):
        """Test weather history validation."""
        # Month must be 1-12
        with pytest.raises(ValueError):
            CircuitWeatherHistory(
                circuit_id="test",
                month=13,  # Invalid
                average_air_temp=20.0,
                average_humidity=50.0,
                rain_probability=0.3,
                average_wind_speed=10.0,
                years_of_data=5,
            )


class TestEnrichedRaceData:
    """Test enriched race data model."""

    def test_enriched_data_creation(self):
        """Test creating enriched race data."""
        track = TrackCharacteristics(
            circuit_id="monza",
            circuit_name="Autodromo Nazionale di Monza",
            length_km=5.793,
            number_of_corners=11,
            track_type=TrackType.PERMANENT,
            downforce_level=DownforceLevel.LOW,
            overtaking_difficulty=3,
            average_safety_car_probability=0.35,
        )

        weather = WeatherData(
            session_date=datetime(2024, 9, 1, 15, 0),
            circuit_id="monza",
            season="2024",
            round="16",
            air_temperature=25.0,
            condition=WeatherCondition.CLEAR,
            precipitation_mm=0.0,
            humidity=45.0,
            wind_speed=8.0,
        )

        enriched = EnrichedRaceData(
            season="2024",
            round="16",
            circuit_id="monza",
            race_weather=weather,
            track_characteristics=track,
            data_completeness_score=0.8,
        )

        assert enriched.circuit_id == "monza"
        assert enriched.race_weather.air_temperature == 25.0
        assert enriched.track_characteristics.downforce_level == DownforceLevel.LOW
        assert enriched.data_completeness_score == 0.8

    def test_enriched_data_minimal(self):
        """Test creating enriched data with minimal information."""
        enriched = EnrichedRaceData(
            season="2024",
            round="1",
            circuit_id="bahrain",
        )

        assert enriched.race_weather is None
        assert enriched.track_characteristics is None
        assert enriched.data_completeness_score == 0.0
