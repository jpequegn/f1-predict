"""Tests for weather-dependent strategy model."""

import pytest

from f1_predict.strategy.tire_degradation import TireCompound
from f1_predict.strategy.weather import WeatherCondition, WeatherStrategyModel


class TestWeatherCondition:
    """Tests for WeatherCondition enum."""

    def test_weather_condition_values(self):
        """Test weather condition enum values."""
        assert WeatherCondition.DRY.value == "dry"
        assert WeatherCondition.LIGHT_RAIN.value == "light_rain"
        assert WeatherCondition.HEAVY_RAIN.value == "heavy_rain"
        assert WeatherCondition.DRYING.value == "drying"


class TestWeatherStrategyModel:
    """Tests for WeatherStrategyModel."""

    def test_initialization(self):
        """Test model initialization."""
        model = WeatherStrategyModel()

        assert model.compound_weather_map is not None
        assert WeatherCondition.DRY in model.compound_weather_map
        assert WeatherCondition.LIGHT_RAIN in model.compound_weather_map

    def test_compound_weather_mapping(self):
        """Test compound mappings for weather conditions."""
        model = WeatherStrategyModel()

        # Dry conditions should have slick compounds
        dry_compounds = model.compound_weather_map[WeatherCondition.DRY]
        assert TireCompound.SOFT in dry_compounds
        assert TireCompound.MEDIUM in dry_compounds
        assert TireCompound.HARD in dry_compounds

        # Rain should have wet tires
        heavy_rain = model.compound_weather_map[WeatherCondition.HEAVY_RAIN]
        assert TireCompound.WET in heavy_rain

        # Light rain should have intermediates
        light_rain = model.compound_weather_map[WeatherCondition.LIGHT_RAIN]
        assert TireCompound.INTERMEDIATE in light_rain

    def test_predict_weather_transitions_basic(self):
        """Test basic weather transition prediction."""
        model = WeatherStrategyModel()

        forecast = [
            (0, WeatherCondition.DRY),
            (20, WeatherCondition.LIGHT_RAIN),
            (40, WeatherCondition.DRY),
        ]

        transitions = model.predict_weather_transitions(0, 50, forecast)

        assert len(transitions) == 2
        assert transitions[0]["lap"] == 20
        assert transitions[0]["from"] == "dry"
        assert transitions[0]["to"] == "light_rain"

    def test_no_transitions_stable_weather(self):
        """Test no transitions with stable weather."""
        model = WeatherStrategyModel()

        forecast = [(0, WeatherCondition.DRY), (50, WeatherCondition.DRY)]

        transitions = model.predict_weather_transitions(0, 50, forecast)

        assert len(transitions) == 0

    def test_transition_impact_assessment(self):
        """Test strategic impact assessment."""
        model = WeatherStrategyModel()

        forecast = [
            (0, WeatherCondition.DRY),
            (20, WeatherCondition.HEAVY_RAIN),
        ]

        transitions = model.predict_weather_transitions(0, 50, forecast)

        assert len(transitions) == 1
        assert "strategic_impact" in transitions[0]
        assert "CRITICAL" in transitions[0]["strategic_impact"]

    def test_dry_to_rain_high_impact(self):
        """Test dry to rain transition has high impact."""
        model = WeatherStrategyModel()

        forecast = [
            (0, WeatherCondition.DRY),
            (25, WeatherCondition.LIGHT_RAIN),
        ]

        transitions = model.predict_weather_transitions(0, 50, forecast)

        assert "HIGH" in transitions[0]["strategic_impact"]

    def test_invalid_race_laps_raises_error(self):
        """Test transitions with invalid race laps."""
        model = WeatherStrategyModel()

        with pytest.raises(ValueError, match="race_laps must be >= 1"):
            model.predict_weather_transitions(0, 0, [])

    def test_invalid_current_lap_raises_error(self):
        """Test transitions with invalid current lap."""
        model = WeatherStrategyModel()

        with pytest.raises(ValueError, match="current_lap must be >= 0"):
            model.predict_weather_transitions(-1, 50, [])

    def test_optimize_mixed_conditions_basic(self):
        """Test basic mixed conditions optimization."""
        model = WeatherStrategyModel()

        forecast = [
            (0, WeatherCondition.DRY),
            (20, WeatherCondition.LIGHT_RAIN),
            (40, WeatherCondition.DRY),
        ]

        strategy = model.optimize_mixed_conditions_strategy(
            race_laps=50, weather_forecast=forecast
        )

        assert "weather_strategy" in strategy
        assert strategy["weather_strategy"] is True
        assert "pit_stops" in strategy
        assert "total_stops" in strategy
        assert "risk_profile" in strategy

    def test_conservative_strategy_waits_for_confirmation(self):
        """Test conservative strategy waits to confirm conditions."""
        model = WeatherStrategyModel()

        forecast = [(0, WeatherCondition.DRY), (20, WeatherCondition.LIGHT_RAIN)]

        strategy = model.optimize_mixed_conditions_strategy(
            race_laps=50, weather_forecast=forecast, risk_tolerance="conservative"
        )

        # Conservative should pit after transition
        stop = strategy["pit_stops"][0]
        assert stop["lap"] >= 20

    def test_aggressive_strategy_pits_early(self):
        """Test aggressive strategy pits early."""
        model = WeatherStrategyModel()

        forecast = [(0, WeatherCondition.DRY), (20, WeatherCondition.LIGHT_RAIN)]

        strategy = model.optimize_mixed_conditions_strategy(
            race_laps=50, weather_forecast=forecast, risk_tolerance="aggressive"
        )

        # Aggressive should pit before transition
        stop = strategy["pit_stops"][0]
        assert stop["lap"] <= 20

    def test_medium_risk_pits_on_transition(self):
        """Test medium risk pits on transition."""
        model = WeatherStrategyModel()

        forecast = [(0, WeatherCondition.DRY), (20, WeatherCondition.LIGHT_RAIN)]

        strategy = model.optimize_mixed_conditions_strategy(
            race_laps=50, weather_forecast=forecast, risk_tolerance="medium"
        )

        # Medium should pit on transition
        stop = strategy["pit_stops"][0]
        assert stop["lap"] == 20

    def test_invalid_risk_tolerance_raises_error(self):
        """Test optimization with invalid risk tolerance."""
        model = WeatherStrategyModel()

        forecast = [(0, WeatherCondition.DRY)]

        with pytest.raises(ValueError, match="risk_tolerance must be one of"):
            model.optimize_mixed_conditions_strategy(
                race_laps=50, weather_forecast=forecast, risk_tolerance="invalid"
            )

    def test_strategy_includes_compound_changes(self):
        """Test strategy includes appropriate compound changes."""
        model = WeatherStrategyModel()

        forecast = [
            (0, WeatherCondition.DRY),
            (20, WeatherCondition.LIGHT_RAIN),
        ]

        strategy = model.optimize_mixed_conditions_strategy(
            race_laps=50, weather_forecast=forecast
        )

        stop = strategy["pit_stops"][0]
        assert "compound" in stop
        assert stop["compound"] == "intermediate"

    def test_strategy_includes_reason(self):
        """Test strategy stops include reasoning."""
        model = WeatherStrategyModel()

        forecast = [
            (0, WeatherCondition.DRY),
            (20, WeatherCondition.LIGHT_RAIN),
        ]

        strategy = model.optimize_mixed_conditions_strategy(
            race_laps=50, weather_forecast=forecast
        )

        stop = strategy["pit_stops"][0]
        assert "reason" in stop
        assert "transition" in stop["reason"].lower()

    def test_multiple_transitions_multiple_stops(self):
        """Test multiple weather transitions create multiple stops."""
        model = WeatherStrategyModel()

        forecast = [
            (0, WeatherCondition.DRY),
            (15, WeatherCondition.LIGHT_RAIN),
            (30, WeatherCondition.HEAVY_RAIN),
            (45, WeatherCondition.DRYING),
        ]

        strategy = model.optimize_mixed_conditions_strategy(
            race_laps=60, weather_forecast=forecast
        )

        # Should have 3 transitions = 3 stops
        assert strategy["total_stops"] == 3

    def test_get_recommended_compound_dry(self):
        """Test compound recommendation for dry conditions."""
        model = WeatherStrategyModel()

        compound = model.get_recommended_compound(WeatherCondition.DRY)

        # Should recommend conservative slick choice
        assert compound in [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD]

    def test_get_recommended_compound_wet(self):
        """Test compound recommendation for heavy rain."""
        model = WeatherStrategyModel()

        compound = model.get_recommended_compound(WeatherCondition.HEAVY_RAIN)

        assert compound == TireCompound.WET

    def test_get_recommended_compound_intermediate(self):
        """Test compound recommendation for light rain."""
        model = WeatherStrategyModel()

        compound = model.get_recommended_compound(WeatherCondition.LIGHT_RAIN)

        assert compound == TireCompound.INTERMEDIATE

    def test_drying_conditions_flexible(self):
        """Test drying conditions allow intermediate or soft."""
        model = WeatherStrategyModel()

        compounds = model.compound_weather_map[WeatherCondition.DRYING]

        # Drying should allow both inters and softs
        assert TireCompound.INTERMEDIATE in compounds
        assert TireCompound.SOFT in compounds

    def test_aggressive_chooses_riskier_compound(self):
        """Test aggressive strategy chooses more aggressive compound."""
        model = WeatherStrategyModel()

        forecast = [(0, WeatherCondition.DRY), (20, WeatherCondition.DRYING)]

        aggressive = model.optimize_mixed_conditions_strategy(
            race_laps=50, weather_forecast=forecast, risk_tolerance="aggressive"
        )

        # Aggressive strategy should have stops with appropriate compounds
        assert aggressive["pit_stops"][0]["compound"] in ["intermediate", "soft"]

    def test_strategy_risk_profile_stored(self):
        """Test risk profile is stored in strategy."""
        model = WeatherStrategyModel()

        forecast = [(0, WeatherCondition.DRY), (20, WeatherCondition.LIGHT_RAIN)]

        strategy = model.optimize_mixed_conditions_strategy(
            race_laps=50, weather_forecast=forecast, risk_tolerance="aggressive"
        )

        assert strategy["risk_profile"] == "aggressive"

    def test_empty_forecast_no_stops(self):
        """Test empty forecast creates no stops."""
        model = WeatherStrategyModel()

        forecast = [(0, WeatherCondition.DRY)]

        strategy = model.optimize_mixed_conditions_strategy(
            race_laps=50, weather_forecast=forecast
        )

        assert strategy["total_stops"] == 0

    def test_rain_to_dry_transition_impact(self):
        """Test rain to dry transition has high impact."""
        model = WeatherStrategyModel()

        forecast = [
            (0, WeatherCondition.LIGHT_RAIN),
            (25, WeatherCondition.DRY),
        ]

        transitions = model.predict_weather_transitions(0, 50, forecast)

        # Transition to dry is strategic gamble
        assert "HIGH" in transitions[0]["strategic_impact"]
        assert "gamble" in transitions[0]["strategic_impact"].lower()
