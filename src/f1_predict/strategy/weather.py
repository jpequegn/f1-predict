"""Weather-dependent strategy modeling.

This module models strategy changes based on weather conditions including:
- Weather transition predictions
- Compound selection for conditions
- Risk tolerance strategies
"""

from enum import Enum
from typing import Any

import structlog

from f1_predict.strategy.tire_degradation import TireCompound

logger = structlog.get_logger(__name__)


class WeatherCondition(Enum):
    """Weather conditions affecting tire choice."""

    DRY = "dry"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    DRYING = "drying"


class WeatherStrategyModel:
    """Model strategy changes based on weather conditions.

    Different weather conditions require different tire compounds.
    This model predicts weather transitions and recommends tire changes.
    """

    def __init__(self):
        """Initialize weather strategy model."""
        self.compound_weather_map = {
            WeatherCondition.DRY: [
                TireCompound.SOFT,
                TireCompound.MEDIUM,
                TireCompound.HARD,
            ],
            WeatherCondition.LIGHT_RAIN: [TireCompound.INTERMEDIATE],
            WeatherCondition.HEAVY_RAIN: [TireCompound.WET],
            WeatherCondition.DRYING: [TireCompound.INTERMEDIATE, TireCompound.SOFT],
        }

        logger.info("weather_strategy_model_initialized")

    def predict_weather_transitions(
        self,
        current_lap: int,
        race_laps: int,
        weather_forecast: list[tuple[int, WeatherCondition]],
    ) -> list[dict[str, Any]]:
        """Predict weather changes during race.

        Args:
            current_lap: Current race lap
            race_laps: Total race laps
            weather_forecast: List of (lap, condition) predictions

        Returns:
            List of weather transition events with strategic impact

        Raises:
            ValueError: If race_laps < 1 or current_lap < 0
        """
        if race_laps < 1:
            msg = "race_laps must be >= 1"
            raise ValueError(msg)
        if current_lap < 0:
            msg = "current_lap must be >= 0"
            raise ValueError(msg)

        transitions = []

        for i in range(len(weather_forecast) - 1):
            lap1, cond1 = weather_forecast[i]
            lap2, cond2 = weather_forecast[i + 1]

            if cond1 != cond2:
                transitions.append(
                    {
                        "lap": lap2,
                        "from": cond1.value,
                        "to": cond2.value,
                        "strategic_impact": self._assess_transition_impact(
                            cond1, cond2
                        ),
                    }
                )

        logger.info("weather_transitions_predicted", num_transitions=len(transitions))

        return transitions

    def _assess_transition_impact(
        self, from_condition: WeatherCondition, to_condition: WeatherCondition
    ) -> str:
        """Assess strategic impact of weather transition.

        Args:
            from_condition: Starting weather condition
            to_condition: Ending weather condition

        Returns:
            Impact assessment string
        """
        impact_matrix = {
            (WeatherCondition.DRY, WeatherCondition.LIGHT_RAIN): (
                "HIGH - Immediate pit for inters"
            ),
            (WeatherCondition.DRY, WeatherCondition.HEAVY_RAIN): (
                "CRITICAL - Pit for wets ASAP"
            ),
            (WeatherCondition.LIGHT_RAIN, WeatherCondition.DRY): (
                "HIGH - Gamble on slicks timing"
            ),
            (WeatherCondition.HEAVY_RAIN, WeatherCondition.LIGHT_RAIN): (
                "MEDIUM - Consider inters"
            ),
            (WeatherCondition.LIGHT_RAIN, WeatherCondition.HEAVY_RAIN): (
                "HIGH - Switch to wets"
            ),
        }

        return impact_matrix.get((from_condition, to_condition), "LOW")

    def optimize_mixed_conditions_strategy(
        self,
        race_laps: int,
        weather_forecast: list[tuple[int, WeatherCondition]],
        risk_tolerance: str = "medium",
    ) -> dict[str, Any]:
        """Optimize strategy for changing weather conditions.

        Args:
            race_laps: Total race laps
            weather_forecast: Weather predictions throughout race
            risk_tolerance: "conservative", "medium", or "aggressive"

        Returns:
            Strategy with tire changes for weather

        Raises:
            ValueError: If race_laps < 1 or invalid risk_tolerance
        """
        if race_laps < 1:
            msg = "race_laps must be >= 1"
            raise ValueError(msg)

        valid_risk = ["conservative", "medium", "aggressive"]
        if risk_tolerance not in valid_risk:
            msg = f"risk_tolerance must be one of {valid_risk}"
            raise ValueError(msg)

        strategy_stops = []
        transitions = self.predict_weather_transitions(0, race_laps, weather_forecast)

        for transition in transitions:
            # Determine when to switch tires based on risk tolerance
            if risk_tolerance == "aggressive":
                # Switch early to gain advantage
                pit_lap = max(1, transition["lap"] - 2)
            elif risk_tolerance == "conservative":
                # Wait to confirm conditions
                pit_lap = transition["lap"] + 1
            else:
                # Medium: switch on transition
                pit_lap = transition["lap"]

            # Select appropriate compound
            new_condition = WeatherCondition(transition["to"])
            available = self.compound_weather_map[new_condition]

            # Choose most aggressive viable compound for aggressive style
            if risk_tolerance == "aggressive" and len(available) > 1:
                new_compound = available[-1]  # Most aggressive
            else:
                new_compound = available[0]  # Safest choice

            strategy_stops.append(
                {
                    "lap": pit_lap,
                    "compound": new_compound.value,
                    "reason": (
                        f"Weather transition: {transition['from']} → {transition['to']}"
                    ),
                    "risk_level": risk_tolerance,
                }
            )

        logger.info(
            "weather_strategy_optimized",
            num_stops=len(strategy_stops),
            risk=risk_tolerance,
        )

        return {
            "weather_strategy": True,
            "pit_stops": strategy_stops,
            "total_stops": len(strategy_stops),
            "risk_profile": risk_tolerance,
        }

    def get_recommended_compound(self, condition: WeatherCondition) -> TireCompound:
        """Get recommended tire compound for weather condition.

        Args:
            condition: Current weather condition

        Returns:
            Recommended tire compound
        """
        compounds = self.compound_weather_map[condition]
        return compounds[0]  # Return safest/default choice
