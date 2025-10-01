"""Safety car impact modeling for race strategy.

This module models safety car probability and strategic impact including:
- Probabilistic safety car window predictions
- Pit advantage calculations under safety car
- Strategy adjustment recommendations
"""

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Constants for pit advantage calculations
OLD_TIRE_THRESHOLD = 15  # laps
FRESH_TIRE_THRESHOLD = 5  # laps
MIN_ADVANTAGE_TO_PIT = 10  # seconds
MIN_BENEFIT_FOR_ADJUSTMENT = 5  # seconds


class SafetyCarModel:
    """Model safety car probability and strategic impact.

    Safety cars bunch up the field, making pit stops much less costly.
    This model predicts likely SC periods and calculates the advantage
    of pitting during safety car vs. normal racing conditions.
    """

    def __init__(self, circuit_safety_car_rate: float = 0.3):
        """Initialize safety car model.

        Args:
            circuit_safety_car_rate: Historical SC probability for circuit (0-1)

        Raises:
            ValueError: If circuit_safety_car_rate not in [0, 1]
        """
        if not 0 <= circuit_safety_car_rate <= 1:
            msg = "circuit_safety_car_rate must be between 0 and 1"
            raise ValueError(msg)

        self.base_probability = circuit_safety_car_rate

        logger.info(
            "safety_car_model_initialized", base_probability=circuit_safety_car_rate
        )

    def predict_safety_car_windows(
        self, race_laps: int, incidents_so_far: int = 0
    ) -> list[tuple[range, float]]:
        """Predict likely safety car periods.

        Safety cars are more likely during:
        - Early laps (1-5): First lap chaos
        - Mid-race: Tire failures and incidents
        - Late race (last 10 laps): Desperate overtaking

        Args:
            race_laps: Total race laps
            incidents_so_far: Number of incidents that have occurred

        Returns:
            List of (lap_range, probability) tuples

        Raises:
            ValueError: If race_laps < 1
        """
        if race_laps < 1:
            msg = "race_laps must be >= 1"
            raise ValueError(msg)

        windows = []

        # Lap 1-5: First lap incidents (higher with more incidents)
        windows.append((range(1, 6), 0.15 * (1 + incidents_so_far * 0.1)))

        # Laps 10-15: Early pit stop chaos
        windows.append((range(10, 16), 0.08))

        # Laps 20-40 (or middle third): Mid-race incidents
        mid_race_start = race_laps // 3
        mid_race_end = 2 * race_laps // 3
        windows.append((range(mid_race_start, mid_race_end), 0.12))

        # Last 10 laps: Desperate moves
        windows.append((range(max(1, race_laps - 10), race_laps + 1), 0.10))

        return windows

    def calculate_pit_under_sc_advantage(
        self,
        current_lap: int,
        tire_age: int,
        position: int,
        field_spread: float = 15.0,
    ) -> dict[str, Any]:
        """Calculate advantage of pitting under safety car.

        Under safety car, the field bunches up, making pit stops much less costly.
        Normal pit stop loses ~23s, but under SC it's effectively free.

        Args:
            current_lap: Current race lap
            tire_age: Laps on current tires
            position: Current race position (1-20)
            field_spread: Average time gap between positions (seconds)

        Returns:
            Dict with:
                - time_advantage: Time saved by pitting under SC (seconds)
                - positions_lost_estimate: Estimated positions lost
                - recommendation: "PIT" or "STAY OUT"

        Raises:
            ValueError: If tire_age < 0 or position < 1
        """
        if tire_age < 0:
            msg = "tire_age must be >= 0"
            raise ValueError(msg)
        if position < 1:
            msg = "position must be >= 1"
            raise ValueError(msg)

        normal_pit_loss = 23.0  # seconds
        sc_pit_loss = 5.0  # Only lose positions to those who don't pit

        advantage = normal_pit_loss - sc_pit_loss  # ~18s saved

        # Adjust for tire age
        if tire_age > OLD_TIRE_THRESHOLD:
            # Old tires: high advantage to pit
            advantage *= 1.5
        elif tire_age < FRESH_TIRE_THRESHOLD:
            # Fresh tires: low advantage
            advantage *= 0.5

        # Position factor: leaders have less to lose
        position_factor = 1 - (position / 20) * 0.3
        advantage *= position_factor

        positions_lost = max(0, int(advantage / field_spread))

        recommendation = "PIT" if advantage > MIN_ADVANTAGE_TO_PIT else "STAY OUT"

        logger.info(
            "sc_advantage_calculated",
            lap=current_lap,
            tire_age=tire_age,
            position=position,
            advantage=advantage,
            recommendation=recommendation,
        )

        return {
            "time_advantage": advantage,
            "positions_lost_estimate": positions_lost,
            "recommendation": recommendation,
        }

    def adjust_strategy_for_sc_probability(
        self, base_strategy: dict[str, Any], sc_windows: list[tuple[range, float]]
    ) -> dict[str, Any]:
        """Adjust pit strategy based on SC likelihood.

        High SC probability → delay pit stops to target SC windows
        Low SC probability → stick with optimal base strategy

        Args:
            base_strategy: Original optimal strategy
            sc_windows: List of (lap_range, probability) for SC periods

        Returns:
            Adjusted strategy with SC-optimized stop timing
        """
        adjusted_stops = []

        for stop in base_strategy.get("pit_stops", []):
            original_lap = stop["lap"]

            # Find if stop falls near high-probability SC window
            best_adjustment = 0
            max_benefit = 0.0

            for window, probability in sc_windows:
                if original_lap in range(min(window) - 5, max(window) + 5):
                    # Potential to adjust stop timing
                    benefit = probability * 18.0  # Expected time save

                    if benefit > max_benefit:
                        max_benefit = benefit
                        # Suggest stopping in middle of window
                        window_mid = (min(window) + max(window)) // 2
                        best_adjustment = window_mid - original_lap

            adjusted_lap = (
                original_lap + best_adjustment
                if max_benefit > MIN_BENEFIT_FOR_ADJUSTMENT
                else original_lap
            )

            adjusted_stops.append(
                {
                    **stop,
                    "adjusted_lap": adjusted_lap,
                    "sc_benefit": max_benefit,
                }
            )

        return {
            **base_strategy,
            "pit_stops": adjusted_stops,
            "sc_adjusted": True,
        }
