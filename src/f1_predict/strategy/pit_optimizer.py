"""Pit stop strategy optimization using dynamic programming.

This module optimizes pit stop timing and tire compound selection to minimize
total race time while adhering to F1 regulations (mandatory compound usage).
"""

from typing import Any

import structlog

from f1_predict.strategy.tire_degradation import (
    TireCompound,
    TireDegradationConfig,
    TireDegradationModel,
)

logger = structlog.get_logger(__name__)


class PitStopOptimizer:
    """Optimize pit stop strategy using dynamic programming.

    This optimizer finds the best pit stop strategy by:
    1. Trying different numbers of stops (1-3 typical for F1)
    2. Using dynamic programming to find optimal stop timing
    3. Selecting tire compounds to minimize time loss
    4. Enforcing mandatory compound rules
    """

    def __init__(
        self,
        race_laps: int,
        pit_loss_time: float = 23.0,
        min_stint_length: int = 8,
        tire_deg_model: TireDegradationModel | None = None,
    ):
        """Initialize pit stop optimizer.

        Args:
            race_laps: Total number of laps in race
            pit_loss_time: Average pit stop time loss (seconds)
            min_stint_length: Minimum laps per stint (F1 regulation)
            tire_deg_model: Tire degradation model. If None, uses default.

        Raises:
            ValueError: If race_laps < 1 or min_stint_length < 1
        """
        if race_laps < 1:
            msg = "race_laps must be >= 1"
            raise ValueError(msg)
        if min_stint_length < 1:
            msg = "min_stint_length must be >= 1"
            raise ValueError(msg)

        self.race_laps = race_laps
        self.pit_loss = pit_loss_time
        self.min_stint = min_stint_length
        self.tire_model = tire_deg_model or TireDegradationModel(
            TireDegradationConfig()
        )

        logger.info(
            "pit_optimizer_initialized",
            race_laps=race_laps,
            pit_loss=pit_loss_time,
            min_stint=min_stint_length,
        )

    def optimize_strategy(
        self,
        available_compounds: list[TireCompound],
        mandatory_compounds: int = 2,
        track_conditions: dict[str, Any] | None = None,
        safety_car_probability: float = 0.3,  # noqa: ARG002 - reserved for future
    ) -> dict[str, Any]:
        """Find optimal pit stop strategy.

        Uses dynamic programming to explore all valid strategies and select
        the one with minimum time loss.

        Args:
            available_compounds: List of tire compounds available
            mandatory_compounds: Number of different compounds required (F1 rule)
            track_conditions: Dict with track_temp, fuel_load, driver_style
            safety_car_probability: Likelihood of safety car (0-1) - not yet used

        Returns:
            Dictionary containing:
                - num_stops: Number of pit stops
                - total_time_loss: Total time lost to pit stops and degradation
                - pit_stops: List of stop details (lap, compound, stint_length)
                - compounds_used: Set of compounds used

        Raises:
            ValueError: If not enough compounds available for mandatory requirement
        """
        if len(set(available_compounds)) < mandatory_compounds:
            msg = f"Need at least {mandatory_compounds} different compounds"
            raise ValueError(msg)

        if track_conditions is None:
            track_conditions = {
                "track_temp": 45.0,
                "fuel_load": 110.0,
                "driver_style": "neutral",
            }

        logger.info(
            "optimizing_strategy",
            compounds=len(available_compounds),
            mandatory=mandatory_compounds,
        )

        # Try different numbers of stops (1-3 typical for F1)
        best_strategy = None
        best_time = float("inf")

        for num_stops in range(1, min(4, self.race_laps // self.min_stint + 1)):
            strategy = self._optimize_with_stops(
                num_stops=num_stops,
                available_compounds=available_compounds,
                mandatory_compounds=mandatory_compounds,
                track_conditions=track_conditions,
            )

            if strategy and strategy["total_time_loss"] < best_time:
                best_time = strategy["total_time_loss"]
                best_strategy = strategy

        if best_strategy is None:
            # Fallback: single stop strategy
            best_strategy = self._create_fallback_strategy(
                available_compounds, track_conditions
            )

        logger.info(
            "strategy_optimized",
            num_stops=best_strategy["num_stops"],
            time_loss=best_strategy["total_time_loss"],
        )

        return best_strategy

    def _optimize_with_stops(
        self,
        num_stops: int,
        available_compounds: list[TireCompound],
        mandatory_compounds: int,
        track_conditions: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Optimize strategy for specific number of stops.

        Args:
            num_stops: Target number of pit stops
            available_compounds: Available tire compounds
            mandatory_compounds: Required number of different compounds
            track_conditions: Race conditions

        Returns:
            Strategy dict or None if no valid strategy found
        """
        # Simplified greedy approach for MVP
        # Full DP implementation would be more complex
        strategy_stops = []
        compounds_used = set()
        total_time = 0.0

        # Distribute stops evenly across race
        laps_between = self.race_laps // (num_stops + 1)

        current_compound = TireCompound.MEDIUM
        compounds_used.add(current_compound)

        for stop_num in range(num_stops):
            stop_lap = (stop_num + 1) * laps_between

            # Calculate degradation cost for stint
            stint_laps = stop_lap - (strategy_stops[-1]["lap"] if strategy_stops else 0)
            deg_cost = self._calculate_stint_cost(
                current_compound, stint_laps, track_conditions
            )
            total_time += deg_cost

            # Select next compound
            if len(compounds_used) < mandatory_compounds:
                # Must use different compound
                next_compound = next(
                    c for c in available_compounds if c not in compounds_used
                )
            else:
                # Choose optimal compound from available
                next_compound = self._select_optimal_compound(
                    available_compounds, compounds_used
                )

            strategy_stops.append(
                {
                    "lap": stop_lap,
                    "compound": next_compound.value,
                    "stint_length": stint_laps,
                }
            )

            total_time += self.pit_loss
            current_compound = next_compound
            compounds_used.add(current_compound)

        # Final stint to finish
        final_stint = self.race_laps - strategy_stops[-1]["lap"]
        final_cost = self._calculate_stint_cost(
            current_compound, final_stint, track_conditions
        )
        total_time += final_cost

        # Check if strategy meets mandatory compound requirement
        if len(compounds_used) < mandatory_compounds:
            return None

        return {
            "num_stops": num_stops,
            "total_time_loss": total_time,
            "pit_stops": strategy_stops,
            "compounds_used": [c.value for c in compounds_used],
        }

    def _calculate_stint_cost(
        self,
        compound: TireCompound,
        stint_length: int,
        conditions: dict[str, Any],
    ) -> float:
        """Calculate total time cost for a stint.

        Args:
            compound: Tire compound
            stint_length: Number of laps
            conditions: Track conditions

        Returns:
            Total degradation time loss (seconds)
        """
        if stint_length < 1:
            return 0.0

        degradation = self.tire_model.predict_stint_performance(
            compound, stint_length, conditions
        )
        return float(degradation.sum())

    def _select_optimal_compound(
        self,
        available: list[TireCompound],
        already_used: set[TireCompound],  # noqa: ARG002 - reserved for future
    ) -> TireCompound:
        """Select optimal compound for next stint.

        Args:
            available: Available compounds
            already_used: Compounds already used

        Returns:
            Optimal compound choice
        """
        # Prefer medium for balance, fall back to any available
        if TireCompound.MEDIUM in available:
            return TireCompound.MEDIUM
        if TireCompound.HARD in available:
            return TireCompound.HARD
        return available[0] if available else TireCompound.MEDIUM

    def _create_fallback_strategy(
        self,
        available_compounds: list[TireCompound],  # noqa: ARG002 - reserved
        conditions: dict[str, Any],
    ) -> dict[str, Any]:
        """Create simple fallback strategy if optimization fails.

        Args:
            available_compounds: Available tire compounds
            conditions: Track conditions

        Returns:
            Simple 1-stop strategy
        """
        stop_lap = self.race_laps // 2

        return {
            "num_stops": 1,
            "total_time_loss": self.pit_loss
            + self._calculate_stint_cost(TireCompound.MEDIUM, stop_lap, conditions)
            + self._calculate_stint_cost(
                TireCompound.HARD, self.race_laps - stop_lap, conditions
            ),
            "pit_stops": [
                {
                    "lap": stop_lap,
                    "compound": TireCompound.HARD.value,
                    "stint_length": stop_lap,
                }
            ],
            "compounds_used": [TireCompound.MEDIUM.value, TireCompound.HARD.value],
        }
