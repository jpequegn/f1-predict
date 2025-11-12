"""Pit stop strategy and tire management for simulations."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from f1_predict.simulation.core.driver_state import TireCompound


class TireStrategy(str, Enum):
    """Tire strategy options."""

    ONE_STOP = "one_stop"
    TWO_STOP = "two_stop"
    THREE_STOP = "three_stop"
    NO_STOP = "no_stop"


@dataclass
class PitStopWindow:
    """Window for executing pit stop.

    Attributes:
        start_lap: Earliest lap to pit
        end_lap: Latest lap to pit
        recommended_lap: Recommended lap for pit stop
    """

    start_lap: int
    end_lap: int
    recommended_lap: int


class PitStopOptimizer:
    """Optimize pit stop timing and tire strategies.

    Calculates optimal pit stop windows and tire compound selections
    based on race duration, tire characteristics, and refueling strategy.
    """

    # Tire degradation rates (time loss per lap, in percent)
    TIRE_DEGRADATION = {
        TireCompound.SOFT: 0.15,  # Highest degradation
        TireCompound.MEDIUM: 0.10,
        TireCompound.HARD: 0.08,
        TireCompound.INTERMEDIATE: 0.05,
        TireCompound.WET: 0.03,
    }

    # Pit stop duration (seconds) including tire change
    PIT_STOP_DURATION = 25.0  # Typical pit stop

    # Fuel consumption per lap (as percentage of tank)
    FUEL_CONSUMPTION_PER_LAP = 1.0

    def __init__(
        self,
        total_laps: int,
        avg_lap_time: float,
        fuel_capacity_laps: int = 60,
    ) -> None:
        """Initialize pit stop optimizer.

        Args:
            total_laps: Total laps in race
            avg_lap_time: Average lap time in seconds
            fuel_capacity_laps: Laps for full tank
        """
        self.total_laps = total_laps
        self.avg_lap_time = avg_lap_time
        self.fuel_capacity_laps = fuel_capacity_laps

    def optimize_strategy(self, fuel_available: float = 100.0) -> TireStrategy:
        """Determine optimal tire strategy.

        Args:
            fuel_available: Available fuel as percentage of tank

        Returns:
            Recommended TireStrategy
        """
        laps_possible = fuel_available * self.fuel_capacity_laps / 100.0

        if laps_possible < self.total_laps * 0.6:
            # Need multiple stops
            return TireStrategy.THREE_STOP

        elif laps_possible < self.total_laps * 0.8:
            # Need two stops
            return TireStrategy.TWO_STOP

        elif laps_possible < self.total_laps:
            # Need one stop
            return TireStrategy.ONE_STOP

        else:
            # Can potentially do full race
            return TireStrategy.ONE_STOP

    def calculate_pit_windows(
        self,
        strategy: TireStrategy,
    ) -> list[PitStopWindow]:
        """Calculate pit stop windows for given strategy.

        Args:
            strategy: Tire strategy to use

        Returns:
            List of PitStopWindow objects
        """
        windows = []

        if strategy == TireStrategy.NO_STOP:
            return windows

        elif strategy == TireStrategy.ONE_STOP:
            # Single stop around lap 30-40
            start = max(1, int(self.total_laps * 0.45))
            end = min(self.total_laps - 5, int(self.total_laps * 0.65))
            recommended = (start + end) // 2
            windows.append(PitStopWindow(start, end, recommended))

        elif strategy == TireStrategy.TWO_STOP:
            # Two stops: lap 15-25 and 35-45
            start1 = max(1, int(self.total_laps * 0.25))
            end1 = int(self.total_laps * 0.35)
            windows.append(PitStopWindow(start1, end1, (start1 + end1) // 2))

            start2 = int(self.total_laps * 0.55)
            end2 = min(self.total_laps - 5, int(self.total_laps * 0.75))
            windows.append(PitStopWindow(start2, end2, (start2 + end2) // 2))

        elif strategy == TireStrategy.THREE_STOP:
            # Three stops: lap 10-18, 25-35, 40-50
            start1 = max(1, int(self.total_laps * 0.15))
            end1 = int(self.total_laps * 0.25)
            windows.append(PitStopWindow(start1, end1, (start1 + end1) // 2))

            start2 = int(self.total_laps * 0.35)
            end2 = int(self.total_laps * 0.5)
            windows.append(PitStopWindow(start2, end2, (start2 + end2) // 2))

            start3 = int(self.total_laps * 0.6)
            end3 = min(self.total_laps - 3, int(self.total_laps * 0.75))
            windows.append(PitStopWindow(start3, end3, (start3 + end3) // 2))

        return windows

    def select_tire_compound(
        self,
        current_lap: int,
        remaining_laps: int,
        weather: str = "dry",
    ) -> TireCompound:
        """Select optimal tire compound.

        Args:
            current_lap: Current lap number
            remaining_laps: Laps remaining in race
            weather: Current weather condition

        Returns:
            Recommended TireCompound
        """
        if weather == "wet":
            return TireCompound.WET
        elif weather == "intermediate":
            return TireCompound.INTERMEDIATE
        else:
            # Dry conditions
            if remaining_laps < 15:
                return TireCompound.SOFT  # Fastest for short stints
            elif remaining_laps < 30:
                return TireCompound.MEDIUM
            else:
                return TireCompound.HARD  # Most durable for long stints

    def calculate_stint_duration(
        self,
        tire_compound: TireCompound,
        fuel_available: float,
    ) -> int:
        """Calculate how long a stint can last with given tire.

        Args:
            tire_compound: Tire compound to use
            fuel_available: Available fuel as percentage

        Returns:
            Estimated laps possible
        """
        # Calculate fuel-limited laps
        fuel_limited_laps = fuel_available * self.fuel_capacity_laps / 100.0

        # Estimate tire-limited laps (based on typical degradation)
        # Assume tires can do 40 laps on soft, 50 on medium, 60 on hard, etc.
        tire_limits = {
            TireCompound.SOFT: 35,
            TireCompound.MEDIUM: 45,
            TireCompound.HARD: 60,
            TireCompound.INTERMEDIATE: 40,
            TireCompound.WET: 25,
        }

        tire_limited_laps = tire_limits.get(tire_compound, 40)

        # Return minimum of fuel and tire limits
        return int(min(fuel_limited_laps, tire_limited_laps))

    def estimate_time_loss(
        self,
        num_stops: int,
    ) -> float:
        """Estimate time loss from pit stops.

        Args:
            num_stops: Number of pit stops

        Returns:
            Total time loss in seconds
        """
        # Each pit stop loses: pit duration + time to regain position (30-60 seconds)
        time_per_stop = self.PIT_STOP_DURATION + 40.0  # 40s to regain position
        return num_stops * time_per_stop
