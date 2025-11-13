"""Driver state tracking during race simulation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TireCompound(str, Enum):
    """F1 tire compounds."""

    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"


class DriverStatus(str, Enum):
    """Driver status during race."""

    RUNNING = "running"
    PIT_STOP = "pit_stop"
    DNF = "dnf"
    FINISHED = "finished"


@dataclass
class DriverState:
    """State of a driver during race simulation.

    Attributes:
        driver_id: Unique driver identifier
        driver_name: Driver full name
        position: Current position (1-indexed)
        lap: Current lap number
        gap_to_leader: Gap to leader in seconds
        gap_to_previous: Gap to driver ahead in seconds
        status: Current status (running, pit_stop, dnf, finished)
        dnf_reason: Reason for DNF if applicable
        tire_compound: Current tire compound
        laps_on_tire: Laps completed on current tires
        pit_stop_count: Number of pit stops completed
        fuel_level: Current fuel as percentage (0-100)
        expected_lap_time: Expected lap time for current conditions (seconds)
        pace_variance: Performance variance multiplier (0.95-1.05)
    """

    driver_id: str
    driver_name: str
    position: int = 1
    lap: int = 1
    gap_to_leader: float = 0.0
    gap_to_previous: float = 0.0
    status: DriverStatus = DriverStatus.RUNNING
    dnf_reason: Optional[str] = None
    tire_compound: TireCompound = TireCompound.SOFT
    laps_on_tire: int = 0
    pit_stop_count: int = 0
    fuel_level: float = 100.0
    expected_lap_time: float = 90.0  # Seconds
    pace_variance: float = 1.0
    best_lap_time: Optional[float] = None
    pit_stop_durations: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate initial state."""
        if not 0 <= self.fuel_level <= 100:
            raise ValueError(f"Fuel level must be 0-100, got {self.fuel_level}")
        if self.position < 1:
            raise ValueError(f"Position must be >= 1, got {self.position}")
        if self.lap < 1:
            raise ValueError(f"Lap must be >= 1, got {self.lap}")

    def update_position(self, new_position: int) -> None:
        """Update driver position.

        Args:
            new_position: New position (1-indexed)

        Raises:
            ValueError: If position is invalid
        """
        if new_position < 1:
            raise ValueError(f"Position must be >= 1, got {new_position}")
        self.position = new_position

    def update_gaps(self, gap_to_leader: float, gap_to_previous: float) -> None:
        """Update gap information.

        Args:
            gap_to_leader: Gap to leader in seconds
            gap_to_previous: Gap to driver ahead in seconds
        """
        self.gap_to_leader = max(0, gap_to_leader)
        self.gap_to_previous = max(0, gap_to_previous)

    def consume_fuel(self, consumption_per_lap: float) -> None:
        """Consume fuel for one lap.

        Args:
            consumption_per_lap: Fuel consumption percentage per lap
        """
        self.fuel_level = max(0, self.fuel_level - consumption_per_lap)

    def pit_stop(self, tire_compound: TireCompound, stop_duration: float) -> None:
        """Execute pit stop.

        Args:
            tire_compound: New tire compound
            stop_duration: Pit stop duration in seconds
        """
        self.status = DriverStatus.PIT_STOP
        self.tire_compound = tire_compound
        self.laps_on_tire = 0
        self.pit_stop_count += 1
        self.pit_stop_durations.append(stop_duration)
        self.fuel_level = 100.0  # Refuel

    def resume_racing(self) -> None:
        """Resume racing after pit stop."""
        self.status = DriverStatus.RUNNING

    def dnf(self, reason: str) -> None:
        """Mark driver as DNF.

        Args:
            reason: Reason for DNF (e.g., "engine failure", "crash", "mechanical")
        """
        self.status = DriverStatus.DNF
        self.dnf_reason = reason

    def finish_race(self) -> None:
        """Mark driver as finished."""
        self.status = DriverStatus.FINISHED

    def complete_lap(self, lap_time: float) -> None:
        """Complete a lap with recorded time.

        Args:
            lap_time: Lap time in seconds
        """
        self.lap += 1
        self.laps_on_tire += 1
        if self.best_lap_time is None or lap_time < self.best_lap_time:
            self.best_lap_time = lap_time

    def copy(self) -> "DriverState":
        """Create a deep copy of driver state.

        Returns:
            New DriverState instance with same values
        """
        return DriverState(
            driver_id=self.driver_id,
            driver_name=self.driver_name,
            position=self.position,
            lap=self.lap,
            gap_to_leader=self.gap_to_leader,
            gap_to_previous=self.gap_to_previous,
            status=self.status,
            dnf_reason=self.dnf_reason,
            tire_compound=self.tire_compound,
            laps_on_tire=self.laps_on_tire,
            pit_stop_count=self.pit_stop_count,
            fuel_level=self.fuel_level,
            expected_lap_time=self.expected_lap_time,
            pace_variance=self.pace_variance,
            best_lap_time=self.best_lap_time,
            pit_stop_durations=self.pit_stop_durations.copy(),
        )

    @property
    def is_active(self) -> bool:
        """Check if driver is still actively racing."""
        return self.status == DriverStatus.RUNNING

    @property
    def is_finished(self) -> bool:
        """Check if driver has finished race."""
        return self.status == DriverStatus.FINISHED

    @property
    def is_dnf(self) -> bool:
        """Check if driver has retired."""
        return self.status == DriverStatus.DNF
