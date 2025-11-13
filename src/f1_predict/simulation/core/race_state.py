"""Race state management and progression tracking."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from f1_predict.simulation.core.driver_state import DriverState, DriverStatus


@dataclass
class CircuitContext:
    """Circuit characteristics for simulation.

    Attributes:
        circuit_name: Name of the circuit
        circuit_type: Type of circuit (street, tight, intermediate, high_speed)
        total_laps: Total laps in race
        lap_distance: Distance per lap in km
        safety_car_prob: Probability of safety car (0.0-1.0)
        dnf_rate: Base DNF rate for the circuit
    """

    circuit_name: str
    circuit_type: str = "intermediate"
    total_laps: int = 58
    lap_distance: float = 5.3  # Average lap distance
    safety_car_prob: float = 0.08
    dnf_rate: float = 0.08


@dataclass
class RaceState:
    """Complete race state at any point in time.

    Manages driver states, lap information, and race progression tracking.

    Attributes:
        circuit: Circuit information
        drivers: Dictionary mapping driver_id to DriverState
        current_lap: Current lap number
        safety_car_active: Whether safety car is currently out
        race_finished: Whether race has finished
        weather_condition: Current weather (dry, intermediate, wet)
        track_temperature: Track temperature in Celsius
    """

    circuit: CircuitContext
    drivers: dict[str, DriverState] = field(default_factory=dict)
    current_lap: int = 1
    safety_car_active: bool = False
    race_finished: bool = False
    weather_condition: str = "dry"
    track_temperature: float = 25.0
    lap_history: list[dict] = field(default_factory=list)

    def add_driver(self, driver: DriverState) -> None:
        """Add driver to race.

        Args:
            driver: DriverState to add

        Raises:
            ValueError: If driver already exists
        """
        if driver.driver_id in self.drivers:
            raise ValueError(f"Driver {driver.driver_id} already in race")
        self.drivers[driver.driver_id] = driver

    def remove_driver(self, driver_id: str, reason: str = "") -> None:
        """Remove driver from race (DNF).

        Args:
            driver_id: ID of driver to remove
            reason: Reason for removal
        """
        if driver_id not in self.drivers:
            raise ValueError(f"Driver {driver_id} not in race")

        self.drivers[driver_id].dnf(reason)

    def get_driver(self, driver_id: str) -> DriverState:
        """Get driver state by ID.

        Args:
            driver_id: ID of driver

        Returns:
            DriverState for the driver

        Raises:
            ValueError: If driver not found
        """
        if driver_id not in self.drivers:
            raise ValueError(f"Driver {driver_id} not found")
        return self.drivers[driver_id]

    def get_active_drivers(self) -> list[DriverState]:
        """Get all drivers still actively racing.

        Returns:
            List of active DriverState objects sorted by position
        """
        active = [d for d in self.drivers.values() if d.is_active]
        return sorted(active, key=lambda d: d.position)

    def get_finished_drivers(self) -> list[DriverState]:
        """Get drivers who finished the race.

        Returns:
            List of finished DriverState objects in finishing order
        """
        finished = [d for d in self.drivers.values() if d.is_finished]
        return sorted(finished, key=lambda d: d.position)

    def get_dnf_drivers(self) -> list[DriverState]:
        """Get drivers who did not finish.

        Returns:
            List of DNF DriverState objects
        """
        return [d for d in self.drivers.values() if d.is_dnf]

    def update_positions(self) -> None:
        """Update driver positions based on lap and gap information.

        Positions are determined by lap count, then by gap to leader.
        """
        active_drivers = self.get_active_drivers()

        # Sort by lap count (descending), then by gap to leader (ascending)
        sorted_drivers = sorted(
            active_drivers, key=lambda d: (-d.lap, d.gap_to_leader)
        )

        # Update positions
        for new_position, driver in enumerate(sorted_drivers, start=1):
            driver.update_position(new_position)

        # Update gaps
        if sorted_drivers:
            for i, driver in enumerate(sorted_drivers):
                if i == 0:
                    driver.gap_to_leader = 0.0
                else:
                    gap_to_leader = sum(
                        sorted_drivers[0].expected_lap_time - d.expected_lap_time
                        for d in sorted_drivers[1 : i + 1]
                    )
                    driver.gap_to_leader = max(0, gap_to_leader)

                if i > 0:
                    driver.gap_to_previous = sorted_drivers[i - 1].gap_to_leader
                else:
                    driver.gap_to_previous = 0.0

    def advance_lap(self) -> None:
        """Advance race to next lap.

        Updates current lap number and checks if race is finished.
        """
        self.current_lap += 1

        # Check if all active drivers have finished
        active = self.get_active_drivers()
        if not active or all(d.lap > self.circuit.total_laps for d in active):
            self.race_finished = True

    def finish_race(self) -> None:
        """Mark race as finished and set final positions."""
        self.race_finished = True

        # Mark remaining active drivers as finished
        for driver in self.get_active_drivers():
            driver.finish_race()

        # Set final positions based on laps completed and gaps
        self.update_positions()

    def get_race_results(self) -> list[dict]:
        """Get final race results in finishing order.

        Returns:
            List of dictionaries with driver info and results
        """
        results = []

        # Finished drivers
        for driver in self.get_finished_drivers():
            results.append(
                {
                    "position": driver.position,
                    "driver_id": driver.driver_id,
                    "driver_name": driver.driver_name,
                    "status": driver.status.value,
                    "laps": driver.lap - 1,  # Actual completed laps
                    "pit_stops": driver.pit_stop_count,
                    "best_lap": driver.best_lap_time,
                }
            )

        # DNF drivers
        for driver in self.get_dnf_drivers():
            results.append(
                {
                    "position": None,
                    "driver_id": driver.driver_id,
                    "driver_name": driver.driver_name,
                    "status": f"DNF ({driver.dnf_reason})",
                    "laps": driver.lap - 1,
                    "pit_stops": driver.pit_stop_count,
                    "best_lap": driver.best_lap_time,
                }
            )

        return results

    def record_lap_snapshot(self) -> None:
        """Record current lap state for history tracking."""
        snapshot = {
            "lap": self.current_lap,
            "drivers": [
                {
                    "driver_id": d.driver_id,
                    "position": d.position,
                    "lap": d.lap,
                    "gap_to_leader": d.gap_to_leader,
                    "status": d.status.value,
                }
                for d in self.get_active_drivers()
            ],
        }
        self.lap_history.append(snapshot)

    def copy(self) -> "RaceState":
        """Create a deep copy of race state.

        Returns:
            New RaceState with copied driver states
        """
        new_state = RaceState(
            circuit=self.circuit,
            current_lap=self.current_lap,
            safety_car_active=self.safety_car_active,
            race_finished=self.race_finished,
            weather_condition=self.weather_condition,
            track_temperature=self.track_temperature,
        )

        # Copy all drivers
        for driver_id, driver in self.drivers.items():
            new_state.drivers[driver_id] = driver.copy()

        new_state.lap_history = [
            h.copy() if isinstance(h, dict) else h for h in self.lap_history
        ]

        return new_state

    def is_race_complete(self) -> bool:
        """Check if race is complete.

        Returns:
            True if race is finished
        """
        return self.race_finished or all(
            d.is_dnf or d.is_finished for d in self.drivers.values()
        )

    def get_leader(self) -> Optional[DriverState]:
        """Get current race leader.

        Returns:
            DriverState of leader, or None if no active drivers
        """
        active = self.get_active_drivers()
        return active[0] if active else None
