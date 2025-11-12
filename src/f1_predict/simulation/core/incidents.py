"""Incident and event simulation for race progression."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class IncidentType(str, Enum):
    """Types of race incidents."""

    SAFETY_CAR = "safety_car"
    RED_FLAG = "red_flag"
    DNF_MECHANICAL = "dnf_mechanical"
    DNF_CRASH = "dnf_crash"
    DNF_OTHER = "dnf_other"
    WEATHER_CHANGE = "weather_change"


@dataclass
class IncidentEvent:
    """Single incident event in race.

    Attributes:
        lap: Lap on which incident occurred
        incident_type: Type of incident
        affected_driver_id: ID of affected driver (if applicable)
        description: Human-readable description
    """

    lap: int
    incident_type: IncidentType
    affected_driver_id: Optional[str] = None
    description: str = ""

    def __str__(self) -> str:
        """String representation of incident."""
        return (
            f"Lap {self.lap}: {self.incident_type.value.upper()} - {self.description}"
        )


class IncidentGenerator:
    """Generate random incidents during race simulation.

    Uses probabilistic models based on circuit characteristics and weather
    to generate realistic safety car, DNF, and weather change events.
    """

    # Base probabilities by circuit type
    CIRCUIT_SAFETY_CAR_PROBS = {
        "street": 0.12,  # Higher for street circuits
        "tight": 0.10,  # Hungary, Monaco
        "intermediate": 0.08,  # Most circuits
        "high_speed": 0.05,  # Monza, Silverstone
    }

    DNF_BASE_RATE = 0.08  # 8% DNF rate per driver

    def __init__(
        self,
        circuit_type: str = "intermediate",
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize incident generator.

        Args:
            circuit_type: Type of circuit (street, tight, intermediate, high_speed)
            random_state: Random seed for reproducibility
        """
        self.circuit_type = circuit_type
        self.safety_car_prob = self.CIRCUIT_SAFETY_CAR_PROBS.get(
            circuit_type, 0.08
        )
        self.rng = np.random.RandomState(random_state)
        self.incident_log: list[IncidentEvent] = []

    def generate_safety_car(self, current_lap: int, total_laps: int) -> Optional[IncidentEvent]:
        """Generate safety car event probabilistically.

        Args:
            current_lap: Current lap number
            total_laps: Total race laps

        Returns:
            IncidentEvent if safety car triggered, None otherwise
        """
        if current_lap > total_laps * 0.85:  # No SC in final 15% of race
            return None

        if self.rng.random() < self.safety_car_prob:
            event = IncidentEvent(
                lap=current_lap,
                incident_type=IncidentType.SAFETY_CAR,
                description=f"Safety car deployed at lap {current_lap}",
            )
            self.incident_log.append(event)
            logger.debug(f"Generated SC event: {event}")
            return event

        return None

    def generate_dnf(
        self,
        driver_id: str,
        driver_name: str,
        laps_completed: int,
        total_laps: int,
    ) -> Optional[IncidentEvent]:
        """Generate DNF event for specific driver.

        Args:
            driver_id: Driver identifier
            driver_name: Driver name
            laps_completed: Number of laps completed
            total_laps: Total race laps

        Returns:
            IncidentEvent if DNF triggered, None otherwise
        """
        # Adjust DNF probability based on race distance
        adjusted_dnf_rate = self.DNF_BASE_RATE * (laps_completed / total_laps)

        if self.rng.random() < adjusted_dnf_rate:
            # Choose DNF reason
            dnf_reason = self.rng.choice(
                ["mechanical failure", "crash", "engine failure"],
                p=[0.5, 0.3, 0.2],
            )
            event = IncidentEvent(
                lap=laps_completed,
                incident_type=IncidentType.DNF_MECHANICAL
                if dnf_reason == "mechanical failure"
                else IncidentType.DNF_CRASH
                if dnf_reason == "crash"
                else IncidentType.DNF_OTHER,
                affected_driver_id=driver_id,
                description=f"{driver_name} retired ({dnf_reason})",
            )
            self.incident_log.append(event)
            logger.debug(f"Generated DNF event: {event}")
            return event

        return None

    def generate_weather_change(
        self, current_lap: int, current_condition: str = "dry"
    ) -> Optional[IncidentEvent]:
        """Generate weather change event.

        Args:
            current_lap: Current lap number
            current_condition: Current weather condition

        Returns:
            IncidentEvent if weather changes, None otherwise
        """
        # Simplified weather change model
        change_prob = 0.02  # 2% chance per lap

        if self.rng.random() < change_prob:
            conditions = ["dry", "intermediate", "wet"]
            conditions.remove(current_condition)
            new_condition = self.rng.choice(conditions)

            event = IncidentEvent(
                lap=current_lap,
                incident_type=IncidentType.WEATHER_CHANGE,
                description=f"Weather changes to {new_condition} at lap {current_lap}",
            )
            self.incident_log.append(event)
            logger.debug(f"Generated weather change: {event}")
            return event

        return None

    def clear_incidents(self) -> None:
        """Clear incident log."""
        self.incident_log.clear()

    def get_incidents(self) -> list[IncidentEvent]:
        """Get all recorded incidents.

        Returns:
            List of IncidentEvent objects
        """
        return self.incident_log.copy()
