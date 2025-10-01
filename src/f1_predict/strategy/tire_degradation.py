"""Tire degradation modeling for F1 race strategy.

This module implements physics-based tire wear simulation considering:
- Compound characteristics (soft/medium/hard/wet/intermediate)
- Temperature effects on degradation
- Driver style impact
- Fuel load effects
- Track abrasiveness
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

# Constants for tire degradation modeling
TIRE_CLIFF_THRESHOLD = 3.0  # seconds - degradation threshold for cliff effect


class TireCompound(Enum):
    """F1 tire compounds with degradation characteristics."""

    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"
    INTERMEDIATE = "intermediate"
    WET = "wet"


@dataclass
class TireDegradationConfig:
    """Configuration for tire degradation model.

    Attributes:
        soft_deg_rate: Base degradation rate for soft tires (seconds per lap)
        medium_deg_rate: Base degradation rate for medium tires (seconds per lap)
        hard_deg_rate: Base degradation rate for hard tires (seconds per lap)
        optimal_temp: Optimal tire operating temperature (°C)
        temp_deg_multiplier: Degradation increase per °C deviation from optimal
        track_abrasiveness: Circuit-specific multiplier (1.0 = average)
        downforce_level: Aerodynamic load effect (1.0 = average)
        aggressive_multiplier: Multiplier for aggressive driving style
    """

    # Base degradation rates (seconds per lap)
    soft_deg_rate: float = 0.08  # Soft: highest degradation
    medium_deg_rate: float = 0.05  # Medium: balanced
    hard_deg_rate: float = 0.03  # Hard: lowest degradation

    # Temperature effects
    optimal_temp: float = 90.0  # Optimal tire temperature (°C)
    temp_deg_multiplier: float = 0.02  # Degradation increase per °C deviation

    # Track characteristics
    track_abrasiveness: float = 1.0  # Circuit-specific multiplier
    downforce_level: float = 1.0  # Aero load effect

    # Driver style
    aggressive_multiplier: float = 1.15  # Aggressive driving increases wear


class TireDegradationModel:
    """Model tire wear throughout a race stint.

    This model calculates lap time degradation based on multiple factors including
    tire compound, temperature, fuel load, driver style, and track characteristics.
    It implements non-linear degradation patterns including the "tire cliff" effect.
    """

    def __init__(self, config: TireDegradationConfig | None = None):
        """Initialize tire degradation model.

        Args:
            config: Configuration for degradation parameters. If None, uses defaults.
        """
        self.config = config or TireDegradationConfig()

    def calculate_lap_time_delta(
        self,
        compound: TireCompound,
        lap_number: int,
        track_temp: float,
        fuel_load: float,
        driver_style: str = "neutral",
    ) -> float:
        """Calculate lap time delta due to tire degradation.

        Args:
            compound: Tire compound being used
            lap_number: Current lap number in stint (1-indexed)
            track_temp: Track temperature in °C
            fuel_load: Current fuel load in kg
            driver_style: Driver style - "aggressive", "neutral", or "conservative"

        Returns:
            Lap time delta in seconds (positive = slower than fresh tires)

        Raises:
            ValueError: If lap_number < 1 or fuel_load < 0
        """
        if lap_number < 1:
            msg = "lap_number must be >= 1"
            raise ValueError(msg)
        if fuel_load < 0:
            msg = "fuel_load must be >= 0"
            raise ValueError(msg)

        # Base degradation rate by compound
        base_rates = {
            TireCompound.SOFT: self.config.soft_deg_rate,
            TireCompound.MEDIUM: self.config.medium_deg_rate,
            TireCompound.HARD: self.config.hard_deg_rate,
            TireCompound.INTERMEDIATE: 0.04,
            TireCompound.WET: 0.02,
        }
        base_rate = base_rates[compound]

        # Temperature effect (non-linear)
        temp_delta = abs(track_temp - self.config.optimal_temp)
        temp_factor = 1 + (temp_delta * self.config.temp_deg_multiplier)

        # Driver style multiplier
        style_multipliers = {
            "aggressive": self.config.aggressive_multiplier,
            "neutral": 1.0,
            "conservative": 0.85,
        }
        style_factor = style_multipliers.get(driver_style, 1.0)

        # Fuel load effect (lighter car = more tire stress in corners)
        # Max fuel load is ~110kg, effect ranges from 0-10%
        fuel_factor = 1 - (fuel_load / 110) * 0.1

        # Cumulative degradation (exponential growth)
        total_degradation = (
            base_rate
            * lap_number
            * temp_factor
            * style_factor
            * fuel_factor
            * self.config.track_abrasiveness
        )

        # Add non-linearity for cliff effect
        if total_degradation > TIRE_CLIFF_THRESHOLD:
            total_degradation *= 1.5

        return total_degradation

    def predict_stint_performance(
        self, compound: TireCompound, stint_length: int, conditions: dict
    ) -> np.ndarray:
        """Predict lap times for entire tire stint.

        Args:
            compound: Tire compound to use
            stint_length: Number of laps in stint
            conditions: Dictionary with keys:
                - track_temp: Track temperature (°C)
                - fuel_load: Starting fuel load (kg)
                - driver_style: "aggressive", "neutral", or "conservative"

        Returns:
            NumPy array of lap time deltas (seconds) for each lap

        Raises:
            ValueError: If stint_length < 1
            KeyError: If required conditions are missing
        """
        if stint_length < 1:
            msg = "stint_length must be >= 1"
            raise ValueError(msg)

        # Validate required conditions
        required = ["track_temp", "fuel_load"]
        missing = [k for k in required if k not in conditions]
        if missing:
            msg = f"Missing required conditions: {missing}"
            raise KeyError(msg)

        lap_deltas = []

        for lap in range(1, stint_length + 1):
            # Account for fuel burn (reduces weight by ~1kg per lap)
            current_fuel = conditions["fuel_load"] - (lap * 1.0)

            delta = self.calculate_lap_time_delta(
                compound=compound,
                lap_number=lap,
                track_temp=conditions["track_temp"],
                fuel_load=max(0, current_fuel),
                driver_style=conditions.get("driver_style", "neutral"),
            )
            lap_deltas.append(delta)

        return np.array(lap_deltas)

    def estimate_optimal_stint_length(
        self, compound: TireCompound, conditions: dict, max_deg_threshold: float = 2.0
    ) -> int:
        """Estimate optimal stint length before tire cliff.

        Args:
            compound: Tire compound
            conditions: Race conditions (track_temp, fuel_load, driver_style)
            max_deg_threshold: Maximum acceptable degradation (seconds)

        Returns:
            Recommended stint length (laps)
        """
        # Binary search for optimal stint length
        low, high = 1, 60  # Typical F1 stint range

        while low < high:
            mid = (low + high + 1) // 2
            delta = self.calculate_lap_time_delta(
                compound=compound,
                lap_number=mid,
                track_temp=conditions["track_temp"],
                fuel_load=max(0, conditions["fuel_load"] - mid),
                driver_style=conditions.get("driver_style", "neutral"),
            )

            if delta <= max_deg_threshold:
                low = mid
            else:
                high = mid - 1

        return low
