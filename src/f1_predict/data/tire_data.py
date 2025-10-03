"""Tire strategy data parser and enrichment module."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from f1_predict.data.external_models import (
    PitStopStrategy,
    TireCompound,
    TireStintData,
)


class TireDataCollector:
    """Collects and parses tire strategy and stint data from race results."""

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize tire data collector.

        Args:
            data_dir: Base directory for storing data files
        """
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.external_dir = self.data_dir / "external" / "tire_data"
        self.external_dir.mkdir(parents=True, exist_ok=True)

    def parse_tire_compound(self, compound_str: Optional[str]) -> TireCompound:
        """Parse tire compound string to enum.

        Args:
            compound_str: Compound string from data source

        Returns:
            Tire compound enum value
        """
        if not compound_str:
            return TireCompound.MEDIUM  # Default

        compound_lower = compound_str.lower().strip()

        compound_map = {
            "soft": TireCompound.SOFT,
            "medium": TireCompound.MEDIUM,
            "hard": TireCompound.HARD,
            "intermediate": TireCompound.INTERMEDIATE,
            "inter": TireCompound.INTERMEDIATE,
            "wet": TireCompound.WET,
            "hypersoft": TireCompound.HYPERSOFT,
            "ultrasoft": TireCompound.ULTRASOFT,
            "supersoft": TireCompound.SUPERSOFT,
            "superhard": TireCompound.SUPERHARD,
            "s": TireCompound.SOFT,
            "m": TireCompound.MEDIUM,
            "h": TireCompound.HARD,
            "i": TireCompound.INTERMEDIATE,
            "w": TireCompound.WET,
        }

        return compound_map.get(compound_lower, TireCompound.MEDIUM)

    def extract_pit_strategy_from_race(
        self,
        race_data: dict,
        pit_stops: list[dict],
    ) -> Optional[PitStopStrategy]:
        """Extract pit stop strategy from race results and pit stop data.

        Args:
            race_data: Race result data
            pit_stops: List of pit stop records

        Returns:
            Pit stop strategy or None if insufficient data
        """
        if not pit_stops:
            return None

        try:
            # Sort pit stops by lap
            sorted_stops = sorted(pit_stops, key=lambda x: int(x.get("lap", 0)))

            # Extract tire compounds if available
            tire_sequence = []
            for stop in sorted_stops:
                if "compound" in stop:
                    tire_sequence.append(self.parse_tire_compound(stop["compound"]))

            # Calculate average pit duration
            durations = [float(stop.get("duration", 0)) for stop in sorted_stops if "duration" in stop]
            avg_duration = sum(durations) / len(durations) if durations else None

            strategy = PitStopStrategy(
                season=race_data.get("season", ""),
                round=race_data.get("round", ""),
                driver_id=race_data.get("driver_id", ""),
                constructor_id=race_data.get("constructor_id", ""),
                total_pit_stops=len(sorted_stops),
                planned_stops=len(sorted_stops),  # Assume all were planned
                starting_compound=tire_sequence[0] if tire_sequence else TireCompound.MEDIUM,
                tire_sequence=tire_sequence,
                pit_stop_laps=[int(stop.get("lap", 0)) for stop in sorted_stops],
                pit_stop_durations=durations,
                average_pit_duration=avg_duration,
            )

            return strategy

        except Exception as e:
            self.logger.error(f"Failed to extract pit strategy: {e}")
            return None

    def calculate_tire_degradation(
        self,
        stint_data: list[dict],
    ) -> Optional[float]:
        """Calculate tire degradation rate from lap times.

        Args:
            stint_data: List of lap time data

        Returns:
            Degradation rate in seconds per lap or None
        """
        if len(stint_data) < 3:
            return None

        try:
            # Extract lap times (assuming they're in seconds)
            lap_times = []
            for lap in stint_data:
                if "time" in lap:
                    # Parse time string or use float
                    time = lap["time"]
                    if isinstance(time, str):
                        # Parse MM:SS.mmm format
                        parts = time.split(":")
                        if len(parts) == 2:
                            minutes = float(parts[0])
                            seconds = float(parts[1])
                            total_seconds = minutes * 60 + seconds
                            lap_times.append(total_seconds)
                    else:
                        lap_times.append(float(time))

            if len(lap_times) < 3:
                return None

            # Simple linear degradation: (last_lap_avg - first_lap_avg) / num_laps
            # Use average of first 2 and last 2 laps to smooth outliers
            first_avg = sum(lap_times[:2]) / 2
            last_avg = sum(lap_times[-2:]) / 2
            num_laps = len(lap_times) - 1

            degradation = (last_avg - first_avg) / num_laps if num_laps > 0 else 0.0

            return degradation

        except Exception as e:
            self.logger.error(f"Failed to calculate degradation: {e}")
            return None

    def create_tire_stint(
        self,
        season: str,
        round_num: str,
        driver_id: str,
        compound: TireCompound,
        stint_number: int,
        starting_lap: int,
        ending_lap: int,
        lap_times: Optional[list[float]] = None,
        stint_end_reason: Optional[str] = None,
    ) -> TireStintData:
        """Create tire stint data record.

        Args:
            season: Season year
            round_num: Round number
            driver_id: Driver identifier
            compound: Tire compound used
            stint_number: Stint number in the race
            starting_lap: First lap of stint
            ending_lap: Last lap of stint
            lap_times: Optional list of lap times for the stint
            stint_end_reason: Reason for ending stint

        Returns:
            Tire stint data
        """
        laps_completed = ending_lap - starting_lap + 1

        avg_lap_time = None
        fastest_lap_time = None
        degradation = None

        if lap_times:
            avg_lap_time = sum(lap_times) / len(lap_times)
            fastest_lap_time = min(lap_times)

            # Calculate simple degradation
            if len(lap_times) >= 3:
                first_avg = sum(lap_times[:2]) / 2
                last_avg = sum(lap_times[-2:]) / 2
                degradation = (last_avg - first_avg) / (len(lap_times) - 1)

        return TireStintData(
            session_type="race",
            season=season,
            round=round_num,
            driver_id=driver_id,
            compound=compound,
            stint_number=stint_number,
            starting_lap=starting_lap,
            ending_lap=ending_lap,
            laps_completed=laps_completed,
            average_lap_time=avg_lap_time,
            fastest_lap_time=fastest_lap_time,
            degradation_rate=degradation,
            stint_end_reason=stint_end_reason,
        )

    def save_tire_strategies(
        self,
        strategies: list[PitStopStrategy],
        filename: str = "pit_strategies_2020_2024.json",
    ) -> Path:
        """Save tire strategies to file.

        Args:
            strategies: List of pit stop strategies
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_file = self.external_dir / filename

        data = [s.model_dump(mode="json") for s in strategies]

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Saved {len(strategies)} tire strategies to {output_file}")
        return output_file

    def save_tire_stints(
        self,
        stints: list[TireStintData],
        filename: str = "tire_stints_2020_2024.json",
    ) -> Path:
        """Save tire stint data to file.

        Args:
            stints: List of tire stints
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_file = self.external_dir / filename

        data = [s.model_dump(mode="json") for s in stints]

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Saved {len(stints)} tire stints to {output_file}")
        return output_file

    def load_tire_strategies(
        self,
        filename: str = "pit_strategies_2020_2024.json",
    ) -> list[PitStopStrategy]:
        """Load tire strategies from file.

        Args:
            filename: Input filename

        Returns:
            List of pit stop strategies
        """
        input_file = self.external_dir / filename

        if not input_file.exists():
            self.logger.warning(f"Tire strategies file not found: {input_file}")
            return []

        with open(input_file) as f:
            data = json.load(f)

        strategies = [PitStopStrategy(**d) for d in data]
        self.logger.info(f"Loaded {len(strategies)} tire strategies from {input_file}")
        return strategies

    def load_tire_stints(
        self,
        filename: str = "tire_stints_2020_2024.json",
    ) -> list[TireStintData]:
        """Load tire stint data from file.

        Args:
            filename: Input filename

        Returns:
            List of tire stints
        """
        input_file = self.external_dir / filename

        if not input_file.exists():
            self.logger.warning(f"Tire stints file not found: {input_file}")
            return []

        with open(input_file) as f:
            data = json.load(f)

        stints = [TireStintData(**d) for d in data]
        self.logger.info(f"Loaded {len(stints)} tire stints from {input_file}")
        return stints
