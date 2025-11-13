"""Scenario builder for configuring what-if analysis scenarios."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json
from pathlib import Path

from f1_predict.simulation.core.driver_state import DriverState, TireCompound
from f1_predict.simulation.core.race_state import CircuitContext
from f1_predict.simulation.engine.pit_strategy import TireStrategy


class ScenarioType(Enum):
    """Types of scenarios for what-if analysis."""

    BASELINE = "baseline"  # Original race state
    GRID_CHANGE = "grid_change"  # Modified starting grid
    WEATHER = "weather"  # Weather condition override
    STRATEGY = "strategy"  # Custom pit strategies
    MECHANICAL = "mechanical"  # Driver mechanical failures
    CUSTOM = "custom"  # Multiple parameter changes


@dataclass
class GridChange:
    """Represents a change to the starting grid."""

    driver_id: str
    new_position: int
    reason: str = ""

    def __post_init__(self) -> None:
        """Validate grid change."""
        if self.new_position < 1:
            raise ValueError("Position must be >= 1")


@dataclass
class DriverStrategy:
    """Custom pit strategy for a driver."""

    driver_id: str
    strategy: TireStrategy
    pit_laps: Optional[List[int]] = None  # Specific laps to pit
    initial_tire: TireCompound = TireCompound.SOFT

    def __post_init__(self) -> None:
        """Validate strategy configuration."""
        if self.pit_laps is not None:
            for lap in self.pit_laps:
                if lap < 1:
                    raise ValueError("Pit laps must be >= 1")


@dataclass
class DriverMechanicalIssue:
    """Represents a mechanical issue for a driver."""

    driver_id: str
    issue_type: str  # e.g., "engine_failure", "hydraulics", "suspension"
    lap: int
    severity: str = "dnf"  # "dnf" or "performance_loss"
    performance_penalty: Optional[float] = None  # % increase in lap time

    def __post_init__(self) -> None:
        """Validate mechanical issue."""
        if self.lap < 1:
            raise ValueError("Lap must be >= 1")
        if self.severity not in ["dnf", "performance_loss"]:
            raise ValueError("Severity must be 'dnf' or 'performance_loss'")
        if self.severity == "performance_loss" and self.performance_penalty is None:
            raise ValueError("performance_penalty required for performance_loss")


@dataclass
class WeatherCondition:
    """Weather conditions for a scenario."""

    condition_type: str  # "dry", "wet", "intermediate"
    start_lap: int = 1
    duration_laps: Optional[int] = None  # None = entire race

    def __post_init__(self) -> None:
        """Validate weather condition."""
        valid_types = ["dry", "wet", "intermediate"]
        if self.condition_type not in valid_types:
            raise ValueError(f"condition_type must be one of {valid_types}")
        if self.start_lap < 1:
            raise ValueError("start_lap must be >= 1")


@dataclass
class RaceScenario:
    """Complete scenario definition for what-if analysis."""

    scenario_id: str
    scenario_type: ScenarioType
    circuit: CircuitContext
    drivers: List[DriverState]
    description: str = ""
    grid_changes: List[GridChange] = field(default_factory=list)
    driver_strategies: List[DriverStrategy] = field(default_factory=list)
    mechanical_issues: List[DriverMechanicalIssue] = field(default_factory=list)
    weather_conditions: List[WeatherCondition] = field(default_factory=list)
    n_simulations: int = 100
    random_seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate scenario configuration."""
        if not self.scenario_id:
            raise ValueError("scenario_id required")
        if not self.drivers:
            raise ValueError("At least one driver required")

        # Validate grid changes
        grid_positions = set()
        for change in self.grid_changes:
            if change.new_position in grid_positions:
                raise ValueError(f"Duplicate grid position: {change.new_position}")
            grid_positions.add(change.new_position)

    def get_modified_drivers(self) -> List[DriverState]:
        """Get drivers with grid changes applied."""
        drivers = [d.copy() for d in self.drivers]

        # Apply grid changes
        for change in self.grid_changes:
            for driver in drivers:
                if driver.driver_id == change.driver_id:
                    driver.update_position(change.new_position)
                    break

        # Re-sort by position
        drivers.sort(key=lambda d: d.position)

        return drivers

    def get_driver_strategy(self, driver_id: str) -> Optional[DriverStrategy]:
        """Get custom strategy for a driver."""
        for strategy in self.driver_strategies:
            if strategy.driver_id == driver_id:
                return strategy
        return None

    def get_mechanical_issues_for_driver(
        self, driver_id: str
    ) -> List[DriverMechanicalIssue]:
        """Get all mechanical issues for a driver."""
        return [issue for issue in self.mechanical_issues if issue.driver_id == driver_id]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scenario to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type.value,
            "circuit": {
                "circuit_name": self.circuit.circuit_name,
                "circuit_type": self.circuit.circuit_type,
                "total_laps": self.circuit.total_laps,
                "lap_distance": self.circuit.lap_distance,
            },
            "drivers": [
                {
                    "driver_id": d.driver_id,
                    "driver_name": d.driver_name,
                    "position": d.position,
                    "expected_lap_time": d.expected_lap_time,
                }
                for d in self.drivers
            ],
            "description": self.description,
            "grid_changes": [
                {
                    "driver_id": gc.driver_id,
                    "new_position": gc.new_position,
                    "reason": gc.reason,
                }
                for gc in self.grid_changes
            ],
            "driver_strategies": [
                {
                    "driver_id": ds.driver_id,
                    "strategy": ds.strategy.value,
                    "pit_laps": ds.pit_laps,
                    "initial_tire": ds.initial_tire.value,
                }
                for ds in self.driver_strategies
            ],
            "mechanical_issues": [
                {
                    "driver_id": mi.driver_id,
                    "issue_type": mi.issue_type,
                    "lap": mi.lap,
                    "severity": mi.severity,
                    "performance_penalty": mi.performance_penalty,
                }
                for mi in self.mechanical_issues
            ],
            "weather_conditions": [
                {
                    "condition_type": wc.condition_type,
                    "start_lap": wc.start_lap,
                    "duration_laps": wc.duration_laps,
                }
                for wc in self.weather_conditions
            ],
            "n_simulations": self.n_simulations,
            "random_seed": self.random_seed,
            "metadata": self.metadata,
        }


class ScenarioBuilder:
    """Builder pattern for creating scenarios."""

    def __init__(self, scenario_id: str, circuit: CircuitContext):
        """Initialize scenario builder."""
        self.scenario_id = scenario_id
        self.circuit = circuit
        self.scenario_type = ScenarioType.BASELINE
        self.drivers: List[DriverState] = []
        self.description = ""
        self.grid_changes: List[GridChange] = []
        self.driver_strategies: List[DriverStrategy] = []
        self.mechanical_issues: List[DriverMechanicalIssue] = []
        self.weather_conditions: List[WeatherCondition] = []
        self.n_simulations = 100
        self.random_seed: Optional[int] = None
        self.metadata: Dict[str, Any] = {}

    def with_drivers(self, drivers: List[DriverState]) -> "ScenarioBuilder":
        """Set drivers for scenario."""
        self.drivers = drivers
        return self

    def with_description(self, description: str) -> "ScenarioBuilder":
        """Set scenario description."""
        self.description = description
        return self

    def with_type(self, scenario_type: ScenarioType) -> "ScenarioBuilder":
        """Set scenario type."""
        self.scenario_type = scenario_type
        return self

    def add_grid_change(
        self, driver_id: str, new_position: int, reason: str = ""
    ) -> "ScenarioBuilder":
        """Add grid change to scenario."""
        self.grid_changes.append(GridChange(driver_id, new_position, reason))
        return self

    def add_driver_strategy(
        self,
        driver_id: str,
        strategy: TireStrategy,
        pit_laps: Optional[List[int]] = None,
        initial_tire: TireCompound = TireCompound.SOFT,
    ) -> "ScenarioBuilder":
        """Add custom pit strategy for driver."""
        self.driver_strategies.append(
            DriverStrategy(driver_id, strategy, pit_laps, initial_tire)
        )
        return self

    def add_mechanical_issue(
        self,
        driver_id: str,
        issue_type: str,
        lap: int,
        severity: str = "dnf",
        performance_penalty: Optional[float] = None,
    ) -> "ScenarioBuilder":
        """Add mechanical issue for driver."""
        self.mechanical_issues.append(
            DriverMechanicalIssue(driver_id, issue_type, lap, severity, performance_penalty)
        )
        return self

    def add_weather(
        self,
        condition_type: str,
        start_lap: int = 1,
        duration_laps: Optional[int] = None,
    ) -> "ScenarioBuilder":
        """Add weather condition change."""
        self.weather_conditions.append(
            WeatherCondition(condition_type, start_lap, duration_laps)
        )
        return self

    def with_simulations(self, n_simulations: int) -> "ScenarioBuilder":
        """Set number of simulations."""
        if n_simulations < 1:
            raise ValueError("n_simulations must be >= 1")
        self.n_simulations = n_simulations
        return self

    def with_seed(self, seed: int) -> "ScenarioBuilder":
        """Set random seed for reproducibility."""
        self.random_seed = seed
        return self

    def with_metadata(self, key: str, value: Any) -> "ScenarioBuilder":
        """Add metadata entry."""
        self.metadata[key] = value
        return self

    def build(self) -> RaceScenario:
        """Build and return the scenario."""
        if self.grid_changes or self.driver_strategies or self.mechanical_issues:
            self.scenario_type = ScenarioType.CUSTOM

        return RaceScenario(
            scenario_id=self.scenario_id,
            scenario_type=self.scenario_type,
            circuit=self.circuit,
            drivers=self.drivers,
            description=self.description,
            grid_changes=self.grid_changes,
            driver_strategies=self.driver_strategies,
            mechanical_issues=self.mechanical_issues,
            weather_conditions=self.weather_conditions,
            n_simulations=self.n_simulations,
            random_seed=self.random_seed,
            metadata=self.metadata,
        )


class ScenarioRepository:
    """Manage scenario storage and retrieval."""

    def __init__(self, storage_dir: Path):
        """Initialize scenario repository."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_scenario(self, scenario: RaceScenario) -> Path:
        """Save scenario to file."""
        file_path = self.storage_dir / f"{scenario.scenario_id}.json"

        scenario_dict = scenario.to_dict()
        with open(file_path, "w") as f:
            json.dump(scenario_dict, f, indent=2)

        return file_path

    def load_scenario_dict(self, scenario_id: str) -> Dict[str, Any]:
        """Load scenario dictionary from file."""
        file_path = self.storage_dir / f"{scenario_id}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {file_path}")

        with open(file_path, "r") as f:
            return json.load(f)

    def list_scenarios(self) -> List[str]:
        """List all saved scenario IDs."""
        return [f.stem for f in self.storage_dir.glob("*.json")]

    def delete_scenario(self, scenario_id: str) -> None:
        """Delete a saved scenario."""
        file_path = self.storage_dir / f"{scenario_id}.json"
        if file_path.exists():
            file_path.unlink()
