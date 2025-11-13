# Scenario Builder Guide

## Overview

The Scenario Builder provides a fluent API for creating and managing "what-if" scenarios in the F1 Race Simulation engine. It enables users to define custom race conditions, parameter modifications, and constraints for exploring hypothetical race outcomes.

## Quick Start

### Basic Scenario (Baseline)

```python
from f1_predict.simulation import ScenarioBuilder, ScenarioType
from f1_predict.simulation.core.race_state import CircuitContext
from f1_predict.simulation.core.driver_state import DriverState

# Setup circuit and drivers
circuit = CircuitContext(circuit_name="Albert Park", total_laps=58)
drivers = [
    DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
    DriverState("HAM", "Lewis Hamilton", expected_lap_time=82.0),
]

# Create baseline scenario
scenario = (
    ScenarioBuilder("austria_2024_baseline", circuit)
    .with_drivers(drivers)
    .with_description("Austria 2024 - Baseline simulation")
    .with_simulations(1000)
    .with_seed(42)
    .build()
)
```

### Grid Change Scenario

```python
# What if Verstappen had a 10-place penalty?
scenario = (
    ScenarioBuilder("austria_grid_penalty", circuit)
    .with_drivers(drivers)
    .with_description("Austria 2024 - VER with 10-place penalty")
    .add_grid_change("VER", 11, "10-place penalty")
    .with_simulations(1000)
    .build()
)

# Get modified drivers with grid change applied
modified_drivers = scenario.get_modified_drivers()
```

### Pit Strategy Scenario

```python
from f1_predict.simulation.engine.pit_strategy import TireStrategy

# Define custom pit strategies
scenario = (
    ScenarioBuilder("austria_strategy", circuit)
    .with_drivers(drivers)
    .with_type(ScenarioType.STRATEGY)
    .add_driver_strategy("VER", TireStrategy.ONE_STOP, pit_laps=[32])
    .add_driver_strategy("HAM", TireStrategy.TWO_STOP, pit_laps=[20, 38])
    .with_description("Austria 2024 - Custom pit strategies")
    .build()
)

# Retrieve strategy for specific driver
ver_strategy = scenario.get_driver_strategy("VER")
print(f"VER pit laps: {ver_strategy.pit_laps}")
```

### Mechanical Issue Scenario

```python
# What if Hamilton had engine issues on lap 40?
scenario = (
    ScenarioBuilder("austria_engine_issue", circuit)
    .with_drivers(drivers)
    .with_type(ScenarioType.MECHANICAL)
    .add_mechanical_issue(
        driver_id="HAM",
        issue_type="engine_failure",
        lap=40,
        severity="dnf"
    )
    .with_description("Austria 2024 - HAM engine failure")
    .build()
)

# Or performance loss instead of DNF
scenario = (
    ScenarioBuilder("austria_hydraulic", circuit)
    .with_drivers(drivers)
    .add_mechanical_issue(
        driver_id="HAM",
        issue_type="hydraulic_leak",
        lap=30,
        severity="performance_loss",
        performance_penalty=1.5  # 1.5% slower per lap
    )
    .build()
)
```

### Weather Change Scenario

```python
# What if it rained from lap 25?
scenario = (
    ScenarioBuilder("austria_rain", circuit)
    .with_drivers(drivers)
    .add_weather("wet", start_lap=25, duration_laps=10)
    .with_description("Austria 2024 - Rain from lap 25 for 10 laps")
    .build()
)
```

### Complex Multi-Parameter Scenario

```python
# Combine multiple changes
scenario = (
    ScenarioBuilder("austria_complex", circuit)
    .with_drivers(drivers)
    .with_type(ScenarioType.CUSTOM)
    .with_description(
        "Austria 2024 - VER penalty + HAM strategy + Weather change"
    )
    # Grid change
    .add_grid_change("VER", 5, "10-place penalty")
    # Custom strategies
    .add_driver_strategy("HAM", TireStrategy.ONE_STOP)
    # Mechanical issue
    .add_mechanical_issue("VER", "tire_blister", 35, "performance_loss", 0.8)
    # Weather change
    .add_weather("intermediate", start_lap=20, duration_laps=15)
    # Configuration
    .with_simulations(500)
    .with_seed(123)
    .with_metadata("analyst", "john_doe")
    .with_metadata("priority", "high")
    .build()
)
```

## API Reference

### ScenarioBuilder

Fluent builder for constructing scenarios with method chaining.

#### Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `scenario_id: str, circuit: CircuitContext` | - | Initialize builder |
| `with_drivers` | `drivers: List[DriverState]` | `ScenarioBuilder` | Set drivers |
| `with_description` | `description: str` | `ScenarioBuilder` | Set scenario description |
| `with_type` | `scenario_type: ScenarioType` | `ScenarioBuilder` | Set scenario type |
| `add_grid_change` | `driver_id: str, new_position: int, reason: str = ""` | `ScenarioBuilder` | Add grid position change |
| `add_driver_strategy` | `driver_id: str, strategy: TireStrategy, pit_laps: Optional[List[int]], initial_tire: TireCompound` | `ScenarioBuilder` | Add pit strategy |
| `add_mechanical_issue` | `driver_id: str, issue_type: str, lap: int, severity: str, performance_penalty: Optional[float]` | `ScenarioBuilder` | Add mechanical issue |
| `add_weather` | `condition_type: str, start_lap: int = 1, duration_laps: Optional[int]` | `ScenarioBuilder` | Add weather change |
| `with_simulations` | `n_simulations: int` | `ScenarioBuilder` | Set number of simulations |
| `with_seed` | `seed: int` | `ScenarioBuilder` | Set random seed (for reproducibility) |
| `with_metadata` | `key: str, value: Any` | `ScenarioBuilder` | Add metadata entry |
| `build` | - | `RaceScenario` | Build and return scenario |

### RaceScenario

Immutable scenario definition.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `scenario_id` | `str` | Unique scenario identifier |
| `scenario_type` | `ScenarioType` | Type of scenario |
| `circuit` | `CircuitContext` | Circuit configuration |
| `drivers` | `List[DriverState]` | Participating drivers |
| `description` | `str` | Human-readable description |
| `grid_changes` | `List[GridChange]` | Grid position modifications |
| `driver_strategies` | `List[DriverStrategy]` | Custom pit strategies |
| `mechanical_issues` | `List[DriverMechanicalIssue]` | Mechanical issues |
| `weather_conditions` | `List[WeatherCondition]` | Weather changes |
| `n_simulations` | `int` | Number of simulations to run |
| `random_seed` | `Optional[int]` | Random seed for reproducibility |
| `metadata` | `Dict[str, Any]` | Additional metadata |

#### Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `get_modified_drivers` | - | `List[DriverState]` | Get drivers with grid changes applied |
| `get_driver_strategy` | `driver_id: str` | `Optional[DriverStrategy]` | Get strategy for driver |
| `get_mechanical_issues_for_driver` | `driver_id: str` | `List[DriverMechanicalIssue]` | Get issues for driver |
| `to_dict` | - | `Dict[str, Any]` | Serialize to dictionary |

### ScenarioRepository

File-based scenario storage and retrieval.

#### Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `storage_dir: Path` | - | Initialize repository |
| `save_scenario` | `scenario: RaceScenario` | `Path` | Save scenario to JSON file |
| `load_scenario_dict` | `scenario_id: str` | `Dict[str, Any]` | Load scenario as dictionary |
| `list_scenarios` | - | `List[str]` | List all saved scenario IDs |
| `delete_scenario` | `scenario_id: str` | - | Delete scenario file |

### ScenarioType (Enum)

Types of scenarios for categorization.

| Value | Description |
|-------|-------------|
| `BASELINE` | Original race state (no modifications) |
| `GRID_CHANGE` | Modified starting grid |
| `WEATHER` | Weather condition override |
| `STRATEGY` | Custom pit strategies |
| `MECHANICAL` | Driver mechanical failures |
| `CUSTOM` | Multiple parameter changes |

### GridChange

Grid position modification for a driver.

```python
@dataclass
class GridChange:
    driver_id: str
    new_position: int
    reason: str = ""
```

### DriverStrategy

Custom pit strategy for a driver.

```python
@dataclass
class DriverStrategy:
    driver_id: str
    strategy: TireStrategy
    pit_laps: Optional[List[int]] = None
    initial_tire: TireCompound = TireCompound.SOFT
```

### DriverMechanicalIssue

Mechanical issue affecting a driver.

```python
@dataclass
class DriverMechanicalIssue:
    driver_id: str
    issue_type: str
    lap: int
    severity: str = "dnf"
    performance_penalty: Optional[float] = None
```

**Severity Options:**
- `"dnf"`: Driver Did Not Finish
- `"performance_loss"`: Performance reduction (requires `performance_penalty`)

### WeatherCondition

Weather condition change during race.

```python
@dataclass
class WeatherCondition:
    condition_type: str
    start_lap: int = 1
    duration_laps: Optional[int] = None
```

**Weather Types:**
- `"dry"`: Dry conditions
- `"wet"`: Wet conditions
- `"intermediate"`: Intermediate conditions

## Usage with Simulator

### Running Simulation for Scenario

```python
from f1_predict.simulation import MonteCarloSimulator

# Build scenario
scenario = (
    ScenarioBuilder("test_scenario", circuit)
    .with_drivers(drivers)
    .with_simulations(1000)
    .with_seed(42)
    .build()
)

# Run simulation
simulator = MonteCarloSimulator(
    circuit=scenario.circuit,
    random_state=scenario.random_seed
)

# Use modified drivers from scenario
modified_drivers = scenario.get_modified_drivers()
result = simulator.run_simulations(modified_drivers, scenario.n_simulations)

# Access results
print(f"VER win probability: {result.get_winner_probability('VER')}")
print(f"HAM podium probability: {result.get_podium_probability('HAM')}")
```

## Persistence

### Save and Load Scenarios

```python
from pathlib import Path
from f1_predict.simulation import ScenarioRepository

# Create repository
repo = ScenarioRepository(Path("./scenarios"))

# Save scenario
scenario = ScenarioBuilder("my_scenario", circuit).with_drivers(drivers).build()
repo.save_scenario(scenario)

# Load scenario
scenario_dict = repo.load_scenario_dict("my_scenario")

# List all scenarios
all_scenarios = repo.list_scenarios()
print(f"Saved scenarios: {all_scenarios}")

# Delete scenario
repo.delete_scenario("my_scenario")
```

## Validation

All dataclasses include validation in `__post_init__`:

- **GridChange**: Position must be ≥ 1
- **DriverStrategy**: Pit laps must be ≥ 1
- **DriverMechanicalIssue**: Lap ≥ 1, severity must be "dnf" or "performance_loss", performance_loss requires penalty
- **WeatherCondition**: Valid types (dry, wet, intermediate), start_lap ≥ 1
- **RaceScenario**: Non-empty scenario_id, at least one driver, no duplicate grid positions

## Examples

### Comparing Two Scenarios

```python
# Baseline scenario
baseline = (
    ScenarioBuilder("baseline", circuit)
    .with_drivers(drivers)
    .with_simulations(1000)
    .build()
)

# Grid penalty scenario
penalty = (
    ScenarioBuilder("penalty", circuit)
    .with_drivers(drivers)
    .add_grid_change("VER", 5)
    .with_simulations(1000)
    .with_seed(42)
    .build()
)

# Run both and compare
sim1 = MonteCarloSimulator(circuit=circuit, random_state=42)
sim2 = MonteCarloSimulator(circuit=circuit, random_state=42)

result1 = sim1.run_simulations(baseline.get_modified_drivers(), 1000)
result2 = sim2.run_simulations(penalty.get_modified_drivers(), 1000)

print(f"Baseline VER wins: {result1.get_winner_probability('VER'):.2%}")
print(f"Penalty VER wins: {result2.get_winner_probability('VER'):.2%}")
print(f"Difference: {(result1.get_winner_probability('VER') - result2.get_winner_probability('VER')):.2%}")
```

### Template Scenarios

```python
# Create reusable scenario templates

def create_grid_penalty_scenario(circuit, drivers, penalized_driver, penalty_places):
    """Create a grid penalty scenario."""
    return (
        ScenarioBuilder(
            f"{circuit.circuit_name}_{penalized_driver}_penalty",
            circuit
        )
        .with_drivers(drivers)
        .add_grid_change(
            penalized_driver,
            penalty_places + 1,
            f"{penalty_places}-place penalty"
        )
        .build()
    )

def create_weather_scenario(circuit, drivers, weather_type, start_lap):
    """Create a weather change scenario."""
    return (
        ScenarioBuilder(
            f"{circuit.circuit_name}_{weather_type}_lap{start_lap}",
            circuit
        )
        .with_drivers(drivers)
        .add_weather(weather_type, start_lap=start_lap)
        .build()
    )

# Usage
grid_scenario = create_grid_penalty_scenario(circuit, drivers, "VER", 10)
rain_scenario = create_weather_scenario(circuit, drivers, "wet", 25)
```

## Best Practices

1. **Always Set Seeds**: Use `with_seed()` for reproducible results
2. **Descriptive IDs**: Use clear, parseable scenario IDs (e.g., "bahrain_2024_ver_penalty")
3. **Metadata**: Include analyst name and timestamp for audit trails
4. **Validation**: Check scenario constraints before running simulations
5. **Persistence**: Save important scenarios for later analysis
6. **Batch Operations**: Use repository to manage multiple scenarios
7. **Progressive Building**: Start with baseline, then layer modifications

## Integration with Sensitivity Analysis

```python
# Create scenarios for sensitivity analysis
base_drivers = [...]
scenarios = []

# Vary Verstappen's pace
for pace_delta in [-0.5, -0.25, 0, 0.25, 0.5]:
    modified_drivers = [d.copy() for d in base_drivers]
    modified_drivers[0].expected_lap_time += pace_delta

    scenario = (
        ScenarioBuilder(f"pace_sensitivity_{pace_delta}", circuit)
        .with_drivers(modified_drivers)
        .with_metadata("parameter", "pace_delta")
        .with_metadata("value", pace_delta)
        .build()
    )
    scenarios.append(scenario)
```

## Testing

Scenario builder includes comprehensive test coverage (38 tests):

```bash
# Run scenario builder tests
uv run pytest tests/simulation/test_scenario_builder.py -v

# Run with coverage
uv run pytest tests/simulation/test_scenario_builder.py --cov=src/f1_predict/simulation/analysis
```

## Future Enhancements

Planned features:

- Scenario templates for common what-if analyses
- Scenario comparison utilities
- Confidence interval calculation per scenario
- Scenario versioning and history tracking
- Export/import to CSV and Excel formats
- Visual scenario editor in Streamlit UI
