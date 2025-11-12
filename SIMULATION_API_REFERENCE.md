# Simulation Engine - API Reference

## Quick Start

```python
from f1_predict.simulation import (
    MonteCarloSimulator,
    DriverState,
    RaceState,
)
from f1_predict.simulation.core import CircuitContext
from f1_predict.simulation.core.driver_state import TireCompound

# Setup circuit
circuit = CircuitContext(
    circuit_name="Albert Park",
    circuit_type="intermediate",
    total_laps=58,
    lap_distance=5.303,
)

# Create drivers
drivers = [
    DriverState(
        driver_id="VER",
        driver_name="Max Verstappen",
        position=1,
        expected_lap_time=81.5,
    ),
    DriverState(
        driver_id="HAM",
        driver_name="Lewis Hamilton",
        position=2,
        expected_lap_time=82.0,
    ),
]

# Run simulations
simulator = MonteCarloSimulator(circuit, random_state=42)
results = simulator.run_simulations(drivers, n_simulations=1000)

# Get results
print(f"Verstappen win probability: {results.get_winner_probability('VER'):.1%}")
print(f"Hamilton podium probability: {results.get_podium_probability('HAM'):.1%}")
```

## Core Module (`simulation.core`)

### DriverState
Tracks individual driver state throughout race.

```python
from f1_predict.simulation.core import DriverState

driver = DriverState(
    driver_id="VER",
    driver_name="Max Verstappen",
    position=1,
    lap=1,
    expected_lap_time=81.5,
    fuel_level=100.0,
)

# Methods
driver.consume_fuel(1.2)
driver.pit_stop(TireCompound.MEDIUM, stop_duration=25.0)
driver.resume_racing()
driver.complete_lap(81.7)
driver.dnf("engine failure")
driver.finish_race()
driver.copy()  # Deep copy

# Properties
driver.is_active  # True if running
driver.is_finished  # True if finished
driver.is_dnf  # True if did not finish
```

### RaceState
Manages complete race state including all drivers.

```python
from f1_predict.simulation.core import RaceState

state = RaceState(circuit=circuit)

# Add/remove drivers
state.add_driver(driver1)
state.add_driver(driver2)

# Query state
active = state.get_active_drivers()  # List[DriverState]
finished = state.get_finished_drivers()  # List[DriverState]
dnf = state.get_dnf_drivers()  # List[DriverState]
leader = state.get_leader()  # Optional[DriverState]

# Update state
state.update_positions()
state.record_lap_snapshot()
state.advance_lap()
state.finish_race()

# Results
results = state.get_race_results()  # List[dict]
state.copy()  # Deep copy
```

### IncidentGenerator
Generates random incidents during simulation.

```python
from f1_predict.simulation.core import IncidentGenerator

gen = IncidentGenerator(
    circuit_type="street",
    random_state=42,
)

# Generate incidents
sc_event = gen.generate_safety_car(lap=25, total_laps=58)
dnf_event = gen.generate_dnf("VER", "Max Verstappen", 30, 58)
weather_event = gen.generate_weather_change(lap=15, current_condition="dry")

# Get incidents
incidents = gen.get_incidents()  # List[IncidentEvent]
gen.clear_incidents()  # Reset
```

## Engine Module (`simulation.engine`)

### MonteCarloSimulator
Main simulator for running race simulations.

```python
from f1_predict.simulation.engine import MonteCarloSimulator

simulator = MonteCarloSimulator(
    circuit=circuit,
    random_state=42,
)

# Single simulation
run = simulator.simulate_race(drivers, run_id=0)

# Multiple simulations
results = simulator.run_simulations(
    drivers=drivers,
    n_simulations=1000,
)
```

### SimulationRun
Result of single race simulation.

```python
from f1_predict.simulation.engine import SimulationRun

run = SimulationRun(
    run_id=0,
    final_positions=["VER", "HAM", "LEC"],
    final_order=[("VER", "Max Verstappen"), ("HAM", "Lewis Hamilton")],
    dnf_drivers=set(),
    race_duration_laps=58,
)

# Properties
run.final_positions  # List[str]
run.final_order  # List[Tuple[str, str]]
run.dnf_drivers  # Set[str]
run.pit_stop_counts  # Dict[str, int]
run.incidents  # List[IncidentEvent]
```

### SimulationResult
Aggregated results from multiple simulations.

```python
from f1_predict.simulation.engine import SimulationResult

results = SimulationResult(n_runs=1000)

# Probabilities
results.finish_probabilities  # Dict[str, float]
results.dnf_rates  # Dict[str, float]
results.average_pit_stops  # Dict[str, float]
results.position_distributions  # Dict[int, Dict[str, float]]

# Methods
results.get_winner_probability("VER")  # float (0.0-1.0)
results.get_podium_probability("HAM")  # float (0.0-1.0)
```

### PitStopOptimizer
Optimize pit stop strategies and tire selection.

```python
from f1_predict.simulation.engine import PitStopOptimizer

optimizer = PitStopOptimizer(
    total_laps=58,
    avg_lap_time=81.5,
    fuel_capacity_laps=60,
)

# Strategy selection
strategy = optimizer.optimize_strategy(fuel_available=100.0)
# TireStrategy.ONE_STOP | TWO_STOP | THREE_STOP | NO_STOP

# Pit windows
windows = optimizer.calculate_pit_windows(strategy)
# List[PitStopWindow] with (start_lap, end_lap, recommended_lap)

# Tire selection
tire = optimizer.select_tire_compound(
    current_lap=20,
    remaining_laps=38,
    weather="dry",
)
# TireCompound.SOFT | MEDIUM | HARD | INTERMEDIATE | WET

# Stint duration
laps = optimizer.calculate_stint_duration(
    tire_compound=TireCompound.MEDIUM,
    fuel_available=80.0,
)

# Time loss estimation
time_loss = optimizer.estimate_time_loss(num_stops=2)  # seconds
```

## Enums & Constants

### TireCompound
```python
from f1_predict.simulation.core.driver_state import TireCompound

TireCompound.SOFT        # Fastest, high degradation
TireCompound.MEDIUM      # Balanced
TireCompound.HARD        # Durable, slow
TireCompound.INTERMEDIATE # Wet conditions
TireCompound.WET         # Heavy rain
```

### DriverStatus
```python
from f1_predict.simulation.core.driver_state import DriverStatus

DriverStatus.RUNNING      # Currently racing
DriverStatus.PIT_STOP     # In pit lane
DriverStatus.DNF          # Did not finish
DriverStatus.FINISHED     # Completed race
```

### TireStrategy
```python
from f1_predict.simulation.engine import TireStrategy

TireStrategy.NO_STOP      # Full race on one set
TireStrategy.ONE_STOP     # One pit stop
TireStrategy.TWO_STOP     # Two pit stops
TireStrategy.THREE_STOP   # Three pit stops
```

### IncidentType
```python
from f1_predict.simulation.core.incidents import IncidentType

IncidentType.SAFETY_CAR       # Safety car deployment
IncidentType.RED_FLAG         # Red flag
IncidentType.DNF_MECHANICAL   # Mechanical failure
IncidentType.DNF_CRASH        # Crash
IncidentType.DNF_OTHER        # Other reason
IncidentType.WEATHER_CHANGE   # Weather change
```

## Data Classes

### CircuitContext
```python
@dataclass
class CircuitContext:
    circuit_name: str
    circuit_type: str = "intermediate"
    total_laps: int = 58
    lap_distance: float = 5.3
    safety_car_prob: float = 0.08
    dnf_rate: float = 0.08
```

### DriverState
```python
@dataclass
class DriverState:
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
    expected_lap_time: float = 90.0
    pace_variance: float = 1.0
    best_lap_time: Optional[float] = None
    pit_stop_durations: list[float] = field(default_factory=list)
```

### PitStopWindow
```python
@dataclass
class PitStopWindow:
    start_lap: int
    end_lap: int
    recommended_lap: int
```

## Example: Full Simulation Workflow

```python
from f1_predict.simulation import (
    MonteCarloSimulator,
    DriverState,
)
from f1_predict.simulation.core import CircuitContext
from f1_predict.simulation.core.driver_state import TireCompound

# 1. Setup circuit
circuit = CircuitContext(
    circuit_name="Monaco",
    circuit_type="street",
    total_laps=78,
    safety_car_prob=0.12,  # Street circuits have higher SC rate
)

# 2. Create drivers
drivers = [
    DriverState("VER", "Max Verstappen", expected_lap_time=76.0),
    DriverState("LEC", "Charles Leclerc", expected_lap_time=76.5),
    DriverState("SAI", "Carlos Sainz", expected_lap_time=77.0),
]

# 3. Configure simulator
simulator = MonteCarloSimulator(
    circuit=circuit,
    random_state=42,  # Reproducible results
)

# 4. Run simulations
results = simulator.run_simulations(
    drivers=drivers,
    n_simulations=1000,
)

# 5. Analyze results
print("Win Probabilities:")
for driver in drivers:
    prob = results.get_winner_probability(driver.driver_id)
    print(f"  {driver.driver_name}: {prob:.1%}")

print("\nPodium Probabilities:")
for driver in drivers:
    prob = results.get_podium_probability(driver.driver_id)
    print(f"  {driver.driver_name}: {prob:.1%}")

print("\nDNF Rates:")
for driver_id, rate in results.dnf_rates.items():
    print(f"  {driver_id}: {rate:.1%}")

# 6. Position distributions
print("\nPosition Distributions:")
for position in [1, 2, 3]:
    if position in results.position_distributions:
        dist = results.position_distributions[position]
        for driver_id, prob in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  P{position}: {driver_id} = {prob:.1%}")
```

## Performance Targets

- **Simulation Speed**: 1000 races in <60 seconds
- **Memory**: <2GB per full simulation run
- **Drivers**: 20 drivers per simulation
- **Laps**: 78-lap races
- **Randomness**: Seeded for reproducibility

## Testing Readiness

All components are designed for comprehensive testing:
- Unit tests for individual components
- Integration tests for full pipeline
- Validation against historical data
- Edge case testing (all DNF, extreme weather, etc.)
