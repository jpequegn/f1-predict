# Issue #34: Simulation & 'What-If' Analysis Engine - Implementation Plan

## Overview
Build a Monte Carlo simulation engine that enables users to test hypothetical race scenarios and understand how changes to various factors impact outcomes.

## Architecture

### Module Structure
```
src/f1_predict/simulation/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── race_state.py          # RaceState model, lap tracking, position updates
│   ├── driver_state.py        # Driver state, tires, fuel, performance
│   ├── incidents.py           # Safety car, red flags, DNF events
│   └── progression.py         # Lap-by-lap race progression logic
├── engine/
│   ├── __init__.py
│   ├── simulator.py           # Main Monte Carlo simulator
│   ├── pit_strategy.py        # Pit stop logic and tire management
│   ├── weather_model.py       # Weather condition progression
│   └── parallelizer.py        # Parallel execution for 1000+ runs
├── analysis/
│   ├── __init__.py
│   ├── scenario_builder.py    # Scenario configuration and validation
│   ├── sensitivity.py         # Parameter sensitivity analysis
│   ├── statistics.py          # Statistical aggregation and CI calculation
│   └── aggregator.py          # Result aggregation and comparison
└── ui/
    ├── __init__.py
    └── simulation_page.py      # Streamlit UI for simulations
```

## Phase 1: Core Simulation Framework (Week 1-2)

### 1.1 Race State Model (`core/race_state.py`)
- **Classes**: `RaceState`, `DriverState`, `CircuitContext`
- **Responsibilities**:
  - Track current lap, positions, gaps, tire compound, fuel level
  - Manage driver status (DNF reason, pit stop count)
  - Handle position changes and gap calculations
  - Validate state transitions

### 1.2 Race Progression (`core/progression.py`)
- **Classes**: `RaceProgressionEngine`
- **Responsibilities**:
  - Simulate lap-by-lap progression
  - Apply performance deltas based on: tire degradation, fuel load, track conditions
  - Generate probabilistic lap time variations
  - Handle tire strategy decisions

### 1.3 Incident Simulation (`core/incidents.py`)
- **Classes**: `IncidentGenerator`, `IncidentEventQueue`
- **Responsibilities**:
  - Generate safety car probabilities based on circuit/weather
  - Handle DNF events with probabilistic models
  - Track red flags and session state changes
  - Log incident timeline

### 1.4 Monte Carlo Engine (`engine/simulator.py`)
- **Classes**: `MonteCarloSimulator`, `SimulationRun`, `SimulationResult`
- **Responsibilities**:
  - Execute N independent race simulations
  - Seed randomness for reproducibility
  - Aggregate results across runs
  - Calculate confidence intervals and distributions

## Phase 2: Advanced Features (Week 2-3)

### 2.1 Pit Stop Strategy (`engine/pit_strategy.py`)
- **Classes**: `PitStopOptimizer`, `TireStrategy`
- **Responsibilities**:
  - Simulate pit stop timing and execution
  - Model tire compound selection and degradation
  - Calculate optimal stop windows
  - Handle fuel level transitions

### 2.2 Weather Modeling (`engine/weather_model.py`)
- **Classes**: `WeatherSimulator`, `TrackCondition`
- **Responsibilities**:
  - Model weather progression (dry→wet→intermediate)
  - Track temperature impacts on performance
  - Simulate rain/dry conditions
  - Probabilistic weather transitions

### 2.3 Scenario Builder (`analysis/scenario_builder.py`)
- **Classes**: `Scenario`, `ScenarioParameter`, `ScenarioValidator`
- **Responsibilities**:
  - Define scenario parameters (grid changes, weather, strategy)
  - Validate scenario constraints
  - Store/load scenarios
  - Parameter type system (continuous, discrete, categorical)

### 2.4 Sensitivity Analysis (`analysis/sensitivity.py`)
- **Classes**: `SensitivityAnalyzer`, `ParameterEffect`
- **Responsibilities**:
  - One-at-a-time parameter variation
  - Tornado diagrams for parameter importance
  - Interaction analysis between parameters
  - Effect size quantification

## Phase 3: Integration & UI (Week 3-4)

### 3.1 Streamlit Interface (`ui/simulation_page.py`)
- **Features**:
  - Scenario builder UI with parameter sliders
  - Real-time simulation execution progress
  - Result visualization (distributions, comparisons)
  - Export/save scenarios and results

### 3.2 Performance Optimization (`engine/parallelizer.py`)
- **Goals**:
  - Parallel execution of 1000+ simulations
  - Target: <60 seconds total runtime
  - Efficient memory management
  - Progress tracking

## Acceptance Criteria

### Functionality
- [x] Core race state model with accurate state tracking
- [x] Monte Carlo engine simulating 1000+ races in <60s
- [x] Pit stop and tire strategy simulation
- [x] Safety car/incident event generation
- [x] Scenario builder with parameter validation
- [x] Sensitivity analysis across parameters
- [x] Statistical aggregation with confidence intervals
- [x] Streamlit UI for interactive scenarios

### Accuracy
- [x] Simulation results align with historical data (±10% MAE)
- [x] Lap time deltas match actual performance patterns
- [x] Incident probabilities based on historical rates

### Performance
- [x] 1000 simulations in <60 seconds
- [x] Memory efficient (<2GB per simulation run)
- [x] Parallel execution on multi-core
- [x] Progress tracking for long runs

### Testing
- [x] 80%+ unit test coverage
- [x] Integration tests for full simulation pipeline
- [x] Validation against historical race data
- [x] Edge case testing (all drivers DNF, extreme weather, etc.)

## Data Sources & Assumptions

### Historical Data Needed
- Lap time distributions by driver/circuit/weather
- Pit stop time statistics (crew performance)
- Safety car probability by circuit
- DNF rates by cause/season
- Tire degradation curves
- Fuel consumption rates

### Assumptions
- Driver performance variance: 0.2-0.5% per lap
- Safety car probability: 5-15% depending on circuit
- Pit stop duration: 22-45 seconds + tire change (2-4s)
- Tire degradation: Linear model with compound-specific rates
- Weather transitions: Probabilistic based on actual patterns

## Success Metrics

| Metric | Target | Validation |
|--------|--------|-----------|
| Simulation Speed | 1000 runs in <60s | Benchmark on standard hardware |
| Accuracy | ±10% MAE vs historical | Compare predictions to actual races |
| Coverage | 80%+ unit tests | pytest coverage report |
| UI Responsiveness | <2s simulation start | Streamlit latency measurement |
| Memory Usage | <2GB per run | Memory profiling |

## Timeline

**Week 1-2: Core Framework**
- Day 1-2: Race state model
- Day 3-4: Progression engine and incidents
- Day 5-6: Monte Carlo simulator
- Day 7-10: Testing and validation

**Week 2-3: Advanced Features**
- Day 11-12: Pit stop and weather
- Day 13-14: Scenario builder
- Day 15: Sensitivity analysis

**Week 3-4: Integration & Polish**
- Day 18-19: Streamlit UI
- Day 20: Performance optimization
- Day 21: Final testing and documentation

## Dependencies

### Python Packages
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scipy`: Statistical functions
- `joblib` or `ray`: Parallel execution
- `plotly`: Visualization

### Project Dependencies
- Existing prediction models (for baseline performance)
- Historical race data (already collected)
- Data models from `f1_predict.data`

## Future Enhancements

1. **Advanced Weather**: More sophisticated weather transition models
2. **Driver Fatigue**: Incorporate driver performance variance over race distance
3. **Strategic Decisions**: AI decision-making for pit stops, tire compounds
4. **Collaborative Simulations**: Share scenarios between users
5. **Real-time Updates**: Live simulation updates during races
6. **Machine Learning**: Learn incident probabilities from season data
