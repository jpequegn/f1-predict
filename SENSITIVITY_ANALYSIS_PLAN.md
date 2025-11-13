# Phase 2c: Sensitivity Analysis & Confidence Intervals - Implementation Plan

## Overview

Phase 2c implements sensitivity analysis capabilities for the F1 race simulation engine, enabling users to understand how parameter variations affect predictions and to quantify uncertainty in simulation results.

## Goals

1. **Parameter Sensitivity Analysis**: Automate variation of driver/race parameters and measure impact on outcomes
2. **Confidence Intervals**: Calculate statistical bounds on win/podium probabilities
3. **Result Aggregation**: Cross-scenario comparison and meta-analysis
4. **Interpretability**: Clear reporting of sensitivity findings

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│        Sensitivity Analysis Framework (Phase 2c)            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │ SensitivityAnalyzer                                │   │
│  │ - vary_driver_pace(driver_id, deltas)             │   │
│  │ - vary_grid_positions(position_deltas)            │   │
│  │ - vary_pit_strategies(strategies)                 │   │
│  │ - run_sensitivity_study(scenarios, params)        │   │
│  └────────────────────────────────────────────────────┘   │
│           ↓ (creates scenarios)                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │ ParameterSweep                                     │   │
│  │ - define_sweep(param_name, min, max, step)       │   │
│  │ - get_parameter_values()                          │   │
│  │ - generate_scenarios()                            │   │
│  └────────────────────────────────────────────────────┘   │
│           ↓ (executes)                                     │
│  ┌────────────────────────────────────────────────────┐   │
│  │ SensitivityResult                                  │   │
│  │ - base_result: SimulationResult                   │   │
│  │ - sweep_results: Dict[param_value, Result]       │   │
│  │ - sensitivity_metrics: Dict[driver, metrics]      │   │
│  │ - confidence_intervals: Dict[driver, (low, high)]│   │
│  │ - tornado_data: DataFrame for visualization       │   │
│  └────────────────────────────────────────────────────┘   │
│           ↓ (reports)                                      │
│  ┌────────────────────────────────────────────────────┐   │
│  │ SensitivityReport                                  │   │
│  │ - generate_summary()                              │   │
│  │ - get_tornado_chart_data()                        │   │
│  │ - get_sensitivity_table()                         │   │
│  │ - export_to_json()                                │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components to Implement

### 1. Core Sensitivity Analysis Classes

#### `ParameterSweep`
- Define parameter ranges and variations
- Generate parameter values (linear, log-scale, custom)
- Create scenarios for each parameter value
- Track sweep metadata (name, description, parameter bounds)

**Key Methods:**
```python
class ParameterSweep:
    def __init__(self, param_name, param_type, base_value)
    def add_linear_variation(min_val, max_val, num_steps)
    def add_log_variation(min_val, max_val, num_steps)
    def add_custom_values(values)
    def get_parameter_values() -> List[float]
    def apply_to_drivers(drivers, param_values) -> List[List[DriverState]]
```

#### `SensitivityAnalyzer`
- Execute sensitivity studies
- Run simulations across parameter ranges
- Aggregate results across scenarios
- Calculate statistical metrics

**Key Methods:**
```python
class SensitivityAnalyzer:
    def __init__(self, simulator, base_scenario)
    def vary_driver_pace(driver_id, pace_deltas) -> SensitivityResult
    def vary_grid_positions(position_offsets) -> SensitivityResult
    def vary_pit_strategies(strategies) -> SensitivityResult
    def run_parameter_sweep(sweep: ParameterSweep) -> SensitivityResult
    def run_multi_parameter_study(sweeps: List[ParameterSweep]) -> SensitivityResult
```

#### `SensitivityResult`
- Store and organize sensitivity findings
- Calculate confidence intervals
- Compute sensitivity metrics (elasticity, tornado values)
- Support multiple drivers simultaneously

**Key Methods:**
```python
class SensitivityResult:
    def __init__(self, base_result, sweep_results, parameter_name)
    def get_confidence_interval(driver_id, confidence_level=0.95) -> Tuple[float, float]
    def get_elasticity(driver_id) -> float
    def get_sensitivity_metric(driver_id) -> float
    def get_tornado_data() -> Dict[driver, Tuple[low_delta, high_delta]]
    def get_summary() -> Dict[str, Any]
```

#### `SensitivityReport`
- Generate human-readable reports
- Create visualization data structures
- Export results to JSON/CSV

**Key Methods:**
```python
class SensitivityReport:
    def __init__(self, sensitivity_result)
    def generate_summary_text() -> str
    def get_table_data() -> pd.DataFrame
    def get_tornado_chart_data() -> Dict
    def get_sensitivity_plots() -> Dict[driver, List[Tuple[param_val, prob]]]
    def export_json() -> str
```

### 2. Statistical Utilities

#### `ConfidenceIntervalCalculator`
- Bootstrap confidence intervals from simulation runs
- Percentile-based intervals
- Support for different confidence levels (90%, 95%, 99%)

**Key Methods:**
```python
def calculate_confidence_interval(samples, confidence_level=0.95) -> Tuple[float, float]
def bootstrap_confidence_interval(base_result, n_bootstrap=1000) -> Dict[driver, Tuple]
```

### 3. Parameter Variation Strategies

#### Pace Variation
- Vary individual driver lap times (±0.5%, ±1%, ±1.5%)
- Affects expected_lap_time
- Other drivers unchanged

#### Grid Position Variation
- Vary starting grid positions
- One position shifts at a time (tornado analysis)
- All drivers simultaneously (combined effect)

#### Pit Strategy Variation
- Test different pit strategies per driver
- Compare ONE_STOP vs TWO_STOP vs THREE_STOP
- Impact on final result

#### Weather Variation
- Test different weather scenarios
- Dry, wet, intermediate conditions
- Weather start lap and duration

## Implementation Files

### New Files to Create

1. **`sensitivity_analyzer.py`** (Core sensitivity analysis engine)
   - `ParameterSweep` class
   - `SensitivityAnalyzer` class
   - `SensitivityResult` class

2. **`sensitivity_report.py`** (Reporting and visualization data)
   - `SensitivityReport` class
   - Report generation methods

3. **`statistical_utils.py`** (Statistical calculations)
   - Confidence interval calculators
   - Bootstrap resampling
   - Elasticity calculations

4. **`parameter_variations.py`** (Parameter modification strategies)
   - `ParameterVariation` base class
   - `PaceVariation`, `GridVariation`, `StrategyVariation`
   - Custom variation implementations

### Updated Files

- `analysis/__init__.py` - Add new class exports
- `simulation/__init__.py` - Add sensitivity analysis APIs

## Test Strategy

### Test Coverage Goals
- 95%+ code coverage for new modules
- Comprehensive validation of statistical calculations
- Integration tests with simulator

### Test Files to Create

1. **`test_sensitivity_analyzer.py`**
   - Parameter sweep creation
   - Sensitivity analysis execution
   - Result aggregation
   - Confidence interval calculation

2. **`test_sensitivity_report.py`**
   - Report generation
   - Data formatting
   - Export functionality

3. **`test_statistical_utils.py`**
   - Confidence interval calculations
   - Bootstrap resampling
   - Statistical metrics

## Usage Examples

### Example 1: Single Parameter Sensitivity (Pace)

```python
from f1_predict.simulation import MonteCarloSimulator, ScenarioBuilder
from f1_predict.simulation.analysis.sensitivity_analyzer import (
    SensitivityAnalyzer,
    ParameterSweep,
)

# Create base scenario
circuit = CircuitContext(circuit_name="Albert Park", total_laps=58)
drivers = [...]

scenario = ScenarioBuilder("baseline", circuit).with_drivers(drivers).build()

# Create sensitivity analyzer
simulator = MonteCarloSimulator(circuit=circuit, random_state=42)
analyzer = SensitivityAnalyzer(simulator, scenario)

# Analyze pace sensitivity for VER
result = analyzer.vary_driver_pace("VER", pace_deltas=[-1.0, -0.5, 0, 0.5, 1.0])

# Get results
print(f"VER win probability baseline: {result.base_result.get_winner_probability('VER'):.2%}")
print(f"VER win probability at -1% pace: {result.sweep_results[-1.0].get_winner_probability('VER'):.2%}")

# Calculate confidence interval
ci_low, ci_high = result.get_confidence_interval("VER", confidence_level=0.95)
print(f"VER win probability 95% CI: [{ci_low:.2%}, {ci_high:.2%}]")
```

### Example 2: Multi-Parameter Tornado Analysis

```python
# Create sweeps for grid positions (one-at-a-time)
sweeps = []
for i, driver in enumerate(drivers):
    sweep = ParameterSweep(f"grid_position_{driver.driver_id}", "grid", i+1)
    sweep.add_linear_variation(1, 20, 10)
    sweeps.append(sweep)

# Run tornado analysis
results = analyzer.run_multi_parameter_study(sweeps)

# Generate report
report = SensitivityReport(results)
tornado_data = report.get_tornado_chart_data()

# Most impactful parameter?
print(f"Most impactful parameter: {tornado_data['most_impactful']}")
```

### Example 3: Confidence Intervals with Bootstrap

```python
# Run base scenario with high simulation count
scenario = ScenarioBuilder("baseline", circuit).with_drivers(drivers).with_simulations(1000).build()
base_result = simulator.run_simulations(scenario.get_modified_drivers(), 1000)

# Calculate bootstrap confidence intervals
analyzer = SensitivityAnalyzer(simulator, scenario)
ci_dict = analyzer.calculate_bootstrap_confidence_intervals(base_result, n_bootstrap=500)

# Display results
for driver_id, (low, high) in ci_dict.items():
    prob = base_result.get_winner_probability(driver_id)
    print(f"{driver_id}: {prob:.2%} [95% CI: {low:.2%} - {high:.2%}]")
```

## Implementation Timeline

| Task | Estimated Time | Status |
|------|-----------------|--------|
| Design & Planning | ✅ Done | Complete |
| Core Sensitivity Classes | 2-3 hours | Pending |
| Statistical Utilities | 2 hours | Pending |
| Parameter Variations | 1-2 hours | Pending |
| Report Generation | 1-2 hours | Pending |
| Comprehensive Tests (40+ tests) | 3-4 hours | Pending |
| Documentation | 2-3 hours | Pending |
| Integration Testing | 1-2 hours | Pending |
| **Total** | **12-16 hours** | **In Progress** |

## Success Criteria

✅ **Functionality**
- [ ] Single parameter sensitivity analysis working
- [ ] Multi-parameter tornado analysis working
- [ ] Confidence interval calculation accurate
- [ ] Result aggregation correct
- [ ] Report generation complete

✅ **Quality**
- [ ] 95%+ code coverage
- [ ] 40+ test cases, 100% passing
- [ ] Type hints throughout
- [ ] Comprehensive docstrings
- [ ] PEP 8 compliant

✅ **Documentation**
- [ ] User guide with examples
- [ ] API reference
- [ ] Integration guide
- [ ] Interpretation guidelines

✅ **Integration**
- [ ] Works with existing simulator
- [ ] Compatible with scenario builder
- [ ] Seamless API integration
- [ ] Ready for UI integration (Phase 3)

## Deliverables

### Code
- 4 new production modules (~800-1000 lines)
- 3 comprehensive test modules (~400-500 lines)
- Complete type hints and docstrings

### Documentation
- Sensitivity analysis user guide (300+ lines)
- API reference documentation
- Integration examples
- Interpretation guidelines

### Quality Assurance
- 40+ test cases
- 95%+ code coverage
- All tests passing
- Integration validated

## Next Steps After Phase 2c

1. **Phase 3**: Build Streamlit UI with:
   - Interactive sensitivity analysis runner
   - Tornado chart visualization
   - Confidence interval display
   - Export capabilities

2. **Historical Validation**: Test against 2020-2024 races

3. **Performance Optimization**: Optimize for 1000+ simulations

## Notes

- Bootstrap resampling may be computationally intensive; consider parallel execution
- Confidence interval calculation requires sufficient simulation samples (recommend 500+)
- Tornado analysis is 1-at-a-time; combined parameter effects require full factorial
- Statistical validity requires multiple simulation runs with different seeds

---

**Phase 2c Status**: Ready to begin implementation
**Previous Phases**: ✅ Phase 1 & 2 & 2b Complete
**Next Phase**: Phase 3 (Streamlit UI)
