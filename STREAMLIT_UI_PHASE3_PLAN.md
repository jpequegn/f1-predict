# Phase 3: Streamlit UI for Simulation Interface - Implementation Plan

## Overview

Phase 3 builds a comprehensive Streamlit web interface for the F1 race simulation engine, enabling users to interactively configure races, run simulations, and analyze sensitivity results without writing code.

## Goals

1. **Interactive Simulation Configuration**: User-friendly setup of races, drivers, and parameters
2. **Real-Time Simulation Execution**: Run simulations with progress feedback
3. **Sensitivity Analysis Interface**: Interactive parameter variation and result analysis
4. **Visualization Dashboard**: Tornado charts, probability curves, confidence intervals
5. **Result Management**: Export, compare, and share simulation results

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Streamlit UI Layer (Phase 3)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Main App (app.py)                                 │   │
│  │ - Router/Navigation                               │   │
│  │ - Session state management                        │   │
│  │ - Configuration persistence                       │   │
│  └────────────────────────────────────────────────────┘   │
│           ↓                                                 │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Page Modules                                       │   │
│  ├────────────────────────────────────────────────────┤   │
│  │ 1. pages/01_simulator.py                          │   │
│  │    - Race configuration (circuit, weather)         │   │
│  │    - Driver management (setup, pace, strategies)   │   │
│  │    - Simulation execution                          │   │
│  │                                                    │   │
│  │ 2. pages/02_sensitivity.py                        │   │
│  │    - Parameter sweep configuration                 │   │
│  │    - Sensitivity runner                            │   │
│  │    - Result analysis                               │   │
│  │                                                    │   │
│  │ 3. pages/03_results.py                            │   │
│  │    - Results dashboard                             │   │
│  │    - Visualization (charts, tables)                │   │
│  │    - Export functionality                          │   │
│  │                                                    │   │
│  │ 4. pages/04_compare.py                            │   │
│  │    - Compare multiple runs                         │   │
│  │    - Scenario comparison                           │   │
│  │    - Diff visualization                            │   │
│  └────────────────────────────────────────────────────┘   │
│           ↓                                                 │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Visualization Components (components/)             │   │
│  ├────────────────────────────────────────────────────┤   │
│  │ - tornado_chart.py: Tornado chart rendering        │   │
│  │ - probability_curves.py: Win/podium curves        │   │
│  │ - sensitivity_table.py: Sensitivity metrics table │   │
│  │ - race_visualization.py: Grid/lap visualization   │   │
│  │ - confidence_bands.py: CI visualization           │   │
│  └────────────────────────────────────────────────────┘   │
│           ↓                                                 │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Business Logic / Utils (utils/)                   │   │
│  ├────────────────────────────────────────────────────┤   │
│  │ - simulation_manager.py: Orchestrate simulations   │   │
│  │ - state_manager.py: Session state handling         │   │
│  │ - export_manager.py: CSV/PDF export               │   │
│  │ - cache_manager.py: Result caching                │   │
│  │ - validators.py: Input validation                 │   │
│  └────────────────────────────────────────────────────┘   │
│           ↓ (uses)                                         │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Existing Simulation Engine (Phase 1-2c)          │   │
│  ├────────────────────────────────────────────────────┤   │
│  │ - MonteCarloSimulator                              │   │
│  │ - ScenarioBuilder                                  │   │
│  │ - SensitivityAnalyzer                              │   │
│  │ - SensitivityReport                                │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
src/f1_predict/web/
├── pages/
│   ├── 01_simulator.py         # Main simulation interface
│   ├── 02_sensitivity.py       # Sensitivity analysis interface
│   ├── 03_results.py           # Results dashboard
│   └── 04_compare.py           # Comparison interface
├── components/
│   ├── __init__.py
│   ├── tornado_chart.py        # Tornado chart component
│   ├── probability_curves.py   # Probability curve component
│   ├── sensitivity_table.py    # Sensitivity metrics table
│   ├── race_visualization.py   # Race grid/lap visualization
│   └── confidence_bands.py     # Confidence interval bands
├── utils/
│   ├── __init__.py
│   ├── simulation_manager.py   # Simulation orchestration
│   ├── state_manager.py        # Session state management
│   ├── export_manager.py       # Export to CSV/PDF/JSON
│   ├── cache_manager.py        # Result caching
│   └── validators.py           # Input validation
├── app.py                      # Main Streamlit app
└── config.py                   # Configuration constants

tests/web/
├── __init__.py
├── test_simulator_page.py      # Test 01_simulator.py
├── test_sensitivity_page.py    # Test 02_sensitivity.py
├── test_results_page.py        # Test 03_results.py
├── test_compare_page.py        # Test 04_compare.py
├── components/
│   ├── test_tornado_chart.py
│   ├── test_probability_curves.py
│   ├── test_sensitivity_table.py
│   └── test_confidence_bands.py
└── utils/
    ├── test_simulation_manager.py
    ├── test_state_manager.py
    ├── test_export_manager.py
    └── test_cache_manager.py
```

## Implementation Plan

### Phase 3.1: Core Infrastructure (Week 1)

**Files to Create**:
1. `src/f1_predict/web/app.py` - Main Streamlit app with routing
2. `src/f1_predict/web/config.py` - Configuration constants
3. `src/f1_predict/web/utils/simulation_manager.py` - Orchestration logic
4. `src/f1_predict/web/utils/state_manager.py` - Session state handling
5. `src/f1_predict/web/utils/validators.py` - Input validation

**Key Components**:
- Multi-page Streamlit app with navigation
- Session state management for configuration persistence
- Input validation for all user inputs
- Error handling and user feedback

**Tests**: 15-20 tests covering routing, state management, validation

### Phase 3.2: Simulator Page (Week 1-2)

**Files to Create**:
1. `src/f1_predict/web/pages/01_simulator.py` - Main simulator interface
2. `src/f1_predict/web/components/race_visualization.py` - Grid visualization

**Features**:
- Circuit selection (dropdown of F1 circuits)
- Weather configuration (dry/wet, temperature)
- Driver management (add/remove, name, pace setup)
- Strategy configuration (pit stops, tire compounds)
- Real-time progress feedback during simulation
- Results summary display

**Layout**:
```
┌─────────────────────────────────────────┐
│         Race Configuration              │
├──────────────┬──────────────────────────┤
│ Circuit:     │ [Circuit Selector]       │
│ Weather:     │ [Weather Options]        │
│ Drivers:     │ [Driver Management]      │
│ Strategy:    │ [Strategy Config]        │
│ Simulations: │ [Num Simulations: 1000]  │
├─────────────────────────────────────────┤
│         [Run Simulation Button]          │
├─────────────────────────────────────────┤
│         Simulation Progress              │
│ [Progress Bar: 45/1000]                 │
├─────────────────────────────────────────┤
│         Results Summary                  │
│ Driver  │ Win Prob │ Podium Prob │ ...  │
├─────────────────────────────────────────┤
│   [View Details] [Export] [Compare]      │
└─────────────────────────────────────────┘
```

**Tests**: 25-30 tests for UI components, integration with simulator

### Phase 3.3: Sensitivity Analysis Page (Week 2-3)

**Files to Create**:
1. `src/f1_predict/web/pages/02_sensitivity.py` - Sensitivity interface
2. `src/f1_predict/web/components/tornado_chart.py` - Tornado visualization
3. `src/f1_predict/web/components/probability_curves.py` - Curve visualization

**Features**:
- Parameter selection (pace, grid, strategy, weather)
- Sweep configuration (linear, log, custom values)
- Sensitivity runner with progress
- Tornado chart visualization
- Confidence interval display
- Probability curves (win/podium)
- Elasticity ranking

**Layout**:
```
┌──────────────────────────────────────────┐
│    Sensitivity Analysis Configuration    │
├──────────────────┬──────────────────────┤
│ Parameter Type:  │ [Pace/Grid/Strategy] │
│ Target Driver:   │ [Driver Selector]    │
│ Sweep Type:      │ [Linear/Log/Custom]  │
│ Min Value:       │ [Input Field]        │
│ Max Value:       │ [Input Field]        │
│ Num Steps:       │ [Slider 5-21]        │
├──────────────────────────────────────────┤
│    [Run Sensitivity Analysis Button]     │
├──────────────────────────────────────────┤
│         Analysis Results                 │
│  [Tornado Chart] │ [Probability Curves]  │
│                  │ [Confidence Intervals]│
│                  │ [Elasticity Ranking] │
├──────────────────────────────────────────┤
│    [Export Analysis] [Compare Drivers]   │
└──────────────────────────────────────────┘
```

**Tests**: 20-25 tests for sensitivity UI, visualization components

### Phase 3.4: Results & Export (Week 3)

**Files to Create**:
1. `src/f1_predict/web/pages/03_results.py` - Results dashboard
2. `src/f1_predict/web/utils/export_manager.py` - Export functionality
3. `src/f1_predict/web/components/sensitivity_table.py` - Table component

**Features**:
- Results history/caching
- Multiple visualization formats
- Export to CSV, JSON, PDF
- Comparison utilities
- Results persistence

**Tests**: 15-20 tests for results display, export

### Phase 3.5: Comparison Interface (Week 4)

**Files to Create**:
1. `src/f1_predict/web/pages/04_compare.py` - Comparison page
2. `src/f1_predict/web/components/confidence_bands.py` - CI visualization

**Features**:
- Compare multiple simulation runs
- Side-by-side sensitivity results
- Scenario comparison
- Difference highlighting

**Tests**: 10-15 tests for comparison functionality

### Phase 3.6: Testing & Polish (Week 4)

**Activities**:
- Write comprehensive test suite (80+ tests total)
- Performance optimization
- UI/UX refinement
- Documentation

**Tests**: 80+ total tests covering all UI components and integration

## Key Implementation Details

### Session State Management

```python
# Use Streamlit's session_state for:
- Current race configuration (circuit, drivers, weather)
- Last simulation result
- Sensitivity analysis results
- User preferences (export format, display options)
- Cached results history

st.session_state.current_race
st.session_state.last_result
st.session_state.sensitivity_result
st.session_state.cache
```

### Progress Feedback

```python
# For long-running operations (simulations, sensitivity)
progress_bar = st.progress(0)
status_text = st.empty()

for i in range(n_sims):
    # Run simulation step
    progress_bar.progress((i+1) / n_sims)
    status_text.text(f"Running simulation {i+1}/{n_sims}...")
```

### Error Handling

```python
# Wrap simulation calls with error handling
try:
    result = simulator.run_simulations(...)
except ValueError as e:
    st.error(f"Invalid configuration: {str(e)}")
except Exception as e:
    st.error(f"Simulation failed: {str(e)}")
    logger.exception("Simulation error")
```

## Technology Stack

- **Framework**: Streamlit 1.28+
- **Visualization**: Plotly (interactive charts)
- **Data Processing**: Pandas, NumPy
- **Export**: CSV (pandas), JSON, PDF (reportlab)
- **Testing**: Pytest + streamlit.testing.v1

## Success Criteria

1. ✅ All 4 pages functional and integrated
2. ✅ Real-time progress feedback on simulations
3. ✅ Tornado charts rendering correctly
4. ✅ Confidence intervals displayed properly
5. ✅ Export to multiple formats working
6. ✅ Session state persisting across interactions
7. ✅ 80+ tests with >85% coverage for UI components
8. ✅ Performance: <5s for 100-sim runs, <30s for sensitivity analysis
9. ✅ Comprehensive documentation and user guide
10. ✅ Full integration with Phase 1-2c simulation engine

## Integration Points

- **MonteCarloSimulator**: Core simulation execution
- **ScenarioBuilder**: Scenario configuration from UI inputs
- **SensitivityAnalyzer**: Sensitivity runs initiated from page
- **SensitivityReport**: Report generation for display
- **CircuitContext, DriverState**: Data models for UI configuration

## Deliverables

### Code
- 5 page modules (600+ lines)
- 5+ component modules (400+ lines)
- 3 utility modules (300+ lines)
- 80+ tests (1200+ lines)

### Documentation
- `STREAMLIT_UI_PHASE3_PLAN.md` (this file)
- `SIMULATION_PROGRESS_PHASE3.md` (completion report)
- User guide and API reference

### Quality
- 100% test pass rate
- >85% code coverage
- All input validation working
- Performance benchmarks met

## Timeline

**Week 1**: Infrastructure + Simulator page
**Week 2**: Sensitivity page
**Week 3**: Results + Export
**Week 4**: Comparison + Testing + Polish

**Overall**: Phase 3 estimated at 3-4 weeks (medium complexity)

## Next After Phase 3

**Phase 3b: Validation & Optimization**
- Validate simulation accuracy against historical races
- Performance optimization for 1000+ simulations
- Production readiness checklist
- Performance tuning to meet <60s benchmark for 1000 sims

