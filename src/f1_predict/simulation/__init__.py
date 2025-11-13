"""Race simulation and what-if analysis engine.

This module provides Monte Carlo simulation capabilities for F1 race predictions,
enabling users to explore hypothetical scenarios and perform sensitivity analysis.

Main Components:
- Core: Race and driver state models
- Engine: Monte Carlo simulator and strategy engines
- Analysis: Scenario building, sensitivity analysis, statistics
- UI: Streamlit interface for interactive simulations
"""

from f1_predict.simulation.core.driver_state import DriverState
from f1_predict.simulation.core.incidents import IncidentEvent, IncidentGenerator
from f1_predict.simulation.core.race_state import RaceState
from f1_predict.simulation.engine.pit_strategy import PitStopOptimizer, TireStrategy
from f1_predict.simulation.engine.simulator import (
    MonteCarloSimulator,
    SimulationResult,
    SimulationRun,
)
from f1_predict.simulation.analysis.scenario_builder import (
    RaceScenario,
    ScenarioBuilder,
    ScenarioRepository,
    ScenarioType,
)
from f1_predict.simulation.analysis.sensitivity_analyzer import (
    SensitivityAnalyzer,
    SensitivityResult,
    ParameterSweep,
    ParameterType,
)
from f1_predict.simulation.analysis.sensitivity_report import (
    SensitivityReport,
    TornadoChartData,
)

__all__ = [
    # Core models
    "RaceState",
    "DriverState",
    "IncidentEvent",
    "IncidentGenerator",
    # Engine
    "MonteCarloSimulator",
    "SimulationRun",
    "SimulationResult",
    "PitStopOptimizer",
    "TireStrategy",
    # Analysis - Scenarios
    "RaceScenario",
    "ScenarioBuilder",
    "ScenarioRepository",
    "ScenarioType",
    # Analysis - Sensitivity
    "SensitivityAnalyzer",
    "SensitivityResult",
    "ParameterSweep",
    "ParameterType",
    "SensitivityReport",
    "TornadoChartData",
]
