"""Sensitivity analysis for race simulations."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np
from scipy import stats

from f1_predict.simulation.core.driver_state import DriverState
from f1_predict.simulation.core.race_state import CircuitContext
from f1_predict.simulation.engine.simulator import MonteCarloSimulator, SimulationResult
from f1_predict.simulation.analysis.scenario_builder import (
    RaceScenario,
    ScenarioBuilder,
    GridChange,
    DriverStrategy,
)
from f1_predict.simulation.engine.pit_strategy import TireStrategy


class ParameterType(Enum):
    """Types of parameters that can be varied."""

    PACE = "pace"  # Driver lap time variation
    GRID = "grid"  # Grid position variation
    STRATEGY = "strategy"  # Pit strategy variation
    WEATHER = "weather"  # Weather condition variation


@dataclass
class ParameterSweep:
    """Definition of parameter variation for sensitivity analysis."""

    param_name: str
    param_type: ParameterType
    base_value: float
    values: List[float] = field(default_factory=list)
    description: str = ""

    def add_linear_variation(
        self, min_val: float, max_val: float, num_steps: int
    ) -> "ParameterSweep":
        """Add linearly spaced parameter values."""
        if num_steps < 2:
            raise ValueError("num_steps must be >= 2")
        self.values = list(np.linspace(min_val, max_val, num_steps))
        return self

    def add_log_variation(
        self, min_val: float, max_val: float, num_steps: int
    ) -> "ParameterSweep":
        """Add logarithmically spaced parameter values."""
        if num_steps < 2:
            raise ValueError("num_steps must be >= 2")
        if min_val <= 0 or max_val <= 0:
            raise ValueError("min_val and max_val must be positive for log scale")
        self.values = list(np.logspace(np.log10(min_val), np.log10(max_val), num_steps))
        return self

    def add_custom_values(self, values: List[float]) -> "ParameterSweep":
        """Add custom parameter values."""
        if not values:
            raise ValueError("values list cannot be empty")
        self.values = sorted(values)
        return self

    def get_parameter_values(self) -> List[float]:
        """Get all parameter values for sweep."""
        if not self.values:
            raise ValueError("Parameter values not set. Call add_linear_variation, add_log_variation, or add_custom_values")
        return self.values

    def __post_init__(self) -> None:
        """Validate sweep configuration."""
        if not self.param_name:
            raise ValueError("param_name required")


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis."""

    parameter_name: str
    parameter_type: ParameterType
    base_result: SimulationResult
    sweep_results: Dict[float, SimulationResult] = field(default_factory=dict)
    parameter_values: List[float] = field(default_factory=list)
    drivers: List[str] = field(default_factory=list)

    def get_win_probability_by_parameter(
        self, driver_id: str
    ) -> Dict[float, float]:
        """Get win probability for each parameter value."""
        result = {}
        for param_val, sim_result in self.sweep_results.items():
            result[param_val] = sim_result.get_winner_probability(driver_id)
        return result

    def get_podium_probability_by_parameter(
        self, driver_id: str
    ) -> Dict[float, float]:
        """Get podium probability for each parameter value."""
        result = {}
        for param_val, sim_result in self.sweep_results.items():
            result[param_val] = sim_result.get_podium_probability(driver_id)
        return result

    def get_confidence_interval(
        self, driver_id: str, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval from sweep results."""
        if not self.sweep_results:
            raise ValueError("No sweep results available")

        probabilities = list(
            self.get_win_probability_by_parameter(driver_id).values()
        )

        # Use percentile method for confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower = np.percentile(probabilities, lower_percentile)
        upper = np.percentile(probabilities, upper_percentile)

        return (float(lower), float(upper))

    def get_elasticity(self, driver_id: str) -> float:
        """Calculate elasticity: % change in win probability / % change in parameter."""
        if len(self.parameter_values) < 2:
            raise ValueError("Need at least 2 parameter values for elasticity calculation")

        probs = self.get_win_probability_by_parameter(driver_id)

        # Use first and last values
        param_min = self.parameter_values[0]
        param_max = self.parameter_values[-1]
        prob_min = probs[param_min]
        prob_max = probs[param_max]

        # Handle edge case: param_min is 0 or very close to 0
        if abs(param_min) < 1e-6:
            # Use absolute change instead
            param_change = param_max - param_min
        else:
            param_change = ((param_max - param_min) / param_min) * 100

        # Handle edge case: prob_min is 0 or very close to 0
        if prob_min < 1e-6:
            # If probability is near zero, return large elasticity value
            return 100.0 if prob_max > prob_min else 0.0

        prob_change_pct = ((prob_max - prob_min) / prob_min) * 100

        if abs(param_change) < 1e-6:
            return 0.0

        elasticity = prob_change_pct / param_change
        return float(elasticity)

    def get_sensitivity_metric(self, driver_id: str) -> float:
        """Get sensitivity metric: range of win probabilities."""
        probs = list(
            self.get_win_probability_by_parameter(driver_id).values()
        )
        return float(max(probs) - min(probs))

    def get_tornado_value(self, driver_id: str) -> Tuple[float, float]:
        """Get tornado chart values: (negative_impact, positive_impact)."""
        base_prob = self.base_result.get_winner_probability(driver_id)
        probs = list(
            self.get_win_probability_by_parameter(driver_id).values()
        )

        min_prob = min(probs)
        max_prob = max(probs)

        # Negative impact: how much probability can decrease
        negative_impact = max(0.0, base_prob - min_prob)
        # Positive impact: how much probability can increase
        positive_impact = max(0.0, max_prob - base_prob)

        return (float(negative_impact), float(positive_impact))

    def get_summary(self) -> Dict[str, Any]:
        """Generate summary of sensitivity results."""
        summary = {
            "parameter_name": self.parameter_name,
            "parameter_type": self.parameter_type.value,
            "parameter_values": self.parameter_values,
            "drivers": self.drivers,
            "sensitivity_metrics": {},
        }

        for driver_id in self.drivers:
            base_prob = self.base_result.get_winner_probability(driver_id)
            ci_low, ci_high = self.get_confidence_interval(driver_id)
            elasticity = self.get_elasticity(driver_id)
            sensitivity = self.get_sensitivity_metric(driver_id)
            tornado_neg, tornado_pos = self.get_tornado_value(driver_id)

            summary["sensitivity_metrics"][driver_id] = {
                "base_probability": float(base_prob),
                "confidence_interval_95": {
                    "lower": float(ci_low),
                    "upper": float(ci_high),
                },
                "elasticity": float(elasticity),
                "sensitivity_metric": float(sensitivity),
                "tornado_values": {
                    "negative_impact": float(tornado_neg),
                    "positive_impact": float(tornado_pos),
                },
            }

        return summary


class SensitivityAnalyzer:
    """Analyze sensitivity of race outcomes to parameter variations."""

    def __init__(
        self,
        simulator: MonteCarloSimulator,
        base_scenario: RaceScenario,
        n_simulations: Optional[int] = None,
    ):
        """Initialize sensitivity analyzer."""
        self.simulator = simulator
        self.base_scenario = base_scenario
        self.n_simulations = n_simulations or base_scenario.n_simulations
        self.random_seed = base_scenario.random_seed

    def run_base_simulation(self) -> SimulationResult:
        """Run simulation for base scenario."""
        modified_drivers = self.base_scenario.get_modified_drivers()
        result = self.simulator.run_simulations(
            modified_drivers,
            n_simulations=self.n_simulations,
        )
        return result

    def vary_driver_pace(
        self,
        driver_id: str,
        pace_deltas: List[float],
        description: str = "",
    ) -> SensitivityResult:
        """Run sensitivity analysis varying a driver's pace.

        Args:
            driver_id: Driver to vary
            pace_deltas: Pace variations in % (e.g., [-1.0, -0.5, 0, 0.5, 1.0])
            description: Optional description

        Returns:
            SensitivityResult with sweep results
        """
        if not pace_deltas:
            raise ValueError("pace_deltas cannot be empty")

        # Run base simulation
        base_result = self.run_base_simulation()

        # Find driver's base pace
        base_driver = next((d for d in self.base_scenario.drivers if d.driver_id == driver_id), None)
        if not base_driver:
            raise ValueError(f"Driver {driver_id} not found in scenario")

        base_pace = base_driver.expected_lap_time

        # Run simulations for each pace delta
        sweep_results = {}
        for delta in sorted(pace_deltas):
            # Create modified drivers with adjusted pace
            modified_drivers = []
            for driver in self.base_scenario.get_modified_drivers():
                driver_copy = driver.copy()
                if driver_copy.driver_id == driver_id:
                    # Apply pace change
                    pace_change = base_pace * (delta / 100.0)
                    driver_copy.expected_lap_time = base_pace + pace_change
                modified_drivers.append(driver_copy)

            # Run simulation
            result = self.simulator.run_simulations(
                modified_drivers,
                n_simulations=self.n_simulations,
            )
            sweep_results[delta] = result

        # Create sensitivity result
        return SensitivityResult(
            parameter_name=f"pace_{driver_id}",
            parameter_type=ParameterType.PACE,
            base_result=base_result,
            sweep_results=sweep_results,
            parameter_values=sorted(pace_deltas),
            drivers=[d.driver_id for d in self.base_scenario.drivers],
        )

    def vary_grid_positions(
        self,
        driver_id: str,
        position_offsets: List[int],
        description: str = "",
    ) -> SensitivityResult:
        """Run sensitivity analysis varying a driver's grid position.

        Args:
            driver_id: Driver to vary
            position_offsets: Position offsets (e.g., [-3, -2, -1, 0, 1, 2, 3])
            description: Optional description

        Returns:
            SensitivityResult with sweep results
        """
        if not position_offsets:
            raise ValueError("position_offsets cannot be empty")

        # Run base simulation
        base_result = self.run_base_simulation()

        # Find driver's base position
        base_driver = next((d for d in self.base_scenario.drivers if d.driver_id == driver_id), None)
        if not base_driver:
            raise ValueError(f"Driver {driver_id} not found in scenario")

        base_position = base_driver.position

        # Run simulations for each position offset
        sweep_results = {}
        for offset in sorted(position_offsets):
            new_position = base_position + offset

            # Validate position
            if new_position < 1 or new_position > len(self.base_scenario.drivers):
                # Skip invalid positions
                continue

            # Create scenario with grid change
            scenario = ScenarioBuilder(
                f"grid_sensitivity_{driver_id}_{offset}",
                self.base_scenario.circuit,
            ).with_drivers(self.base_scenario.drivers).add_grid_change(
                driver_id, new_position, f"Offset: {offset:+d}"
            ).build()

            modified_drivers = scenario.get_modified_drivers()

            # Run simulation
            result = self.simulator.run_simulations(
                modified_drivers,
                n_simulations=self.n_simulations,
            )
            sweep_results[float(offset)] = result

        # Create sensitivity result
        return SensitivityResult(
            parameter_name=f"grid_position_{driver_id}",
            parameter_type=ParameterType.GRID,
            base_result=base_result,
            sweep_results=sweep_results,
            parameter_values=sorted([o for o in position_offsets if o in [x - base_position for x in range(1, len(self.base_scenario.drivers) + 1)]]),
            drivers=[d.driver_id for d in self.base_scenario.drivers],
        )

    def vary_pit_strategies(
        self,
        driver_id: str,
        strategies: List[TireStrategy],
        description: str = "",
    ) -> SensitivityResult:
        """Run sensitivity analysis varying pit strategy.

        Args:
            driver_id: Driver to vary
            strategies: List of pit strategies to test
            description: Optional description

        Returns:
            SensitivityResult with sweep results
        """
        if not strategies:
            raise ValueError("strategies cannot be empty")

        # Run base simulation
        base_result = self.run_base_simulation()

        # Run simulations for each strategy
        sweep_results = {}
        for idx, strategy in enumerate(strategies):
            # Create scenario with strategy change
            scenario = ScenarioBuilder(
                f"strategy_sensitivity_{driver_id}_{strategy.value}",
                self.base_scenario.circuit,
            ).with_drivers(
                self.base_scenario.drivers
            ).add_driver_strategy(
                driver_id, strategy
            ).build()

            modified_drivers = scenario.get_modified_drivers()

            # Run simulation
            result = self.simulator.run_simulations(
                modified_drivers,
                n_simulations=self.n_simulations,
            )
            sweep_results[float(idx)] = result

        # Create sensitivity result
        return SensitivityResult(
            parameter_name=f"pit_strategy_{driver_id}",
            parameter_type=ParameterType.STRATEGY,
            base_result=base_result,
            sweep_results=sweep_results,
            parameter_values=list(range(len(strategies))),
            drivers=[d.driver_id for d in self.base_scenario.drivers],
        )

    def run_parameter_sweep(self, sweep: ParameterSweep) -> SensitivityResult:
        """Run sensitivity analysis for a parameter sweep.

        Args:
            sweep: ParameterSweep definition

        Returns:
            SensitivityResult with sweep results
        """
        if sweep.param_type == ParameterType.PACE:
            # Extract driver_id from parameter name
            driver_id = sweep.param_name.split("_")[-1]
            return self.vary_driver_pace(driver_id, sweep.get_parameter_values())

        elif sweep.param_type == ParameterType.GRID:
            driver_id = sweep.param_name.split("_")[-1]
            offsets = [int(v - sweep.base_value) for v in sweep.get_parameter_values()]
            return self.vary_grid_positions(driver_id, offsets)

        elif sweep.param_type == ParameterType.STRATEGY:
            raise NotImplementedError("Strategy sweep via ParameterSweep not yet implemented")

        else:
            raise ValueError(f"Unknown parameter type: {sweep.param_type}")

    def get_confidence_intervals_bootstrap(
        self,
        n_bootstrap: int = 500,
        confidence_level: float = 0.95,
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for base scenario.

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95)

        Returns:
            Dict mapping driver_id to (lower, upper) CI bounds
        """
        result = self.run_base_simulation()

        ci_dict = {}
        for driver_id in [d.driver_id for d in self.base_scenario.drivers]:
            base_prob = result.get_winner_probability(driver_id)

            # Generate bootstrap samples
            bootstrap_probs = []
            for _ in range(n_bootstrap):
                sim_result = self.simulator.run_simulations(
                    self.base_scenario.get_modified_drivers(),
                    n_simulations=max(10, self.n_simulations // 10),  # Smaller samples for bootstrap
                )
                bootstrap_probs.append(sim_result.get_winner_probability(driver_id))

            # Calculate confidence interval
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower = np.percentile(bootstrap_probs, lower_percentile)
            upper = np.percentile(bootstrap_probs, upper_percentile)

            ci_dict[driver_id] = (float(lower), float(upper))

        return ci_dict
