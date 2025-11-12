"""Unit tests for sensitivity analysis."""

import pytest
import numpy as np

from f1_predict.simulation.core.driver_state import DriverState, TireCompound
from f1_predict.simulation.core.race_state import CircuitContext
from f1_predict.simulation.engine.simulator import MonteCarloSimulator
from f1_predict.simulation.analysis.scenario_builder import ScenarioBuilder
from f1_predict.simulation.analysis.sensitivity_analyzer import (
    ParameterSweep,
    ParameterType,
    SensitivityAnalyzer,
    SensitivityResult,
)


class TestParameterSweep:
    """Test ParameterSweep class."""

    def test_parameter_sweep_creation(self):
        """Test creating a parameter sweep."""
        sweep = ParameterSweep(
            param_name="pace_VER",
            param_type=ParameterType.PACE,
            base_value=81.5,
        )
        assert sweep.param_name == "pace_VER"
        assert sweep.param_type == ParameterType.PACE
        assert sweep.base_value == 81.5

    def test_parameter_sweep_linear_variation(self):
        """Test linear variation generation."""
        sweep = ParameterSweep(
            param_name="pace_VER",
            param_type=ParameterType.PACE,
            base_value=81.5,
        ).add_linear_variation(-1.0, 1.0, 5)

        values = sweep.get_parameter_values()
        assert len(values) == 5
        assert values[0] == pytest.approx(-1.0)
        assert values[-1] == pytest.approx(1.0)
        assert values[2] == pytest.approx(0.0)

    def test_parameter_sweep_log_variation(self):
        """Test logarithmic variation generation."""
        sweep = ParameterSweep(
            param_name="pace_VER",
            param_type=ParameterType.PACE,
            base_value=81.5,
        ).add_log_variation(0.1, 10.0, 5)

        values = sweep.get_parameter_values()
        assert len(values) == 5
        assert values[0] == pytest.approx(0.1)
        assert values[-1] == pytest.approx(10.0)

    def test_parameter_sweep_log_invalid_values(self):
        """Test that log scale rejects non-positive values."""
        sweep = ParameterSweep(
            param_name="pace_VER",
            param_type=ParameterType.PACE,
            base_value=81.5,
        )
        with pytest.raises(ValueError):
            sweep.add_log_variation(-1.0, 1.0, 5)

    def test_parameter_sweep_custom_values(self):
        """Test custom value specification."""
        custom_values = [-1.5, -0.5, 0, 0.5, 1.5]
        sweep = ParameterSweep(
            param_name="pace_VER",
            param_type=ParameterType.PACE,
            base_value=81.5,
        ).add_custom_values(custom_values)

        values = sweep.get_parameter_values()
        assert values == sorted(custom_values)

    def test_parameter_sweep_missing_values(self):
        """Test that getting values without setting them raises error."""
        sweep = ParameterSweep(
            param_name="pace_VER",
            param_type=ParameterType.PACE,
            base_value=81.5,
        )
        with pytest.raises(ValueError):
            sweep.get_parameter_values()

    def test_parameter_sweep_invalid_name(self):
        """Test that empty param_name raises error."""
        with pytest.raises(ValueError):
            ParameterSweep(
                param_name="",
                param_type=ParameterType.PACE,
                base_value=81.5,
            )


class TestSensitivityResult:
    """Test SensitivityResult class."""

    @pytest.fixture
    def base_result(self):
        """Create a base simulation result."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=58)
        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
            DriverState("HAM", "Lewis Hamilton", expected_lap_time=82.0),
        ]
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)
        return simulator.run_simulations(drivers, n_simulations=10)

    def test_sensitivity_result_creation(self, base_result):
        """Test creating sensitivity result."""
        result = SensitivityResult(
            parameter_name="pace_VER",
            parameter_type=ParameterType.PACE,
            base_result=base_result,
            parameter_values=[-1.0, 0.0, 1.0],
            drivers=["VER", "HAM"],
        )
        assert result.parameter_name == "pace_VER"
        assert result.parameter_type == ParameterType.PACE
        assert len(result.parameter_values) == 3

    def test_get_confidence_interval(self, base_result):
        """Test confidence interval calculation."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=58)
        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
            DriverState("HAM", "Lewis Hamilton", expected_lap_time=82.0),
        ]
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        sweep_results = {}
        for pace_delta in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            modified_drivers = []
            for d in drivers:
                d_copy = d.copy()
                if d_copy.driver_id == "VER":
                    d_copy.expected_lap_time = 81.5 + (81.5 * pace_delta / 100.0)
                modified_drivers.append(d_copy)

            sweep_results[pace_delta] = simulator.run_simulations(
                modified_drivers, n_simulations=10
            )

        result = SensitivityResult(
            parameter_name="pace_VER",
            parameter_type=ParameterType.PACE,
            base_result=base_result,
            sweep_results=sweep_results,
            parameter_values=[-1.0, -0.5, 0.0, 0.5, 1.0],
            drivers=["VER", "HAM"],
        )

        ci_low, ci_high = result.get_confidence_interval("VER")
        assert ci_low <= ci_high
        assert 0 <= ci_low <= 1
        assert 0 <= ci_high <= 1

    def test_get_elasticity(self, base_result):
        """Test elasticity calculation."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=58)
        drivers = [DriverState("VER", "Max Verstappen", expected_lap_time=81.5)]
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        sweep_results = {}
        for pace_delta in [-1.0, 0.0, 1.0]:
            modified_drivers = []
            for d in drivers:
                d_copy = d.copy()
                d_copy.expected_lap_time = 81.5 + (81.5 * pace_delta / 100.0)
                modified_drivers.append(d_copy)

            sweep_results[pace_delta] = simulator.run_simulations(
                modified_drivers, n_simulations=10
            )

        result = SensitivityResult(
            parameter_name="pace_VER",
            parameter_type=ParameterType.PACE,
            base_result=base_result,
            sweep_results=sweep_results,
            parameter_values=[-1.0, 0.0, 1.0],
            drivers=["VER"],
        )

        elasticity = result.get_elasticity("VER")
        assert isinstance(elasticity, float)

    def test_get_sensitivity_metric(self, base_result):
        """Test sensitivity metric calculation."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=58)
        drivers = [DriverState("VER", "Max Verstappen")]
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        sweep_results = {}
        for pace_delta in [-1.0, 0.0, 1.0]:
            modified_drivers = [d.copy() for d in drivers]
            modified_drivers[0].expected_lap_time = 81.5 + (81.5 * pace_delta / 100.0)

            sweep_results[pace_delta] = simulator.run_simulations(
                modified_drivers, n_simulations=10
            )

        result = SensitivityResult(
            parameter_name="pace_VER",
            parameter_type=ParameterType.PACE,
            base_result=base_result,
            sweep_results=sweep_results,
            parameter_values=[-1.0, 0.0, 1.0],
            drivers=["VER"],
        )

        sensitivity = result.get_sensitivity_metric("VER")
        assert sensitivity >= 0
        assert sensitivity <= 1

    def test_get_tornado_value(self, base_result):
        """Test tornado chart value calculation."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=58)
        drivers = [DriverState("VER", "Max Verstappen")]
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        sweep_results = {}
        for pace_delta in [-1.0, 0.0, 1.0]:
            modified_drivers = [d.copy() for d in drivers]
            modified_drivers[0].expected_lap_time = 81.5 + (81.5 * pace_delta / 100.0)

            sweep_results[pace_delta] = simulator.run_simulations(
                modified_drivers, n_simulations=10
            )

        result = SensitivityResult(
            parameter_name="pace_VER",
            parameter_type=ParameterType.PACE,
            base_result=base_result,
            sweep_results=sweep_results,
            parameter_values=[-1.0, 0.0, 1.0],
            drivers=["VER"],
        )

        neg_impact, pos_impact = result.get_tornado_value("VER")
        assert neg_impact >= 0
        assert pos_impact >= 0

    def test_get_summary(self, base_result):
        """Test summary generation."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=58)
        drivers = [DriverState("VER", "Max Verstappen")]
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        sweep_results = {}
        for pace_delta in [-1.0, 0.0, 1.0]:
            modified_drivers = [d.copy() for d in drivers]
            modified_drivers[0].expected_lap_time = 81.5 + (81.5 * pace_delta / 100.0)

            sweep_results[pace_delta] = simulator.run_simulations(
                modified_drivers, n_simulations=10
            )

        result = SensitivityResult(
            parameter_name="pace_VER",
            parameter_type=ParameterType.PACE,
            base_result=base_result,
            sweep_results=sweep_results,
            parameter_values=[-1.0, 0.0, 1.0],
            drivers=["VER"],
        )

        summary = result.get_summary()
        assert "parameter_name" in summary
        assert "drivers" in summary
        assert "sensitivity_metrics" in summary
        assert "VER" in summary["sensitivity_metrics"]


class TestSensitivityAnalyzer:
    """Test SensitivityAnalyzer class."""

    @pytest.fixture
    def setup(self):
        """Set up test fixtures."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=58)
        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
            DriverState("HAM", "Lewis Hamilton", expected_lap_time=82.0),
        ]
        scenario = ScenarioBuilder("baseline", circuit).with_drivers(drivers).build()
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        return circuit, drivers, scenario, simulator

    def test_sensitivity_analyzer_creation(self, setup):
        """Test creating sensitivity analyzer."""
        circuit, drivers, scenario, simulator = setup

        analyzer = SensitivityAnalyzer(simulator, scenario)
        assert analyzer.base_scenario == scenario
        assert analyzer.simulator == simulator

    def test_run_base_simulation(self, setup):
        """Test running base simulation."""
        circuit, drivers, scenario, simulator = setup

        analyzer = SensitivityAnalyzer(simulator, scenario)
        result = analyzer.run_base_simulation()

        assert result is not None
        assert len(result.finish_probabilities) > 0

    def test_vary_driver_pace(self, setup):
        """Test pace sensitivity analysis."""
        circuit, drivers, scenario, simulator = setup

        analyzer = SensitivityAnalyzer(simulator, scenario, n_simulations=10)
        result = analyzer.vary_driver_pace("VER", pace_deltas=[-1.0, 0.0, 1.0])

        assert result.parameter_type == ParameterType.PACE
        assert len(result.sweep_results) == 3
        assert len(result.parameter_values) == 3

    def test_vary_driver_pace_invalid_driver(self, setup):
        """Test pace sensitivity with invalid driver."""
        circuit, drivers, scenario, simulator = setup

        analyzer = SensitivityAnalyzer(simulator, scenario)
        with pytest.raises(ValueError):
            analyzer.vary_driver_pace("INVALID", pace_deltas=[-1.0, 0.0, 1.0])

    def test_vary_driver_pace_empty_deltas(self, setup):
        """Test pace sensitivity with empty deltas."""
        circuit, drivers, scenario, simulator = setup

        analyzer = SensitivityAnalyzer(simulator, scenario)
        with pytest.raises(ValueError):
            analyzer.vary_driver_pace("VER", pace_deltas=[])

    def test_vary_grid_positions(self, setup):
        """Test grid position sensitivity analysis."""
        circuit, drivers, scenario, simulator = setup

        analyzer = SensitivityAnalyzer(simulator, scenario, n_simulations=10)
        result = analyzer.vary_grid_positions("VER", position_offsets=[-1, 0, 1])

        assert result.parameter_type == ParameterType.GRID

    def test_vary_pit_strategies(self, setup):
        """Test pit strategy sensitivity analysis."""
        from f1_predict.simulation.engine.pit_strategy import TireStrategy

        circuit, drivers, scenario, simulator = setup

        analyzer = SensitivityAnalyzer(simulator, scenario, n_simulations=10)
        result = analyzer.vary_pit_strategies(
            "VER",
            strategies=[TireStrategy.ONE_STOP, TireStrategy.TWO_STOP],
        )

        assert result.parameter_type == ParameterType.STRATEGY
        assert len(result.sweep_results) == 2

    def test_get_confidence_intervals_bootstrap(self, setup):
        """Test bootstrap confidence interval calculation."""
        circuit, drivers, scenario, simulator = setup

        analyzer = SensitivityAnalyzer(simulator, scenario, n_simulations=5)
        ci_dict = analyzer.get_confidence_intervals_bootstrap(n_bootstrap=5)

        assert "VER" in ci_dict
        assert "HAM" in ci_dict

        for driver_id, (ci_low, ci_high) in ci_dict.items():
            assert ci_low <= ci_high
            assert 0 <= ci_low <= 1
            assert 0 <= ci_high <= 1
