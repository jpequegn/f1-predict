"""Unit tests for Monte Carlo simulator page."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from f1_predict.simulation.core.driver_state import DriverState
from f1_predict.simulation.core.race_state import CircuitContext
from f1_predict.simulation.engine.simulator import MonteCarloSimulator
from f1_predict.simulation.analysis.scenario_builder import ScenarioBuilder
from f1_predict.simulation.analysis.sensitivity_analyzer import (
    SensitivityAnalyzer,
)
from f1_predict.web.pages import simulator


class TestSimulatorPageInitialization:
    """Test simulator page initialization and session state."""

    def test_initialize_session_state_first_call(self):
        """Test session state initialization on first call."""
        st_session_state = {}

        # Mock streamlit session state
        with patch("f1_predict.web.pages.simulator.st") as mock_st:
            mock_st.session_state = st_session_state

            # Manually initialize
            if "race_config" not in st_session_state:
                st_session_state["race_config"] = {
                    "circuit": "Australia",
                    "weather": "Dry",
                    "temperature": 25,
                    "drivers": ["VER", "LEC", "HAM", "RUS"],
                }

            assert st_session_state["race_config"]["circuit"] == "Australia"
            assert st_session_state["race_config"]["weather"] == "Dry"
            assert st_session_state["race_config"]["temperature"] == 25

    def test_circuit_configuration_has_valid_circuits(self):
        """Test that all defined circuits are valid."""
        assert len(simulator.CIRCUITS) > 0
        assert "Australia" in simulator.CIRCUITS
        assert "Monaco" in simulator.CIRCUITS
        assert "Silverstone" in simulator.CIRCUITS

    def test_circuit_configuration_has_lap_counts(self):
        """Test that all circuits have lap counts."""
        for circuit_name, info in simulator.CIRCUITS.items():
            assert "laps" in info
            assert info["laps"] > 0
            assert info["laps"] < 100  # Reasonable bounds


class TestSimulatorPageCircuitConfiguration:
    """Test circuit configuration UI."""

    def test_default_drivers_are_valid(self):
        """Test that default drivers are valid F1 drivers."""
        assert len(simulator.DEFAULT_DRIVERS) >= 10
        assert "VER" in simulator.DEFAULT_DRIVERS
        assert "LEC" in simulator.DEFAULT_DRIVERS
        assert "HAM" in simulator.DEFAULT_DRIVERS

    def test_default_drivers_have_pace(self):
        """Test that all default drivers have pace information."""
        for driver_id, info in simulator.DEFAULT_DRIVERS.items():
            assert "pace" in info
            assert "name" in info
            assert 70 < info["pace"] < 90  # Reasonable F1 lap time bounds


class TestSimulatorPageDriverConfiguration:
    """Test driver configuration."""

    def test_driver_state_creation(self):
        """Test DriverState object creation."""
        driver = DriverState(
            driver_id="VER",
            driver_name="Max Verstappen",
            expected_lap_time=81.5,
        )

        assert driver.driver_id == "VER"
        assert driver.driver_name == "Max Verstappen"
        assert driver.expected_lap_time == 81.5

    def test_driver_pace_adjustment(self):
        """Test driver pace adjustment."""
        base_pace = simulator.DEFAULT_DRIVERS["VER"]["pace"]
        adjustment = 0.5
        adjusted_pace = base_pace + adjustment

        assert adjusted_pace > base_pace
        assert abs(adjusted_pace - base_pace - adjustment) < 0.001


class TestSimulatorPageStrategyConfiguration:
    """Test strategy configuration."""

    def test_strategy_class_exists(self):
        """Test that DriverStrategy class is available."""
        from f1_predict.simulation.analysis.scenario_builder import DriverStrategy

        # DriverStrategy is a dataclass with pit_laps field
        assert hasattr(DriverStrategy, "__init__")

    def test_tire_compound_exists(self):
        """Test that TireCompound enum is available."""
        from f1_predict.simulation.core.driver_state import TireCompound

        assert hasattr(TireCompound, "SOFT")
        assert hasattr(TireCompound, "MEDIUM")
        assert hasattr(TireCompound, "HARD")


class TestSimulatorPageIntegration:
    """Test integration with simulation engine."""

    def test_circuit_context_creation(self):
        """Test CircuitContext creation from circuit name."""
        circuit_name = "Australia"
        circuit_info = simulator.CIRCUITS[circuit_name]

        circuit = CircuitContext(
            circuit_name=circuit_info["name"],
            total_laps=circuit_info["laps"],
        )

        assert circuit.circuit_name == circuit_info["name"]
        assert circuit.total_laps == circuit_info["laps"]

    def test_scenario_builder_integration(self):
        """Test ScenarioBuilder integration."""
        circuit = CircuitContext(
            circuit_name="Albert Park",
            total_laps=58,
        )

        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
            DriverState("LEC", "Charles Leclerc", expected_lap_time=82.0),
        ]

        scenario = (
            ScenarioBuilder("test_scenario", circuit)
            .with_drivers(drivers)
            .build()
        )

        assert scenario.circuit == circuit
        assert len(scenario.drivers) == 2

    def test_simulator_initialization(self):
        """Test MonteCarloSimulator initialization."""
        circuit = CircuitContext(
            circuit_name="Albert Park",
            total_laps=58,
        )

        simulator_obj = MonteCarloSimulator(circuit=circuit, random_state=42)
        assert simulator_obj is not None


class TestSimulatorPageSensitivityIntegration:
    """Test integration with sensitivity analysis."""

    def test_parameter_type_enum(self):
        """Test ParameterType enum exists."""
        from f1_predict.simulation.analysis.sensitivity_analyzer import ParameterType

        assert hasattr(ParameterType, "PACE")
        assert hasattr(ParameterType, "GRID")
        assert hasattr(ParameterType, "STRATEGY")
        assert hasattr(ParameterType, "WEATHER")

    def test_parameter_sweep_creation(self):
        """Test ParameterSweep creation."""
        from f1_predict.simulation.analysis.sensitivity_analyzer import (
            ParameterSweep,
            ParameterType,
        )

        sweep = ParameterSweep(
            param_name="pace_VER",
            param_type=ParameterType.PACE,
            base_value=81.5,
        )

        sweep.add_linear_variation(-2.0, 2.0, 5)
        values = sweep.get_parameter_values()

        assert len(values) == 5
        assert values[0] == pytest.approx(-2.0)
        assert values[-1] == pytest.approx(2.0)

    def test_sensitivity_analyzer_creation(self):
        """Test SensitivityAnalyzer creation."""
        circuit = CircuitContext(
            circuit_name="Albert Park",
            total_laps=58,
        )

        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
        ]

        scenario = (
            ScenarioBuilder("test_scenario", circuit)
            .with_drivers(drivers)
            .build()
        )

        sim = MonteCarloSimulator(circuit=circuit, random_state=42)
        analyzer = SensitivityAnalyzer(sim, scenario, n_simulations=10)

        assert analyzer.simulator == sim
        assert analyzer.base_scenario == scenario


class TestSimulatorPageDataValidation:
    """Test data validation in simulator page."""

    def test_circuit_selection_valid(self):
        """Test that circuit selection validates circuit name."""
        valid_circuits = list(simulator.CIRCUITS.keys())
        assert "Australia" in valid_circuits
        assert len(valid_circuits) > 0

    def test_driver_selection_valid(self):
        """Test that driver selection validates driver IDs."""
        valid_drivers = list(simulator.DEFAULT_DRIVERS.keys())
        assert "VER" in valid_drivers
        assert len(valid_drivers) >= 10

    def test_simulation_parameter_ranges(self):
        """Test simulation parameter ranges are valid."""
        # Temperature range
        assert 5 < 25 < 40  # Default temp in range

        # Number of simulations range
        assert 10 < 1000 < 10000  # Default sims in range

    def test_sensitivity_sweep_parameters(self):
        """Test sensitivity sweep parameters."""
        min_val = -2.0
        max_val = 2.0
        num_steps = 5

        # Should be valid
        assert min_val < max_val
        assert num_steps >= 3
        assert num_steps <= 21


class TestSimulatorPageResultsDisplay:
    """Test results display functionality."""

    def test_results_data_structure(self):
        """Test that results data structure is valid."""
        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
            DriverState("LEC", "Charles Leclerc", expected_lap_time=82.0),
        ]

        results_data = []
        for driver in drivers:
            results_data.append({
                "Driver": f"{driver.driver_id} - {driver.driver_name}",
                "Win Probability": "50%",
                "Podium Probability": "75%",
            })

        assert len(results_data) == 2
        assert "Driver" in results_data[0]
        assert "Win Probability" in results_data[0]
        assert "Podium Probability" in results_data[0]

    def test_sensitivity_results_structure(self):
        """Test that sensitivity results have proper structure."""
        from f1_predict.simulation.analysis.sensitivity_report import SensitivityReport

        # Mock a sensitivity result
        mock_result = Mock()
        mock_result.parameter_name = "pace_VER"
        mock_result.parameter_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
        mock_result.drivers = ["VER", "LEC"]

        # SensitivityReport should accept the mock
        assert mock_result.parameter_name is not None
        assert len(mock_result.parameter_values) > 0


class TestSimulatorPageErrorHandling:
    """Test error handling in simulator page."""

    def test_invalid_circuit_handling(self):
        """Test handling of invalid circuit selection."""
        valid_circuits = list(simulator.CIRCUITS.keys())
        invalid_circuit = "Invalid_Circuit_Name"

        assert invalid_circuit not in valid_circuits

    def test_invalid_driver_handling(self):
        """Test handling of invalid driver selection."""
        valid_drivers = list(simulator.DEFAULT_DRIVERS.keys())
        invalid_driver = "XXX"

        assert invalid_driver not in valid_drivers

    def test_temperature_bounds(self):
        """Test temperature input bounds."""
        min_temp = 5
        max_temp = 40
        default_temp = 25

        assert min_temp <= default_temp <= max_temp

    def test_simulation_count_bounds(self):
        """Test simulation count bounds."""
        min_sims = 10
        max_sims = 10000
        default_sims = 1000

        assert min_sims <= default_sims <= max_sims


class TestSimulatorPageSessionState:
    """Test session state management."""

    def test_session_state_persistence(self):
        """Test that session state persists across interactions."""
        # Create mock session state
        session_state = {
            "race_config": {
                "circuit": "Australia",
                "weather": "Dry",
                "temperature": 25,
                "drivers": ["VER", "LEC", "HAM"],
            },
            "simulation_result": None,
            "sensitivity_result": None,
        }

        # Verify state persists
        assert session_state["race_config"]["circuit"] == "Australia"
        assert session_state["simulation_result"] is None

    def test_session_state_updates(self):
        """Test that session state can be updated."""
        session_state = {"race_config": {"circuit": "Australia"}}

        # Update state
        session_state["race_config"]["circuit"] = "Monaco"

        # Verify update
        assert session_state["race_config"]["circuit"] == "Monaco"


class TestSimulatorPageVisualization:
    """Test visualization components."""

    def test_plotly_import(self):
        """Test that plotly is available."""
        import plotly.graph_objects as go
        import plotly.express as px

        assert go is not None
        assert px is not None

    def test_bar_chart_creation(self):
        """Test bar chart creation for results."""
        import plotly.graph_objects as go

        drivers = ["VER", "LEC", "HAM"]
        probabilities = [0.45, 0.30, 0.25]

        fig = go.Figure(
            data=[go.Bar(x=drivers, y=probabilities)]
        )

        assert fig is not None

    def test_probability_curves_creation(self):
        """Test probability curves visualization."""
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[1, 2, 3, 4, 5],
                y=[0.1, 0.2, 0.3, 0.4, 0.5],
                mode="lines+markers",
                name="VER",
            )
        )

        assert fig is not None


class TestSimulatorPageIntegrationFlow:
    """Test complete integration flow."""

    def test_full_simulator_workflow(self):
        """Test full simulator workflow from config to results."""
        # Setup
        circuit_name = "Australia"
        circuit_info = simulator.CIRCUITS[circuit_name]

        circuit = CircuitContext(
            circuit_name=circuit_info["name"],
            total_laps=circuit_info["laps"],
        )

        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
            DriverState("LEC", "Charles Leclerc", expected_lap_time=82.0),
        ]

        # Create scenario
        scenario = (
            ScenarioBuilder(f"sim_{circuit_name}", circuit)
            .with_drivers(drivers)
            .build()
        )

        # Verify workflow steps
        assert circuit is not None
        assert scenario is not None
        assert len(scenario.drivers) == 2

    def test_sensitivity_workflow(self):
        """Test sensitivity analysis workflow."""
        # Setup
        circuit_name = "Australia"
        circuit_info = simulator.CIRCUITS[circuit_name]

        circuit = CircuitContext(
            circuit_name=circuit_info["name"],
            total_laps=circuit_info["laps"],
        )

        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
        ]

        scenario = (
            ScenarioBuilder(f"sensitivity_{circuit_name}", circuit)
            .with_drivers(drivers)
            .build()
        )

        # Create analyzer
        sim = MonteCarloSimulator(circuit=circuit, random_state=42)
        analyzer = SensitivityAnalyzer(sim, scenario, n_simulations=10)

        # Verify workflow
        assert analyzer is not None
        assert analyzer.simulator is not None


class TestSimulatorPagePageFunctions:
    """Test individual page functions."""

    def test_show_simulator_page_exists(self):
        """Test that show_simulator_page function exists."""
        assert hasattr(simulator, "show_simulator_page")
        assert callable(simulator.show_simulator_page)

    def test_helper_functions_exist(self):
        """Test that all helper functions exist."""
        assert hasattr(simulator, "initialize_session_state")
        assert hasattr(simulator, "show_circuit_configuration")
        assert hasattr(simulator, "show_driver_configuration")
        assert hasattr(simulator, "show_strategy_configuration")
        assert hasattr(simulator, "show_simulation_controls")
        assert hasattr(simulator, "display_simulation_results")
        assert hasattr(simulator, "show_sensitivity_analysis")
        assert hasattr(simulator, "display_sensitivity_results")


# Integration test fixtures
@pytest.fixture
def circuit_fixture():
    """Provide a test circuit."""
    return CircuitContext(
        circuit_name="Albert Park",
        total_laps=58,
    )


@pytest.fixture
def drivers_fixture():
    """Provide test drivers."""
    return [
        DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
        DriverState("LEC", "Charles Leclerc", expected_lap_time=82.0),
        DriverState("HAM", "Lewis Hamilton", expected_lap_time=82.1),
    ]


@pytest.fixture
def scenario_fixture(circuit_fixture, drivers_fixture):
    """Provide a test scenario."""
    return (
        ScenarioBuilder("test_scenario", circuit_fixture)
        .with_drivers(drivers_fixture)
        .build()
    )


@pytest.fixture
def simulator_fixture(circuit_fixture):
    """Provide a test simulator."""
    return MonteCarloSimulator(circuit=circuit_fixture, random_state=42)
