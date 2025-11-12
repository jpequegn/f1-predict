"""Unit tests for scenario builder."""

import pytest
from pathlib import Path
import tempfile

from f1_predict.simulation.analysis.scenario_builder import (
    RaceScenario,
    ScenarioBuilder,
    ScenarioRepository,
    ScenarioType,
    GridChange,
    DriverStrategy,
    DriverMechanicalIssue,
    WeatherCondition,
)
from f1_predict.simulation.core.driver_state import DriverState, TireCompound
from f1_predict.simulation.core.race_state import CircuitContext
from f1_predict.simulation.engine.pit_strategy import TireStrategy


class TestGridChange:
    """Test GridChange dataclass."""

    def test_grid_change_creation(self):
        """Test creating a grid change."""
        change = GridChange(driver_id="VER", new_position=5, reason="penalty")
        assert change.driver_id == "VER"
        assert change.new_position == 5
        assert change.reason == "penalty"

    def test_grid_change_invalid_position(self):
        """Test that invalid position raises ValueError."""
        with pytest.raises(ValueError):
            GridChange(driver_id="VER", new_position=0)

    def test_grid_change_negative_position(self):
        """Test that negative position raises ValueError."""
        with pytest.raises(ValueError):
            GridChange(driver_id="VER", new_position=-1)


class TestDriverStrategy:
    """Test DriverStrategy dataclass."""

    def test_driver_strategy_creation(self):
        """Test creating a driver strategy."""
        strategy = DriverStrategy(
            driver_id="VER",
            strategy=TireStrategy.TWO_STOP,
            pit_laps=[20, 35],
        )
        assert strategy.driver_id == "VER"
        assert strategy.strategy == TireStrategy.TWO_STOP
        assert strategy.pit_laps == [20, 35]

    def test_driver_strategy_invalid_pit_lap(self):
        """Test that invalid pit laps raise ValueError."""
        with pytest.raises(ValueError):
            DriverStrategy(
                driver_id="VER",
                strategy=TireStrategy.TWO_STOP,
                pit_laps=[0],
            )

    def test_driver_strategy_default_tire(self):
        """Test default tire compound."""
        strategy = DriverStrategy(driver_id="VER", strategy=TireStrategy.ONE_STOP)
        assert strategy.initial_tire == TireCompound.SOFT


class TestDriverMechanicalIssue:
    """Test DriverMechanicalIssue dataclass."""

    def test_mechanical_issue_creation(self):
        """Test creating mechanical issue."""
        issue = DriverMechanicalIssue(
            driver_id="VER",
            issue_type="engine_failure",
            lap=30,
            severity="dnf",
        )
        assert issue.driver_id == "VER"
        assert issue.issue_type == "engine_failure"
        assert issue.lap == 30
        assert issue.severity == "dnf"

    def test_mechanical_issue_performance_loss(self):
        """Test mechanical issue with performance loss."""
        issue = DriverMechanicalIssue(
            driver_id="VER",
            issue_type="hydraulic_leak",
            lap=15,
            severity="performance_loss",
            performance_penalty=2.5,
        )
        assert issue.severity == "performance_loss"
        assert issue.performance_penalty == 2.5

    def test_mechanical_issue_missing_penalty(self):
        """Test that performance_loss requires penalty."""
        with pytest.raises(ValueError):
            DriverMechanicalIssue(
                driver_id="VER",
                issue_type="hydraulic_leak",
                lap=15,
                severity="performance_loss",
            )

    def test_mechanical_issue_invalid_severity(self):
        """Test that invalid severity raises ValueError."""
        with pytest.raises(ValueError):
            DriverMechanicalIssue(
                driver_id="VER",
                issue_type="engine_failure",
                lap=30,
                severity="unknown",
            )

    def test_mechanical_issue_invalid_lap(self):
        """Test that invalid lap raises ValueError."""
        with pytest.raises(ValueError):
            DriverMechanicalIssue(
                driver_id="VER",
                issue_type="engine_failure",
                lap=0,
                severity="dnf",
            )


class TestWeatherCondition:
    """Test WeatherCondition dataclass."""

    def test_weather_condition_creation(self):
        """Test creating weather condition."""
        weather = WeatherCondition(condition_type="wet", start_lap=20, duration_laps=10)
        assert weather.condition_type == "wet"
        assert weather.start_lap == 20
        assert weather.duration_laps == 10

    def test_weather_condition_defaults(self):
        """Test weather condition defaults."""
        weather = WeatherCondition(condition_type="dry")
        assert weather.start_lap == 1
        assert weather.duration_laps is None

    def test_weather_condition_invalid_type(self):
        """Test that invalid weather type raises ValueError."""
        with pytest.raises(ValueError):
            WeatherCondition(condition_type="snow")

    def test_weather_condition_invalid_start_lap(self):
        """Test that invalid start lap raises ValueError."""
        with pytest.raises(ValueError):
            WeatherCondition(condition_type="wet", start_lap=0)


class TestRaceScenario:
    """Test RaceScenario dataclass."""

    def test_scenario_creation(self):
        """Test creating a race scenario."""
        circuit = CircuitContext(circuit_name="Albert Park")
        drivers = [
            DriverState("VER", "Max Verstappen"),
            DriverState("HAM", "Lewis Hamilton"),
        ]

        scenario = RaceScenario(
            scenario_id="test_scenario",
            scenario_type=ScenarioType.BASELINE,
            circuit=circuit,
            drivers=drivers,
        )

        assert scenario.scenario_id == "test_scenario"
        assert scenario.scenario_type == ScenarioType.BASELINE
        assert len(scenario.drivers) == 2

    def test_scenario_missing_id(self):
        """Test that missing scenario_id raises ValueError."""
        circuit = CircuitContext(circuit_name="Albert Park")
        drivers = [DriverState("VER", "Max Verstappen")]

        with pytest.raises(ValueError):
            RaceScenario(
                scenario_id="",
                scenario_type=ScenarioType.BASELINE,
                circuit=circuit,
                drivers=drivers,
            )

    def test_scenario_no_drivers(self):
        """Test that empty drivers list raises ValueError."""
        circuit = CircuitContext(circuit_name="Albert Park")

        with pytest.raises(ValueError):
            RaceScenario(
                scenario_id="test",
                scenario_type=ScenarioType.BASELINE,
                circuit=circuit,
                drivers=[],
            )

    def test_scenario_duplicate_grid_positions(self):
        """Test that duplicate grid positions raise ValueError."""
        circuit = CircuitContext(circuit_name="Albert Park")
        drivers = [
            DriverState("VER", "Max Verstappen"),
            DriverState("HAM", "Lewis Hamilton"),
        ]

        with pytest.raises(ValueError):
            RaceScenario(
                scenario_id="test",
                scenario_type=ScenarioType.GRID_CHANGE,
                circuit=circuit,
                drivers=drivers,
                grid_changes=[
                    GridChange("VER", 5),
                    GridChange("HAM", 5),
                ],
            )

    def test_get_modified_drivers(self):
        """Test getting drivers with grid changes applied."""
        circuit = CircuitContext(circuit_name="Albert Park")
        drivers = [
            DriverState("VER", "Max Verstappen", position=1),
            DriverState("HAM", "Lewis Hamilton", position=2),
        ]

        scenario = RaceScenario(
            scenario_id="test",
            scenario_type=ScenarioType.GRID_CHANGE,
            circuit=circuit,
            drivers=drivers,
            grid_changes=[GridChange("VER", 5)],
        )

        modified = scenario.get_modified_drivers()
        assert len(modified) == 2

        # Find VER in modified list
        ver = next((d for d in modified if d.driver_id == "VER"), None)
        assert ver is not None
        assert ver.position == 5

    def test_get_driver_strategy(self):
        """Test getting driver strategy."""
        circuit = CircuitContext(circuit_name="Albert Park")
        drivers = [DriverState("VER", "Max Verstappen")]

        strategy = DriverStrategy("VER", TireStrategy.TWO_STOP)

        scenario = RaceScenario(
            scenario_id="test",
            scenario_type=ScenarioType.STRATEGY,
            circuit=circuit,
            drivers=drivers,
            driver_strategies=[strategy],
        )

        retrieved = scenario.get_driver_strategy("VER")
        assert retrieved is not None
        assert retrieved.strategy == TireStrategy.TWO_STOP

    def test_get_mechanical_issues(self):
        """Test getting mechanical issues for driver."""
        circuit = CircuitContext(circuit_name="Albert Park")
        drivers = [DriverState("VER", "Max Verstappen")]

        issue1 = DriverMechanicalIssue("VER", "engine_failure", 30)
        issue2 = DriverMechanicalIssue("VER", "hydraulic_leak", 40)

        scenario = RaceScenario(
            scenario_id="test",
            scenario_type=ScenarioType.MECHANICAL,
            circuit=circuit,
            drivers=drivers,
            mechanical_issues=[issue1, issue2],
        )

        issues = scenario.get_mechanical_issues_for_driver("VER")
        assert len(issues) == 2

    def test_scenario_to_dict(self):
        """Test serializing scenario to dictionary."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=58)
        drivers = [DriverState("VER", "Max Verstappen")]

        scenario = RaceScenario(
            scenario_id="test",
            scenario_type=ScenarioType.BASELINE,
            circuit=circuit,
            drivers=drivers,
            description="Test scenario",
            n_simulations=1000,
        )

        scenario_dict = scenario.to_dict()

        assert scenario_dict["scenario_id"] == "test"
        assert scenario_dict["scenario_type"] == "baseline"
        assert scenario_dict["circuit"]["circuit_name"] == "Albert Park"
        assert len(scenario_dict["drivers"]) == 1
        assert scenario_dict["description"] == "Test scenario"
        assert scenario_dict["n_simulations"] == 1000


class TestScenarioBuilder:
    """Test ScenarioBuilder pattern."""

    def test_builder_basic(self):
        """Test basic builder usage."""
        circuit = CircuitContext(circuit_name="Albert Park")
        drivers = [DriverState("VER", "Max Verstappen")]

        scenario = (
            ScenarioBuilder("scenario1", circuit)
            .with_drivers(drivers)
            .with_description("Test scenario")
            .build()
        )

        assert scenario.scenario_id == "scenario1"
        assert scenario.description == "Test scenario"
        assert len(scenario.drivers) == 1

    def test_builder_with_grid_change(self):
        """Test builder with grid changes."""
        circuit = CircuitContext(circuit_name="Albert Park")
        drivers = [
            DriverState("VER", "Max Verstappen"),
            DriverState("HAM", "Lewis Hamilton"),
        ]

        scenario = (
            ScenarioBuilder("scenario1", circuit)
            .with_drivers(drivers)
            .add_grid_change("VER", 5, "10-place penalty")
            .build()
        )

        assert len(scenario.grid_changes) == 1
        assert scenario.grid_changes[0].driver_id == "VER"
        assert scenario.scenario_type == ScenarioType.CUSTOM

    def test_builder_with_strategy(self):
        """Test builder with pit strategies."""
        circuit = CircuitContext(circuit_name="Albert Park")
        drivers = [DriverState("VER", "Max Verstappen")]

        scenario = (
            ScenarioBuilder("scenario1", circuit)
            .with_drivers(drivers)
            .add_driver_strategy("VER", TireStrategy.TWO_STOP)
            .build()
        )

        assert len(scenario.driver_strategies) == 1
        assert scenario.driver_strategies[0].strategy == TireStrategy.TWO_STOP

    def test_builder_with_mechanical_issue(self):
        """Test builder with mechanical issues."""
        circuit = CircuitContext(circuit_name="Albert Park")
        drivers = [DriverState("VER", "Max Verstappen")]

        scenario = (
            ScenarioBuilder("scenario1", circuit)
            .with_drivers(drivers)
            .add_mechanical_issue("VER", "engine_failure", 30)
            .build()
        )

        assert len(scenario.mechanical_issues) == 1
        assert scenario.mechanical_issues[0].issue_type == "engine_failure"

    def test_builder_with_weather(self):
        """Test builder with weather changes."""
        circuit = CircuitContext(circuit_name="Albert Park")
        drivers = [DriverState("VER", "Max Verstappen")]

        scenario = (
            ScenarioBuilder("scenario1", circuit)
            .with_drivers(drivers)
            .add_weather("wet", start_lap=20, duration_laps=10)
            .build()
        )

        assert len(scenario.weather_conditions) == 1
        assert scenario.weather_conditions[0].condition_type == "wet"

    def test_builder_with_simulations(self):
        """Test builder with custom simulation count."""
        circuit = CircuitContext(circuit_name="Albert Park")
        drivers = [DriverState("VER", "Max Verstappen")]

        scenario = (
            ScenarioBuilder("scenario1", circuit)
            .with_drivers(drivers)
            .with_simulations(500)
            .build()
        )

        assert scenario.n_simulations == 500

    def test_builder_invalid_simulations(self):
        """Test that invalid simulation count raises ValueError."""
        circuit = CircuitContext(circuit_name="Albert Park")

        with pytest.raises(ValueError):
            ScenarioBuilder("scenario1", circuit).with_simulations(0)

    def test_builder_with_seed(self):
        """Test builder with random seed."""
        circuit = CircuitContext(circuit_name="Albert Park")
        drivers = [DriverState("VER", "Max Verstappen")]

        scenario = (
            ScenarioBuilder("scenario1", circuit)
            .with_drivers(drivers)
            .with_seed(42)
            .build()
        )

        assert scenario.random_seed == 42

    def test_builder_with_metadata(self):
        """Test builder with metadata."""
        circuit = CircuitContext(circuit_name="Albert Park")
        drivers = [DriverState("VER", "Max Verstappen")]

        scenario = (
            ScenarioBuilder("scenario1", circuit)
            .with_drivers(drivers)
            .with_metadata("created_by", "test_user")
            .with_metadata("version", "1.0")
            .build()
        )

        assert scenario.metadata["created_by"] == "test_user"
        assert scenario.metadata["version"] == "1.0"

    def test_builder_complex_scenario(self):
        """Test builder with multiple parameter changes."""
        circuit = CircuitContext(circuit_name="Albert Park")
        drivers = [
            DriverState("VER", "Max Verstappen"),
            DriverState("HAM", "Lewis Hamilton"),
        ]

        scenario = (
            ScenarioBuilder("complex_scenario", circuit)
            .with_drivers(drivers)
            .with_type(ScenarioType.CUSTOM)
            .with_description("What-if: VER grid penalty + weather change")
            .add_grid_change("VER", 5, "10-place penalty")
            .add_driver_strategy("HAM", TireStrategy.ONE_STOP)
            .add_mechanical_issue("VER", "hydraulic_leak", 40, "performance_loss", 1.5)
            .add_weather("wet", 20, 15)
            .with_simulations(200)
            .with_seed(123)
            .with_metadata("analyst", "test_user")
            .build()
        )

        assert scenario.scenario_id == "complex_scenario"
        assert len(scenario.grid_changes) == 1
        assert len(scenario.driver_strategies) == 1
        assert len(scenario.mechanical_issues) == 1
        assert len(scenario.weather_conditions) == 1
        assert scenario.n_simulations == 200
        assert scenario.random_seed == 123


class TestScenarioRepository:
    """Test ScenarioRepository for storage and retrieval."""

    def test_repository_save_scenario(self):
        """Test saving scenario to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ScenarioRepository(Path(tmpdir))

            circuit = CircuitContext(circuit_name="Albert Park")
            drivers = [DriverState("VER", "Max Verstappen")]

            scenario = RaceScenario(
                scenario_id="test_scenario",
                scenario_type=ScenarioType.BASELINE,
                circuit=circuit,
                drivers=drivers,
                description="Test scenario",
            )

            file_path = repo.save_scenario(scenario)

            assert file_path.exists()
            assert file_path.name == "test_scenario.json"

    def test_repository_load_scenario(self):
        """Test loading scenario from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ScenarioRepository(Path(tmpdir))

            circuit = CircuitContext(circuit_name="Albert Park")
            drivers = [DriverState("VER", "Max Verstappen")]

            scenario = RaceScenario(
                scenario_id="test_scenario",
                scenario_type=ScenarioType.BASELINE,
                circuit=circuit,
                drivers=drivers,
                description="Test scenario",
            )

            repo.save_scenario(scenario)
            loaded_dict = repo.load_scenario_dict("test_scenario")

            assert loaded_dict["scenario_id"] == "test_scenario"
            assert loaded_dict["description"] == "Test scenario"
            assert len(loaded_dict["drivers"]) == 1

    def test_repository_load_nonexistent(self):
        """Test that loading nonexistent scenario raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ScenarioRepository(Path(tmpdir))

            with pytest.raises(FileNotFoundError):
                repo.load_scenario_dict("nonexistent")

    def test_repository_list_scenarios(self):
        """Test listing saved scenarios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ScenarioRepository(Path(tmpdir))

            circuit = CircuitContext(circuit_name="Albert Park")
            drivers = [DriverState("VER", "Max Verstappen")]

            for i in range(3):
                scenario = RaceScenario(
                    scenario_id=f"scenario_{i}",
                    scenario_type=ScenarioType.BASELINE,
                    circuit=circuit,
                    drivers=drivers,
                )
                repo.save_scenario(scenario)

            scenarios = repo.list_scenarios()

            assert len(scenarios) == 3
            assert "scenario_0" in scenarios
            assert "scenario_1" in scenarios
            assert "scenario_2" in scenarios

    def test_repository_delete_scenario(self):
        """Test deleting a saved scenario."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ScenarioRepository(Path(tmpdir))

            circuit = CircuitContext(circuit_name="Albert Park")
            drivers = [DriverState("VER", "Max Verstappen")]

            scenario = RaceScenario(
                scenario_id="test_scenario",
                scenario_type=ScenarioType.BASELINE,
                circuit=circuit,
                drivers=drivers,
            )

            repo.save_scenario(scenario)
            repo.delete_scenario("test_scenario")

            assert "test_scenario" not in repo.list_scenarios()
