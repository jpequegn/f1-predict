"""Unit tests for RaceState and CircuitContext classes."""

import pytest

from f1_predict.simulation.core.driver_state import DriverState, DriverStatus
from f1_predict.simulation.core.race_state import CircuitContext, RaceState


class TestCircuitContext:
    """Test CircuitContext class."""

    def test_circuit_context_creation(self):
        """Test circuit context creation."""
        circuit = CircuitContext(
            circuit_name="Albert Park",
            circuit_type="intermediate",
            total_laps=58,
            lap_distance=5.303,
        )
        assert circuit.circuit_name == "Albert Park"
        assert circuit.circuit_type == "intermediate"
        assert circuit.total_laps == 58
        assert circuit.lap_distance == 5.303

    def test_circuit_context_defaults(self):
        """Test circuit context default values."""
        circuit = CircuitContext(circuit_name="Test Circuit")
        assert circuit.circuit_type == "intermediate"
        assert circuit.total_laps == 58
        assert circuit.lap_distance == 5.3
        assert circuit.safety_car_prob == 0.08
        assert circuit.dnf_rate == 0.08


class TestRaceStateCreation:
    """Test RaceState initialization."""

    def test_race_state_creation(self):
        """Test basic race state creation."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)

        assert state.circuit == circuit
        assert len(state.drivers) == 0
        assert state.current_lap == 1
        assert state.safety_car_active is False
        assert state.race_finished is False
        assert state.weather_condition == "dry"

    def test_race_state_with_drivers(self):
        """Test race state with drivers."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)

        driver1 = DriverState("VER", "Max Verstappen")
        driver2 = DriverState("HAM", "Lewis Hamilton")

        state.add_driver(driver1)
        state.add_driver(driver2)

        assert len(state.drivers) == 2
        assert "VER" in state.drivers
        assert "HAM" in state.drivers


class TestRaceStateDriverManagement:
    """Test driver management methods."""

    def test_add_driver(self):
        """Test adding a driver to race."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)
        driver = DriverState("VER", "Max Verstappen")

        state.add_driver(driver)
        assert state.get_driver("VER") == driver

    def test_add_duplicate_driver_raises_error(self):
        """Test that adding duplicate driver raises ValueError."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)
        driver = DriverState("VER", "Max Verstappen")

        state.add_driver(driver)
        with pytest.raises(ValueError):
            state.add_driver(driver)

    def test_get_driver_not_found(self):
        """Test getting non-existent driver raises ValueError."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)

        with pytest.raises(ValueError):
            state.get_driver("VER")

    def test_remove_driver(self):
        """Test removing driver (DNF)."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)
        driver = DriverState("VER", "Max Verstappen")
        state.add_driver(driver)

        state.remove_driver("VER", reason="engine failure")

        driver = state.get_driver("VER")
        assert driver.is_dnf
        assert driver.dnf_reason == "engine failure"

    def test_remove_non_existent_driver(self):
        """Test removing non-existent driver raises ValueError."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)

        with pytest.raises(ValueError):
            state.remove_driver("VER")


class TestRaceStateQueries:
    """Test query methods."""

    def test_get_active_drivers(self):
        """Test getting active drivers."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)

        driver1 = DriverState("VER", "Max Verstappen", position=1)
        driver2 = DriverState("HAM", "Lewis Hamilton", position=2)
        driver3 = DriverState("LEC", "Charles Leclerc", position=3)

        state.add_driver(driver1)
        state.add_driver(driver2)
        state.add_driver(driver3)

        # Mark one as DNF
        driver2.dnf("crash")

        active = state.get_active_drivers()
        assert len(active) == 2
        assert driver1 in active
        assert driver3 in active
        assert driver2 not in active

    def test_get_finished_drivers(self):
        """Test getting finished drivers."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)

        driver1 = DriverState("VER", "Max Verstappen")
        driver2 = DriverState("HAM", "Lewis Hamilton")

        state.add_driver(driver1)
        state.add_driver(driver2)

        driver1.finish_race()

        finished = state.get_finished_drivers()
        assert len(finished) == 1
        assert driver1 in finished

    def test_get_dnf_drivers(self):
        """Test getting DNF drivers."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)

        driver1 = DriverState("VER", "Max Verstappen")
        driver2 = DriverState("HAM", "Lewis Hamilton")

        state.add_driver(driver1)
        state.add_driver(driver2)

        driver1.dnf("engine failure")

        dnf = state.get_dnf_drivers()
        assert len(dnf) == 1
        assert driver1 in dnf

    def test_get_leader(self):
        """Test getting race leader."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)

        driver1 = DriverState("VER", "Max Verstappen", position=1)
        driver2 = DriverState("HAM", "Lewis Hamilton", position=2)

        state.add_driver(driver1)
        state.add_driver(driver2)

        leader = state.get_leader()
        assert leader == driver1

    def test_get_leader_no_active_drivers(self):
        """Test getting leader when no active drivers."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)

        leader = state.get_leader()
        assert leader is None


class TestRaceStatePositionUpdates:
    """Test position update methods."""

    def test_update_positions(self):
        """Test updating driver positions."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)

        driver1 = DriverState("VER", "Max Verstappen", position=1, lap=20)
        driver2 = DriverState("HAM", "Lewis Hamilton", position=2, lap=20)
        driver3 = DriverState("LEC", "Charles Leclerc", position=3, lap=19)

        state.add_driver(driver1)
        state.add_driver(driver2)
        state.add_driver(driver3)

        state.update_positions()

        # Verify positions updated based on lap count
        active = state.get_active_drivers()
        assert active[0].driver_id == "VER"
        assert active[1].driver_id == "HAM"
        assert active[2].driver_id == "LEC"

    def test_record_lap_snapshot(self):
        """Test recording lap snapshots."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)

        driver1 = DriverState("VER", "Max Verstappen", position=1)
        driver2 = DriverState("HAM", "Lewis Hamilton", position=2)

        state.add_driver(driver1)
        state.add_driver(driver2)

        state.record_lap_snapshot()
        assert len(state.lap_history) == 1

        state.advance_lap()
        state.record_lap_snapshot()
        assert len(state.lap_history) == 2


class TestRaceStateProgression:
    """Test race progression methods."""

    def test_advance_lap(self):
        """Test advancing race lap."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=58)
        state = RaceState(circuit=circuit)

        assert state.current_lap == 1
        state.advance_lap()
        assert state.current_lap == 2

    def test_finish_race(self):
        """Test finishing the race."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)

        driver1 = DriverState("VER", "Max Verstappen", position=1)
        driver2 = DriverState("HAM", "Lewis Hamilton", position=2)

        state.add_driver(driver1)
        state.add_driver(driver2)

        state.finish_race()

        assert state.race_finished is True
        assert driver1.is_finished
        assert driver2.is_finished

    def test_is_race_complete(self):
        """Test race completion check."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)

        driver1 = DriverState("VER", "Max Verstappen")
        driver2 = DriverState("HAM", "Lewis Hamilton")

        state.add_driver(driver1)
        state.add_driver(driver2)

        assert state.is_race_complete() is False

        driver1.finish_race()
        driver2.finish_race()

        assert state.is_race_complete() is True


class TestRaceStateResults:
    """Test race results generation."""

    def test_get_race_results(self):
        """Test getting race results."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)

        driver1 = DriverState("VER", "Max Verstappen", position=1)
        driver2 = DriverState("HAM", "Lewis Hamilton", position=2)
        driver3 = DriverState("LEC", "Charles Leclerc", position=3)

        state.add_driver(driver1)
        state.add_driver(driver2)
        state.add_driver(driver3)

        # Simulate race progress
        driver1.lap = 59
        driver1.finish_race()
        driver2.lap = 59
        driver2.finish_race()
        driver3.lap = 57
        driver3.dnf("crash")

        state.update_positions()
        results = state.get_race_results()

        assert len(results) == 3
        assert results[0]["status"] == DriverStatus.FINISHED.value
        assert "DNF" in results[2]["status"]

    def test_get_race_results_empty(self):
        """Test getting results with no drivers."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state = RaceState(circuit=circuit)

        results = state.get_race_results()
        assert results == []


class TestRaceStateCopy:
    """Test race state copying."""

    def test_copy_creates_independent_copy(self):
        """Test that copy creates an independent race state."""
        circuit = CircuitContext(circuit_name="Albert Park")
        state1 = RaceState(circuit=circuit)

        driver1 = DriverState("VER", "Max Verstappen")
        state1.add_driver(driver1)

        state2 = state1.copy()

        # Verify copy has same structure
        assert state2.circuit == state1.circuit
        assert len(state2.drivers) == len(state1.drivers)

        # Modify copy and verify original unchanged
        state2.drivers["VER"].update_position(5)
        state2.advance_lap()

        assert state1.drivers["VER"].position == 1
        assert state1.current_lap == 1
        assert state2.drivers["VER"].position == 5
        assert state2.current_lap == 2
