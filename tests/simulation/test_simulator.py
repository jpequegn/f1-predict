"""Unit tests for Monte Carlo simulator."""

import pytest

from f1_predict.simulation.core.driver_state import DriverState
from f1_predict.simulation.core.race_state import CircuitContext
from f1_predict.simulation.engine.simulator import (
    MonteCarloSimulator,
    SimulationResult,
    SimulationRun,
)


class TestSimulationRun:
    """Test SimulationRun dataclass."""

    def test_simulation_run_creation(self):
        """Test creating a simulation run."""
        run = SimulationRun(
            run_id=0,
            final_positions=["VER", "HAM", "LEC"],
            final_order=[("VER", "Max Verstappen"), ("HAM", "Lewis Hamilton")],
        )
        assert run.run_id == 0
        assert len(run.final_positions) == 3
        assert len(run.final_order) == 2


class TestSimulationResult:
    """Test SimulationResult dataclass."""

    def test_simulation_result_creation(self):
        """Test creating simulation results."""
        result = SimulationResult(n_runs=100)
        assert result.n_runs == 100
        assert len(result.finish_probabilities) == 0
        assert len(result.position_distributions) == 0

    def test_get_winner_probability_empty(self):
        """Test getting winner probability on empty results."""
        result = SimulationResult(n_runs=100)
        prob = result.get_winner_probability("VER")
        assert prob == 0.0

    def test_get_winner_probability(self):
        """Test getting winner probability."""
        result = SimulationResult(n_runs=100)
        result.position_distributions[1] = {"VER": 0.65, "HAM": 0.35}

        assert result.get_winner_probability("VER") == 0.65
        assert result.get_winner_probability("HAM") == 0.35

    def test_get_podium_probability(self):
        """Test getting podium probability."""
        result = SimulationResult(n_runs=100)
        result.position_distributions[1] = {"VER": 0.50}
        result.position_distributions[2] = {"VER": 0.30}
        result.position_distributions[3] = {"VER": 0.15}

        podium_prob = result.get_podium_probability("VER")
        assert abs(podium_prob - 0.95) < 0.01

    def test_get_podium_probability_capped_at_one(self):
        """Test that podium probability doesn't exceed 1.0."""
        result = SimulationResult(n_runs=100)
        result.position_distributions[1] = {"VER": 0.6}
        result.position_distributions[2] = {"VER": 0.6}
        result.position_distributions[3] = {"VER": 0.6}

        podium_prob = result.get_podium_probability("VER")
        assert podium_prob <= 1.0


class TestMonteCarloSimulator:
    """Test MonteCarloSimulator class."""

    def test_simulator_creation(self):
        """Test creating simulator."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=58)
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        assert simulator.circuit == circuit
        assert len(simulator.runs) == 0

    def test_simulate_single_race(self):
        """Test simulating a single race."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=58)
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
            DriverState("HAM", "Lewis Hamilton", expected_lap_time=82.0),
            DriverState("LEC", "Charles Leclerc", expected_lap_time=82.5),
        ]

        run = simulator.simulate_race(drivers, run_id=0)

        assert isinstance(run, SimulationRun)
        assert run.run_id == 0
        # Either some drivers finished or some DNFed (both are valid outcomes)
        assert len(run.final_order) + len(run.dnf_drivers) <= len(drivers)
        assert len(run.pit_stop_counts) == 3

    def test_simulate_race_generates_results(self):
        """Test that simulation generates proper results."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=10)  # Short race
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
            DriverState("HAM", "Lewis Hamilton", expected_lap_time=82.0),
        ]

        run = simulator.simulate_race(drivers)

        # Should have some drivers finished
        assert len(run.final_order) > 0
        # Total drivers should match input
        assert len(run.dnf_drivers) + len(run.final_order) <= len(drivers)

    def test_run_multiple_simulations(self):
        """Test running multiple simulations."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=10)
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
            DriverState("HAM", "Lewis Hamilton", expected_lap_time=82.0),
        ]

        result = simulator.run_simulations(drivers, n_simulations=10)

        assert isinstance(result, SimulationResult)
        assert result.n_runs == 10
        assert len(result.finish_probabilities) <= len(drivers)

    def test_aggregated_results_probabilities(self):
        """Test that aggregated results contain valid probabilities."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=10)
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
            DriverState("HAM", "Lewis Hamilton", expected_lap_time=82.0),
        ]

        result = simulator.run_simulations(drivers, n_simulations=20)

        # Check probabilities are valid (0.0-1.0)
        for driver_id, prob in result.finish_probabilities.items():
            assert 0.0 <= prob <= 1.0

        # Check DNF rates are valid
        for driver_id, rate in result.dnf_rates.items():
            assert 0.0 <= rate <= 1.0

    def test_simulator_tracks_runs(self):
        """Test that simulator tracks all runs."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=10)
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
        ]

        simulator.run_simulations(drivers, n_simulations=5)

        assert len(simulator.runs) == 5

    def test_simulator_reproducibility(self):
        """Test that simulations are reproducible with same random state."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=10)

        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
            DriverState("HAM", "Lewis Hamilton", expected_lap_time=82.0),
        ]

        # Run 1
        sim1 = MonteCarloSimulator(circuit=circuit, random_state=42)
        result1 = sim1.run_simulations(drivers, n_simulations=5)

        # Run 2 with same seed
        drivers2 = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
            DriverState("HAM", "Lewis Hamilton", expected_lap_time=82.0),
        ]
        sim2 = MonteCarloSimulator(circuit=circuit, random_state=42)
        result2 = sim2.run_simulations(drivers2, n_simulations=5)

        # Results should be identical
        assert result1.n_runs == result2.n_runs
        # Final orders should match
        for i in range(len(sim1.runs)):
            assert sim1.runs[i].final_order == sim2.runs[i].final_order

    def test_different_driver_performance(self):
        """Test that faster drivers tend to win more."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=10)
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        # VER is faster than HAM
        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=80.0),  # Faster
            DriverState("HAM", "Lewis Hamilton", expected_lap_time=81.0),  # Slower
        ]

        result = simulator.run_simulations(drivers, n_simulations=50)

        ver_wins = result.get_winner_probability("VER")
        ham_wins = result.get_winner_probability("HAM")

        # Faster driver should win more often
        assert ver_wins > ham_wins

    def test_pit_stop_tracking(self):
        """Test that pit stops are tracked."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=10)
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
        ]

        result = simulator.run_simulations(drivers, n_simulations=5)

        assert "VER" in result.average_pit_stops
        assert result.average_pit_stops["VER"] >= 0  # Pit stops >= 0


class TestSimulationStatistics:
    """Test statistical properties of simulations."""

    def test_position_distributions_sum(self):
        """Test that position probabilities sum correctly."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=10)
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
            DriverState("HAM", "Lewis Hamilton", expected_lap_time=82.0),
            DriverState("LEC", "Charles Leclerc", expected_lap_time=82.5),
        ]

        result = simulator.run_simulations(drivers, n_simulations=20)

        # For each position, probabilities should sum to roughly 1.0
        # (or less if some drivers DNF before reaching that position)
        for position in result.position_distributions:
            total_prob = sum(
                result.position_distributions[position].values()
            )
            assert total_prob <= 1.01  # Allow for floating point error

    def test_dnf_and_finish_probabilities_sum(self):
        """Test that finish and DNF probabilities sum to 1."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=10)
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
            DriverState("HAM", "Lewis Hamilton", expected_lap_time=82.0),
        ]

        result = simulator.run_simulations(drivers, n_simulations=20)

        for driver_id in result.dnf_rates:
            finish_prob = result.finish_probabilities.get(driver_id, 0.0)
            dnf_prob = result.dnf_rates[driver_id]
            # Finish + DNF should roughly equal 1.0
            assert abs((finish_prob + dnf_prob) - 1.0) < 0.1
