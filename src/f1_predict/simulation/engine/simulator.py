"""Main Monte Carlo race simulator."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from f1_predict.simulation.core.driver_state import DriverState, DriverStatus
from f1_predict.simulation.core.incidents import IncidentEvent, IncidentGenerator
from f1_predict.simulation.core.race_state import CircuitContext, RaceState

logger = logging.getLogger(__name__)


@dataclass
class SimulationRun:
    """Result of a single race simulation.

    Attributes:
        run_id: Unique identifier for this simulation run
        final_positions: Finishing positions [driver_id, ...]
        final_order: List of (driver_id, driver_name) tuples in finish order
        dnf_drivers: Set of driver IDs that did not finish
        race_duration_laps: Total laps completed
        incidents: List of IncidentEvent objects
        pit_stop_counts: Dictionary mapping driver_id to pit stop count
    """

    run_id: int
    final_positions: list[str] = field(default_factory=list)
    final_order: list[tuple[str, str]] = field(default_factory=list)
    dnf_drivers: set[str] = field(default_factory=set)
    race_duration_laps: int = 0
    incidents: list[IncidentEvent] = field(default_factory=list)
    pit_stop_counts: dict[str, int] = field(default_factory=dict)
    best_laps: dict[str, Optional[float]] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Aggregated results from multiple simulation runs.

    Attributes:
        n_runs: Number of simulation runs
        finish_probabilities: Dict mapping driver_id to probability of finish
        position_distributions: Dict mapping position (1-N) to driver probabilities
        dnf_rates: Dict mapping driver_id to DNF rate
        average_pit_stops: Dict mapping driver_id to avg pit stops
        incidents_total: List of all incidents across runs
    """

    n_runs: int
    finish_probabilities: dict[str, float] = field(default_factory=dict)
    position_distributions: dict[int, dict[str, float]] = field(
        default_factory=dict
    )
    dnf_rates: dict[str, float] = field(default_factory=dict)
    average_pit_stops: dict[str, float] = field(default_factory=dict)
    incidents_total: list[IncidentEvent] = field(default_factory=list)

    def get_winner_probability(self, driver_id: str) -> float:
        """Get probability of driver finishing in position 1.

        Args:
            driver_id: ID of driver

        Returns:
            Probability (0.0-1.0)
        """
        if 1 in self.position_distributions:
            return self.position_distributions[1].get(driver_id, 0.0)
        return 0.0

    def get_podium_probability(self, driver_id: str) -> float:
        """Get probability of driver finishing in top 3.

        Args:
            driver_id: ID of driver

        Returns:
            Probability (0.0-1.0)
        """
        podium_prob = 0.0
        for position in [1, 2, 3]:
            if position in self.position_distributions:
                podium_prob += self.position_distributions[position].get(driver_id, 0.0)
        return min(1.0, podium_prob)


class MonteCarloSimulator:
    """Execute Monte Carlo race simulations.

    Runs multiple independent race simulations with different random seeds
    to generate probability distributions over outcomes.
    """

    def __init__(
        self,
        circuit: CircuitContext,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize simulator.

        Args:
            circuit: Circuit information
            random_state: Random seed for reproducibility
        """
        self.circuit = circuit
        self.rng = np.random.RandomState(random_state)
        self.runs: list[SimulationRun] = []
        self.incident_gen = IncidentGenerator(circuit.circuit_type, random_state)

    def add_driver(self, driver: DriverState) -> None:
        """Add driver to race template."""
        # This would be used to configure drivers before running sims
        pass

    def simulate_race(self, drivers: list[DriverState], run_id: int = 0) -> SimulationRun:
        """Simulate a single race.

        Args:
            drivers: List of DriverState objects for the race
            run_id: ID for this simulation run

        Returns:
            SimulationRun with results
        """
        # Create race state
        state = RaceState(circuit=self.circuit)

        # Add drivers
        for driver in drivers:
            state.add_driver(driver.copy())

        # Reset incident generator
        self.incident_gen.clear_incidents()

        # Simulate race lap by lap
        while not state.is_race_complete() and state.current_lap <= self.circuit.total_laps + 10:
            # Check for incidents
            if self.rng.random() < 0.02:  # 2% chance of incident per lap
                incident = self.incident_gen.generate_safety_car(
                    state.current_lap, self.circuit.total_laps
                )
                if incident:
                    state.safety_car_active = True

            # Simulate each driver's lap
            for driver_id, driver in state.drivers.items():
                if not driver.is_active:
                    continue

                # Apply performance variance
                driver.pace_variance = self.rng.normal(1.0, 0.02)  # 2% std dev

                # Calculate lap time with variance
                tire_deg_factor = 1.0 + (driver.laps_on_tire * 0.001)
                fuel_factor = 1.0 + ((100 - driver.fuel_level) / 1000)
                lap_time = (
                    driver.expected_lap_time
                    * driver.pace_variance
                    * tire_deg_factor
                    * fuel_factor
                )

                # Consume fuel
                fuel_consumption = 1.0 + self.rng.normal(0, 0.1)
                driver.consume_fuel(fuel_consumption)

                # Check for DNF
                dnf_event = self.incident_gen.generate_dnf(
                    driver.driver_id,
                    driver.driver_name,
                    driver.lap,
                    self.circuit.total_laps,
                )
                if dnf_event:
                    driver.dnf(dnf_event.description)
                    continue

                # Complete lap
                driver.complete_lap(lap_time)

            # Update race state
            state.update_positions()
            state.record_lap_snapshot()

            # Check if race is done
            leader = state.get_leader()
            if leader and leader.lap > self.circuit.total_laps:
                state.finish_race()

            state.advance_lap()

        # Finalize results
        state.finish_race()

        # Build simulation run result
        finished = state.get_finished_drivers()
        final_positions = [d.driver_id for d in finished]
        final_order = [(d.driver_id, d.driver_name) for d in finished]
        dnf_set = {d.driver_id for d in state.get_dnf_drivers()}

        run = SimulationRun(
            run_id=run_id,
            final_positions=final_positions,
            final_order=final_order,
            dnf_drivers=dnf_set,
            race_duration_laps=state.current_lap - 1,
            incidents=self.incident_gen.get_incidents(),
            pit_stop_counts={d.driver_id: d.pit_stop_count for d in state.drivers.values()},
            best_laps={d.driver_id: d.best_lap_time for d in state.drivers.values()},
        )

        self.runs.append(run)
        return run

    def run_simulations(
        self,
        drivers: list[DriverState],
        n_simulations: int = 1000,
    ) -> SimulationResult:
        """Run multiple race simulations.

        Args:
            drivers: List of DriverState objects
            n_simulations: Number of simulations to run

        Returns:
            SimulationResult with aggregated results
        """
        logger.info(f"Starting {n_simulations} race simulations")

        # Reset runs
        self.runs = []

        # Run simulations
        for i in range(n_simulations):
            if (i + 1) % max(1, n_simulations // 10) == 0:
                logger.info(f"Completed {i + 1}/{n_simulations} simulations")

            self.simulate_race(drivers, run_id=i)

        # Aggregate results
        result = self._aggregate_results()
        logger.info("Simulation complete")

        return result

    def _aggregate_results(self) -> SimulationResult:
        """Aggregate results from all runs.

        Returns:
            SimulationResult with probability distributions
        """
        result = SimulationResult(n_runs=len(self.runs))

        # Get unique drivers
        all_driver_ids = set()
        for run in self.runs:
            for driver_id, _ in run.final_order:
                all_driver_ids.add(driver_id)
            all_driver_ids.update(run.dnf_drivers)

        # Calculate finish probabilities
        for driver_id in all_driver_ids:
            finish_count = sum(
                1 for run in self.runs if driver_id not in run.dnf_drivers
            )
            result.finish_probabilities[driver_id] = finish_count / len(self.runs)

        # Calculate position distributions
        max_position = max(len(run.final_order) for run in self.runs)

        for position in range(1, max_position + 1):
            result.position_distributions[position] = {}

            for driver_id in all_driver_ids:
                count = sum(
                    1
                    for run in self.runs
                    if len(run.final_order) >= position
                    and run.final_order[position - 1][0] == driver_id
                )
                if count > 0:
                    result.position_distributions[position][driver_id] = (
                        count / len(self.runs)
                    )

        # Calculate DNF rates
        for driver_id in all_driver_ids:
            dnf_count = sum(1 for run in self.runs if driver_id in run.dnf_drivers)
            result.dnf_rates[driver_id] = dnf_count / len(self.runs)

        # Calculate average pit stops
        for driver_id in all_driver_ids:
            total_stops = sum(
                run.pit_stop_counts.get(driver_id, 0) for run in self.runs
            )
            result.average_pit_stops[driver_id] = total_stops / len(self.runs)

        # Collect all incidents
        for run in self.runs:
            result.incidents_total.extend(run.incidents)

        return result
