"""Export utilities for simulation results."""

import io
import csv
from datetime import datetime
from typing import BinaryIO, List, Dict, Any

import pandas as pd

from f1_predict.simulation.engine.simulator import SimulationResult
from f1_predict.simulation.analysis.sensitivity_analyzer import SensitivityResult


def export_simulation_to_csv(result: SimulationResult, drivers: List[Dict[str, Any]]) -> str:
    """Export simulation results to CSV format.

    Args:
        result: SimulationResult from simulator
        drivers: List of driver information dicts with driver_id and driver_name

    Returns:
        CSV string with simulation results
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["F1 Race Simulation Results"])
    writer.writerow([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
    writer.writerow([f"Number of Simulations: {result.n_runs}"])
    writer.writerow([])

    # Results summary
    writer.writerow(["Driver ID", "Driver Name", "Win Probability", "Podium Probability",
                    "Finish Probability", "DNF Rate", "Avg Pit Stops"])

    driver_map = {d.get("driver_id", d.get("id")): d for d in drivers}

    for driver in drivers:
        driver_id = driver.get("driver_id", driver.get("id"))
        driver_name = driver.get("driver_name", driver.get("name", driver_id))

        win_prob = result.get_winner_probability(driver_id)
        podium_prob = result.get_podium_probability(driver_id)
        finish_prob = result.finish_probabilities.get(driver_id, 0.0)
        dnf_rate = result.dnf_rates.get(driver_id, 0.0)
        avg_pit_stops = result.average_pit_stops.get(driver_id, 0.0)

        writer.writerow([
            driver_id,
            driver_name,
            f"{win_prob:.4f}",
            f"{podium_prob:.4f}",
            f"{finish_prob:.4f}",
            f"{dnf_rate:.4f}",
            f"{avg_pit_stops:.2f}"
        ])

    writer.writerow([])

    # Position distributions
    writer.writerow(["Finishing Position Probabilities"])
    writer.writerow([])

    for position in sorted(result.position_distributions.keys())[:10]:
        writer.writerow([f"Position {position}"])
        position_probs = result.position_distributions[position]
        for driver_id in sorted(position_probs.keys()):
            prob = position_probs[driver_id]
            if prob > 0:
                driver_name = driver_map.get(driver_id, {}).get("driver_name",
                                                               driver_map.get(driver_id, {}).get("name", driver_id))
                writer.writerow(["", driver_id, driver_name, f"{prob:.4f}"])

    return output.getvalue()


def export_sensitivity_to_csv(result: SensitivityResult) -> str:
    """Export sensitivity analysis results to CSV format.

    Args:
        result: SensitivityResult from analyzer

    Returns:
        CSV string with sensitivity analysis results
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["F1 Race Sensitivity Analysis Results"])
    writer.writerow([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
    writer.writerow([f"Parameter: {result.parameter_name}"])
    writer.writerow([f"Number of Simulations per Point: {result.n_simulations}"])
    writer.writerow([])

    # Results by parameter value
    writer.writerow(["Parameter Value", "Driver", "Win Probability", "Podium Probability",
                    "Finish Probability", "DNF Rate"])

    for param_val in sorted(result.results.keys()):
        simulation_result = result.results[param_val]

        # Get unique drivers from first result
        if simulation_result.finish_probabilities:
            for driver_id in sorted(simulation_result.finish_probabilities.keys()):
                win_prob = simulation_result.get_winner_probability(driver_id)
                podium_prob = simulation_result.get_podium_probability(driver_id)
                finish_prob = simulation_result.finish_probabilities.get(driver_id, 0.0)
                dnf_rate = simulation_result.dnf_rates.get(driver_id, 0.0)

                writer.writerow([
                    f"{param_val:.4f}",
                    driver_id,
                    f"{win_prob:.4f}",
                    f"{podium_prob:.4f}",
                    f"{finish_prob:.4f}",
                    f"{dnf_rate:.4f}"
                ])

    return output.getvalue()


def create_simulation_dataframe(result: SimulationResult, drivers: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create pandas DataFrame from simulation results.

    Args:
        result: SimulationResult from simulator
        drivers: List of driver information dicts

    Returns:
        DataFrame with simulation results
    """
    data = []
    driver_map = {d.get("driver_id", d.get("id")): d for d in drivers}

    for driver in drivers:
        driver_id = driver.get("driver_id", driver.get("id"))
        driver_name = driver.get("driver_name", driver.get("name", driver_id))

        data.append({
            "Driver ID": driver_id,
            "Driver Name": driver_name,
            "Win Probability": result.get_winner_probability(driver_id),
            "Podium Probability": result.get_podium_probability(driver_id),
            "Finish Probability": result.finish_probabilities.get(driver_id, 0.0),
            "DNF Rate": result.dnf_rates.get(driver_id, 0.0),
            "Average Pit Stops": result.average_pit_stops.get(driver_id, 0.0),
        })

    return pd.DataFrame(data)


def create_sensitivity_dataframe(result: SensitivityResult) -> pd.DataFrame:
    """Create pandas DataFrame from sensitivity analysis results.

    Args:
        result: SensitivityResult from analyzer

    Returns:
        DataFrame with sensitivity analysis results
    """
    data = []

    for param_val in sorted(result.results.keys()):
        simulation_result = result.results[param_val]

        if simulation_result.finish_probabilities:
            for driver_id in sorted(simulation_result.finish_probabilities.keys()):
                data.append({
                    "Parameter Value": param_val,
                    "Driver ID": driver_id,
                    "Win Probability": simulation_result.get_winner_probability(driver_id),
                    "Podium Probability": simulation_result.get_podium_probability(driver_id),
                    "Finish Probability": simulation_result.finish_probabilities.get(driver_id, 0.0),
                    "DNF Rate": simulation_result.dnf_rates.get(driver_id, 0.0),
                })

    return pd.DataFrame(data)
