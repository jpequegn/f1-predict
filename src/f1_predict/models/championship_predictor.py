"""Championship prediction and scenario analysis system.

Predicts championship winners and generates championship standing forecasts
using time series models and Monte Carlo simulations.
"""

from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class ChampionshipPredictor:
    """Predict championship outcomes for drivers and constructors."""

    def __init__(self, time_series_models: dict[str, Any] | None = None):
        """Initialize championship predictor.

        Args:
            time_series_models: Dictionary of time series models per driver/team
        """
        self.models = time_series_models or {}
        self.logger = logger.bind(component="championship_predictor")

    def predict_champion_winner(
        self,
        current_standings: pd.DataFrame,
        races_remaining: int,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """Predict championship winner with probability distribution.

        Args:
            current_standings: DataFrame with current points, driver/team
            races_remaining: Races left in season
            confidence_level: Confidence level for predictions

        Returns:
            Dictionary with winner, probabilities, and confidence intervals
        """
        try:
            # Simulate remaining races
            n_simulations = 1000
            final_standings = []

            for _ in range(n_simulations):
                standings_copy = current_standings.copy()

                # Project points forward for each race
                for race_num in range(races_remaining):
                    # Estimate points per race (simplified: use recent form)
                    for idx, row in standings_copy.iterrows():
                        driver = row.get("driver_name")
                        if driver in self.models:
                            pred = self.models[driver].predict(steps_ahead=1)
                            # Convert performance forecast to points (simplified)
                            estimated_points = max(0, pred["forecast"][0] * 2)
                            standings_copy.at[idx, "total_points"] += estimated_points

                final_standings.append(standings_copy.sort_values("total_points", ascending=False))

            # Analyze simulation results
            winner_counts = {}
            for standing in final_standings:
                winner = standing.iloc[0]["driver_name"]
                winner_counts[winner] = winner_counts.get(winner, 0) + 1

            # Calculate probabilities
            winner_probs = {
                driver: count / n_simulations
                for driver, count in sorted(
                    winner_counts.items(), key=lambda x: x[1], reverse=True
                )
            }

            top_winner = max(winner_probs.items(), key=lambda x: x[1])

            return {
                "predicted_champion": top_winner[0],
                "win_probability": float(top_winner[1]),
                "top_5_probabilities": dict(list(winner_probs.items())[:5]),
                "confidence_level": confidence_level,
                "simulations": n_simulations,
                "races_remaining": races_remaining,
            }

        except Exception as e:
            self.logger.error("championship_prediction_failed", error=str(e))
            raise

    def predict_points_trajectory(
        self,
        driver_name: str,
        races_ahead: int = 5,
    ) -> dict[str, Any]:
        """Predict driver's championship points trajectory.

        Args:
            driver_name: Driver identifier
            races_ahead: Races to forecast

        Returns:
            Dictionary with points forecast
        """
        if driver_name not in self.models:
            return {"error": f"No model for {driver_name}"}

        try:
            model = self.models[driver_name]
            forecast = model.predict(steps_ahead=races_ahead)

            # Convert performance to points (simplified)
            points_forecast = np.maximum(forecast["forecast"] * 2, 0)

            return {
                "driver": driver_name,
                "races_ahead": races_ahead,
                "points_forecast": points_forecast.tolist(),
                "total_projected": float(np.sum(points_forecast)),
                "average_per_race": float(np.mean(points_forecast)),
                "trend": "improving"
                if points_forecast[-1] > points_forecast[0]
                else "declining",
            }

        except Exception as e:
            self.logger.error("trajectory_prediction_failed", error=str(e))
            raise

    def predict_points_gap_trend(
        self,
        leader_name: str,
        challenger_name: str,
        races_ahead: int = 5,
    ) -> dict[str, Any]:
        """Predict gap between leader and challenger.

        Args:
            leader_name: Leader driver
            challenger_name: Challenger driver
            races_ahead: Races to analyze

        Returns:
            Dictionary with gap trend analysis
        """
        if leader_name not in self.models or challenger_name not in self.models:
            return {"error": "Missing models"}

        try:
            leader_forecast = self.models[leader_name].predict(steps_ahead=races_ahead)
            challenger_forecast = self.models[challenger_name].predict(
                steps_ahead=races_ahead
            )

            leader_points = np.maximum(leader_forecast["forecast"] * 2, 0)
            challenger_points = np.maximum(challenger_forecast["forecast"] * 2, 0)

            gap_trend = leader_points - challenger_points

            return {
                "leader": leader_name,
                "challenger": challenger_name,
                "gap_trend": gap_trend.tolist(),
                "gap_change": float(gap_trend[-1] - gap_trend[0]),
                "narrowing": gap_trend[-1] < gap_trend[0],
                "average_gap": float(np.mean(gap_trend)),
            }

        except Exception as e:
            self.logger.error("gap_trend_failed", error=str(e))
            raise

    def simulate_championship_scenarios(
        self,
        current_standings: pd.DataFrame,
        scenarios: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Simulate different championship scenarios (e.g., DNF, upgrade, etc.).

        Args:
            current_standings: Current championship standings
            scenarios: List of scenario definitions

        Returns:
            List of scenario outcomes
        """
        results = []

        for scenario in scenarios:
            try:
                standings_modified = current_standings.copy()
                scenario_name = scenario.get("name", "unknown")
                modifications = scenario.get("modifications", {})

                # Apply scenario modifications
                for driver, adjustment in modifications.items():
                    mask = standings_modified["driver_name"] == driver
                    if mask.any():
                        standings_modified.loc[mask, "total_points"] += adjustment

                # Predict outcome
                outcome = self.predict_champion_winner(
                    standings_modified, races_remaining=5
                )
                outcome["scenario"] = scenario_name

                results.append(outcome)

            except Exception as e:
                self.logger.warning(
                    "scenario_simulation_failed", scenario=scenario.get("name"), error=str(e)
                )
                continue

        return results

    def estimate_points_needed(
        self,
        driver_name: str,
        current_points: float,
        leader_points: float,
        races_remaining: int,
        max_points_per_race: float = 25.0,
    ) -> dict[str, Any]:
        """Estimate if driver can catch championship leader.

        Args:
            driver_name: Driver identifier
            current_points: Current points
            leader_points: Leader's current points
            races_remaining: Races left
            max_points_per_race: Maximum points per race (25 for win)

        Returns:
            Dictionary with catchup analysis
        """
        try:
            gap = leader_points - current_points
            required_per_race = gap / races_remaining
            leader_avg = leader_points / max(1, sum(1 for _ in range(races_remaining)))

            probability = (
                1.0
                if required_per_race < max_points_per_race
                else max(0.0, 1.0 - (required_per_race / (max_points_per_race * 2)))
            )

            return {
                "driver": driver_name,
                "gap_to_leader": float(gap),
                "races_remaining": races_remaining,
                "required_per_race": float(required_per_race),
                "max_possible_per_race": float(max_points_per_race),
                "catchup_probability": float(probability),
                "feasible": required_per_race <= max_points_per_race,
            }

        except Exception as e:
            self.logger.error("catchup_estimation_failed", error=str(e))
            raise
