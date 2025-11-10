"""Comparison utilities for F1 race data analysis.

This module provides helpers for:
- Driver head-to-head comparisons
- Team performance comparisons
- Statistical aggregation
- Performance metric calculations
"""

from typing import Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class DriverComparison:
    """Handles driver-to-driver comparisons."""

    def __init__(self):
        """Initialize driver comparison handler."""
        self.logger = logger.bind(component="driver_comparison")

    def compare_drivers(
        self,
        race_results: pd.DataFrame,
        driver1_id: str,
        driver2_id: str,
        circuit_filter: Optional[str] = None,
        season_filter: Optional[int] = None,
    ) -> dict:
        """Compare two drivers' performance.

        Args:
            race_results: DataFrame with race results
            driver1_id: First driver identifier
            driver2_id: Second driver identifier
            circuit_filter: Optional circuit filter
            season_filter: Optional season filter

        Returns:
            Dictionary with comparison statistics
        """
        try:
            # Filter data
            filtered_data = race_results[
                race_results["driver_id"].isin([driver1_id, driver2_id])
            ].copy()

            if circuit_filter and circuit_filter != "All Circuits":
                filtered_data = filtered_data[
                    filtered_data.get("circuit", "") == circuit_filter
                ]

            if season_filter:
                filtered_data = filtered_data[
                    filtered_data.get("season", 0) == season_filter
                ]

            # Separate driver data
            d1_data = filtered_data[filtered_data["driver_id"] == driver1_id]
            d2_data = filtered_data[filtered_data["driver_id"] == driver2_id]

            # Calculate statistics
            stats = {
                "driver1_id": driver1_id,
                "driver2_id": driver2_id,
                "wins": {
                    driver1_id: len(d1_data[d1_data.get("position", 999) == 1]),
                    driver2_id: len(d2_data[d2_data.get("position", 999) == 1]),
                },
                "podiums": {
                    driver1_id: len(d1_data[d1_data.get("position", 999) <= 3]),
                    driver2_id: len(d2_data[d2_data.get("position", 999) <= 3]),
                },
                "races_competed": {
                    driver1_id: len(d1_data),
                    driver2_id: len(d2_data),
                },
                "avg_position": {
                    driver1_id: (
                        d1_data.get("position", 0).mean()
                        if len(d1_data) > 0
                        else None
                    ),
                    driver2_id: (
                        d2_data.get("position", 0).mean()
                        if len(d2_data) > 0
                        else None
                    ),
                },
                "dnf_count": {
                    driver1_id: len(d1_data[d1_data.get("status", "") == "DNF"]),
                    driver2_id: len(d2_data[d2_data.get("status", "") == "DNF"]),
                },
                "total_points": {
                    driver1_id: (
                        d1_data.get("points", 0).sum() if len(d1_data) > 0 else 0
                    ),
                    driver2_id: (
                        d2_data.get("points", 0).sum() if len(d2_data) > 0 else 0
                    ),
                },
            }

            self.logger.info(
                "Driver comparison calculated",
                driver1=driver1_id,
                driver2=driver2_id,
                races_d1=len(d1_data),
                races_d2=len(d2_data),
            )
            return stats

        except Exception as e:
            self.logger.error(f"Error comparing drivers: {e}")
            raise

    def get_race_history(
        self, race_results: pd.DataFrame, driver_id: str
    ) -> pd.DataFrame:
        """Get driver's race history.

        Args:
            race_results: DataFrame with race results
            driver_id: Driver identifier

        Returns:
            DataFrame with driver's races
        """
        try:
            driver_races = race_results[race_results["driver_id"] == driver_id]
            return driver_races.sort_values("date", ascending=False)
        except Exception as e:
            self.logger.error(f"Error getting race history: {e}")
            return pd.DataFrame()

    def get_circuit_performance(
        self, race_results: pd.DataFrame, driver_id: str, circuit: str
    ) -> dict:
        """Get driver's performance at a specific circuit.

        Args:
            race_results: DataFrame with race results
            driver_id: Driver identifier
            circuit: Circuit name

        Returns:
            Dictionary with circuit performance stats
        """
        try:
            circuit_races = race_results[
                (race_results["driver_id"] == driver_id)
                & (race_results.get("circuit", "") == circuit)
            ]

            if len(circuit_races) == 0:
                return {
                    "circuit": circuit,
                    "races": 0,
                    "wins": 0,
                    "podiums": 0,
                    "avg_position": None,
                }

            return {
                "circuit": circuit,
                "races": len(circuit_races),
                "wins": len(circuit_races[circuit_races.get("position", 999) == 1]),
                "podiums": len(
                    circuit_races[circuit_races.get("position", 999) <= 3]
                ),
                "avg_position": circuit_races.get("position", 0).mean(),
                "best_finish": circuit_races.get("position", 0).min(),
            }

        except Exception as e:
            self.logger.error(f"Error getting circuit performance: {e}")
            return {}


class TeamComparison:
    """Handles team-to-team comparisons."""

    def __init__(self):
        """Initialize team comparison handler."""
        self.logger = logger.bind(component="team_comparison")

    def compare_teams(
        self,
        race_results: pd.DataFrame,
        team1_id: str,
        team2_id: str,
        season_filter: Optional[int] = None,
    ) -> dict:
        """Compare two teams' performance.

        Args:
            race_results: DataFrame with race results
            team1_id: First team identifier
            team2_id: Second team identifier
            season_filter: Optional season filter

        Returns:
            Dictionary with comparison statistics
        """
        try:
            # Filter data
            filtered_data = race_results[
                race_results.get("team", "").isin([team1_id, team2_id])
            ].copy()

            if season_filter:
                filtered_data = filtered_data[
                    filtered_data.get("season", 0) == season_filter
                ]

            # Separate team data
            t1_data = filtered_data[filtered_data.get("team", "") == team1_id]
            t2_data = filtered_data[filtered_data.get("team", "") == team2_id]

            # Calculate statistics
            stats = {
                "team1_id": team1_id,
                "team2_id": team2_id,
                "wins": {
                    team1_id: len(t1_data[t1_data.get("position", 999) == 1]),
                    team2_id: len(t2_data[t2_data.get("position", 999) == 1]),
                },
                "podiums": {
                    team1_id: len(t1_data[t1_data.get("position", 999) <= 3]),
                    team2_id: len(t2_data[t2_data.get("position", 999) <= 3]),
                },
                "races_competed": {
                    team1_id: len(t1_data),
                    team2_id: len(t2_data),
                },
                "total_points": {
                    team1_id: (
                        t1_data.get("points", 0).sum() if len(t1_data) > 0 else 0
                    ),
                    team2_id: (
                        t2_data.get("points", 0).sum() if len(t2_data) > 0 else 0
                    ),
                },
                "dnf_rate": {
                    team1_id: (
                        (len(t1_data[t1_data.get("status", "") == "DNF"]) / len(t1_data))
                        if len(t1_data) > 0
                        else 0
                    ),
                    team2_id: (
                        (len(t2_data[t2_data.get("status", "") == "DNF"]) / len(t2_data))
                        if len(t2_data) > 0
                        else 0
                    ),
                },
            }

            self.logger.info(
                "Team comparison calculated",
                team1=team1_id,
                team2=team2_id,
                races_t1=len(t1_data),
                races_t2=len(t2_data),
            )
            return stats

        except Exception as e:
            self.logger.error(f"Error comparing teams: {e}")
            raise

    def get_team_standings(
        self, race_results: pd.DataFrame, season: Optional[int] = None
    ) -> pd.DataFrame:
        """Get team standings for a season.

        Args:
            race_results: DataFrame with race results
            season: Optional season filter

        Returns:
            DataFrame with team standings
        """
        try:
            data = race_results.copy()

            if season:
                data = data[data.get("season", 0) == season]

            return (
                data.groupby(data.get("team", ""))
                .agg(
                    {
                        "points": "sum",
                        "position": "count",
                    }
                )
                .rename(columns={"position": "races"})
                .sort_values("points", ascending=False)
            )

        except Exception as e:
            self.logger.error(f"Error getting team standings: {e}")
            return pd.DataFrame()


class StatisticsCalculator:
    """Calculates aggregated statistics for comparisons."""

    @staticmethod
    def calculate_head_to_head_record(
        results1: pd.DataFrame, results2: pd.DataFrame
    ) -> dict:
        """Calculate head-to-head record between two entities.

        Args:
            results1: Results for entity 1
            results2: Results for entity 2

        Returns:
            Dictionary with head-to-head record
        """
        entity1_wins = len(results1[results1.get("position", 999) < results2.get("position", 999)])
        entity2_wins = len(results2[results2.get("position", 999) < results1.get("position", 999)])
        ties = len(results1) - entity1_wins - entity2_wins

        return {
            "entity1_wins": entity1_wins,
            "entity2_wins": entity2_wins,
            "ties": ties,
            "total_races": len(results1),
        }

    @staticmethod
    def calculate_win_rate(results: pd.DataFrame) -> float:
        """Calculate win rate from results.

        Args:
            results: DataFrame with results

        Returns:
            Win rate as decimal (0.0-1.0)
        """
        if len(results) == 0:
            return 0.0

        wins = len(results[results.get("position", 999) == 1])
        return wins / len(results)

    @staticmethod
    def calculate_podium_rate(results: pd.DataFrame) -> float:
        """Calculate podium rate from results.

        Args:
            results: DataFrame with results

        Returns:
            Podium rate as decimal (0.0-1.0)
        """
        if len(results) == 0:
            return 0.0

        podiums = len(results[results.get("position", 999) <= 3])
        return podiums / len(results)

    @staticmethod
    def calculate_dnf_rate(results: pd.DataFrame) -> float:
        """Calculate DNF rate from results.

        Args:
            results: DataFrame with results

        Returns:
            DNF rate as decimal (0.0-1.0)
        """
        if len(results) == 0:
            return 0.0

        dnfs = len(results[results.get("status", "") == "DNF"])
        return dnfs / len(results)

    @staticmethod
    def calculate_avg_points_per_race(results: pd.DataFrame) -> float:
        """Calculate average points per race.

        Args:
            results: DataFrame with results

        Returns:
            Average points per race
        """
        if len(results) == 0:
            return 0.0

        return results.get("points", 0).sum() / len(results)
