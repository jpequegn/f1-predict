"""Performance metrics calculation for F1 data analysis.

This module provides calculators for various performance metrics:
- Driver championship points trends
- Team performance at specific circuits
- Qualifying position vs final position analysis
- DNF (Did Not Finish) rates and reliability scores
- Head-to-head teammate comparisons
"""

from datetime import datetime
from typing import Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class ChampionshipPointsAnalyzer:
    """Analyze driver championship points trends.

    Calculates various metrics related to points accumulation:
    - Points per race average
    - Points trend (improving/declining)
    - Consistency in points scoring
    - Podium rate
    - Win rate
    """

    def __init__(self, season: Optional[str] = None):
        """Initialize championship points analyzer.

        Args:
            season: Specific season to analyze (None for all seasons)
        """
        self.season = season
        self.logger = logger.bind(analyzer="championship_points")

    def analyze_driver_points_trend(
        self, race_results: pd.DataFrame, driver_id: str
    ) -> dict[str, float]:
        """Analyze championship points trend for a driver.

        Args:
            race_results: DataFrame with race results
            driver_id: Driver identifier

        Returns:
            Dictionary with points trend metrics
        """
        driver_results = race_results[race_results["driver_id"] == driver_id].copy()

        if self.season:
            driver_results = driver_results[driver_results["season"] == self.season]

        if len(driver_results) == 0:
            self.logger.warning("no_results_found", driver_id=driver_id)
            return {
                "total_points": 0.0,
                "avg_points_per_race": 0.0,
                "points_trend": 0.0,
                "podium_rate": 0.0,
                "win_rate": 0.0,
                "consistency_score": 0.0,
            }

        driver_results = driver_results.sort_values("date")

        # Basic stats
        total_points = driver_results["points"].sum()
        avg_points = driver_results["points"].mean()
        races_count = len(driver_results)

        # Calculate trend (linear regression slope of points over time)
        if races_count >= 3:
            x = range(races_count)
            y = driver_results["points"].values
            # Simple linear regression
            x_mean = sum(x) / len(x)
            y_mean = sum(y) / len(y)
            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
            points_trend = numerator / denominator if denominator != 0 else 0.0
        else:
            points_trend = 0.0

        # Podium and win rates
        podiums = (driver_results["position"] <= 3).sum()
        wins = (driver_results["position"] == 1).sum()
        podium_rate = podiums / races_count
        win_rate = wins / races_count

        # Consistency (inverse of coefficient of variation)
        points_std = driver_results["points"].std()
        if avg_points > 0:
            cv = points_std / avg_points
            consistency_score = max(0, 100 - (cv * 100))
        else:
            consistency_score = 0.0

        metrics = {
            "total_points": round(total_points, 1),
            "avg_points_per_race": round(avg_points, 2),
            "points_trend": round(points_trend, 3),
            "podium_rate": round(podium_rate, 3),
            "win_rate": round(win_rate, 3),
            "consistency_score": round(consistency_score, 2),
            "races_count": races_count,
        }

        self.logger.debug("points_trend_analyzed", driver_id=driver_id, **metrics)

        return metrics

    def get_championship_standings(
        self, race_results: pd.DataFrame
    ) -> pd.DataFrame:
        """Get current championship standings.

        Args:
            race_results: DataFrame with race results

        Returns:
            DataFrame with standings (driver_id, total_points, position)
        """
        results = race_results.copy()

        if self.season:
            results = results[results["season"] == self.season]

        # Group by driver and sum points
        standings = (
            results.groupby("driver_id")
            .agg(
                {
                    "points": "sum",
                    "position": lambda x: (x == 1).sum(),  # wins
                }
            )
            .rename(columns={"position": "wins"})
        )

        standings = standings.sort_values("points", ascending=False).reset_index()
        standings["position"] = range(1, len(standings) + 1)

        self.logger.info("standings_calculated", drivers=len(standings))

        return standings


class TeamCircuitAnalyzer:
    """Analyze team performance at specific circuits.

    Calculates metrics for how teams perform at individual tracks:
    - Historical results at circuit
    - Average finishing position
    - Points scored
    - Podium/win rate
    - Reliability at track
    """

    def __init__(self):
        """Initialize team circuit analyzer."""
        self.logger = logger.bind(analyzer="team_circuit")

    def analyze_team_at_circuit(
        self,
        race_results: pd.DataFrame,
        constructor_id: str,
        circuit_id: str,
    ) -> dict[str, float]:
        """Analyze team performance at a specific circuit.

        Args:
            race_results: DataFrame with race results
            constructor_id: Constructor/team identifier
            circuit_id: Circuit identifier

        Returns:
            Dictionary with circuit performance metrics
        """
        team_at_circuit = race_results[
            (race_results["constructor_id"] == constructor_id)
            & (race_results["circuit_id"] == circuit_id)
        ].copy()

        if len(team_at_circuit) == 0:
            self.logger.warning(
                "no_results_found",
                constructor_id=constructor_id,
                circuit_id=circuit_id,
            )
            return {
                "races_count": 0,
                "avg_position": 20.0,
                "best_position": 20,
                "total_points": 0.0,
                "avg_points": 0.0,
                "podium_rate": 0.0,
                "win_rate": 0.0,
                "dnf_rate": 0.0,
            }

        # Basic stats
        races_count = len(team_at_circuit)
        avg_position = team_at_circuit["position"].mean()
        best_position = team_at_circuit["position"].min()
        total_points = team_at_circuit["points"].sum()
        avg_points = team_at_circuit["points"].mean()

        # Podium and win rates
        podiums = (team_at_circuit["position"] <= 3).sum()
        wins = (team_at_circuit["position"] == 1).sum()
        podium_rate = podiums / races_count
        win_rate = wins / races_count

        # DNF rate (mechanical failures)
        dnf_count = team_at_circuit["status_id"].isin([3, 4, 5, 6]).sum()
        dnf_rate = dnf_count / races_count

        metrics = {
            "races_count": races_count,
            "avg_position": round(avg_position, 2),
            "best_position": int(best_position),
            "total_points": round(total_points, 1),
            "avg_points": round(avg_points, 2),
            "podium_rate": round(podium_rate, 3),
            "win_rate": round(win_rate, 3),
            "dnf_rate": round(dnf_rate, 3),
        }

        self.logger.debug(
            "team_circuit_analyzed",
            constructor_id=constructor_id,
            circuit_id=circuit_id,
            **metrics,
        )

        return metrics

    def get_best_circuits_for_team(
        self, race_results: pd.DataFrame, constructor_id: str, top_n: int = 5
    ) -> pd.DataFrame:
        """Get best performing circuits for a team.

        Args:
            race_results: DataFrame with race results
            constructor_id: Constructor/team identifier
            top_n: Number of top circuits to return

        Returns:
            DataFrame with top circuits and their metrics
        """
        team_results = race_results[
            race_results["constructor_id"] == constructor_id
        ].copy()

        if len(team_results) == 0:
            return pd.DataFrame()

        # Group by circuit
        circuit_performance = (
            team_results.groupby("circuit_id")
            .agg(
                {
                    "position": ["mean", "min", "count"],
                    "points": ["sum", "mean"],
                }
            )
            .round(2)
        )

        circuit_performance.columns = [
            "avg_position",
            "best_position",
            "races",
            "total_points",
            "avg_points",
        ]

        # Sort by average points (best circuits first)
        circuit_performance = circuit_performance.sort_values(
            "avg_points", ascending=False
        ).head(top_n)

        self.logger.info(
            "best_circuits_found",
            constructor_id=constructor_id,
            circuits=len(circuit_performance),
        )

        return circuit_performance.reset_index()


class QualifyingAnalyzer:
    """Analyze qualifying position vs final position.

    Calculates detailed analysis of qualifying performance:
    - Average qualifying position
    - Average race finish position
    - Position change (qualifying to race)
    - Ability to overtake
    - Ability to defend position
    """

    def __init__(self, window_size: int = 10):
        """Initialize qualifying analyzer.

        Args:
            window_size: Number of recent races to analyze
        """
        self.window_size = window_size
        self.logger = logger.bind(analyzer="qualifying")

    def analyze_driver_qualifying(
        self,
        race_results: pd.DataFrame,
        qualifying_results: pd.DataFrame,
        driver_id: str,
    ) -> dict[str, float]:
        """Analyze driver's qualifying vs race performance.

        Args:
            race_results: DataFrame with race results
            qualifying_results: DataFrame with qualifying results
            driver_id: Driver identifier

        Returns:
            Dictionary with qualifying analysis metrics
        """
        # Merge race and qualifying results
        merged = pd.merge(
            race_results[["season", "round", "driver_id", "position", "date"]],
            qualifying_results[["season", "round", "driver_id", "position"]],
            on=["season", "round", "driver_id"],
            suffixes=("_race", "_quali"),
        )

        driver_data = merged[merged["driver_id"] == driver_id].copy()
        driver_data = driver_data.sort_values("date").tail(self.window_size)

        if len(driver_data) == 0:
            return {
                "avg_quali_position": 10.0,
                "avg_race_position": 10.0,
                "avg_position_change": 0.0,
                "positions_gained_rate": 0.0,
                "positions_lost_rate": 0.0,
                "overtaking_score": 50.0,
            }

        avg_quali_position = driver_data["position_quali"].mean()
        avg_race_position = driver_data["position_race"].mean()

        # Calculate position changes
        driver_data["position_change"] = (
            driver_data["position_quali"] - driver_data["position_race"]
        )
        avg_position_change = driver_data["position_change"].mean()

        # Rate of gaining/losing positions
        positions_gained = driver_data[driver_data["position_change"] > 0]
        positions_lost = driver_data[driver_data["position_change"] < 0]

        positions_gained_rate = len(positions_gained) / len(driver_data)
        positions_lost_rate = len(positions_lost) / len(driver_data)

        # Overtaking score (0-100, higher is better)
        # Based on average position gain and consistency
        overtaking_score = 50 + (avg_position_change * 5)
        overtaking_score = max(0, min(100, overtaking_score))

        metrics = {
            "avg_quali_position": round(avg_quali_position, 2),
            "avg_race_position": round(avg_race_position, 2),
            "avg_position_change": round(avg_position_change, 2),
            "positions_gained_rate": round(positions_gained_rate, 3),
            "positions_lost_rate": round(positions_lost_rate, 3),
            "overtaking_score": round(overtaking_score, 2),
            "races_analyzed": len(driver_data),
        }

        self.logger.debug("qualifying_analyzed", driver_id=driver_id, **metrics)

        return metrics


class DNFReliabilityAnalyzer:
    """Analyze DNF rates and reliability scores.

    Calculates comprehensive reliability metrics:
    - DNF rate (overall and by type)
    - Finish rate
    - Mechanical failure rate
    - Accident rate
    - Reliability score
    """

    def __init__(self, window_size: int = 20):
        """Initialize DNF reliability analyzer.

        Args:
            window_size: Number of recent races to analyze
        """
        self.window_size = window_size
        self.logger = logger.bind(analyzer="dnf_reliability")

    def analyze_driver_reliability(
        self, race_results: pd.DataFrame, driver_id: str
    ) -> dict[str, float]:
        """Analyze driver reliability (DNF rates).

        Args:
            race_results: DataFrame with race results
            driver_id: Driver identifier

        Returns:
            Dictionary with reliability metrics
        """
        driver_results = race_results[race_results["driver_id"] == driver_id].copy()
        driver_results = driver_results.sort_values("date").tail(self.window_size)

        if len(driver_results) == 0:
            return {
                "races_count": 0,
                "finish_rate": 0.5,
                "dnf_rate": 0.5,
                "mechanical_dnf_rate": 0.25,
                "accident_dnf_rate": 0.25,
                "reliability_score": 50.0,
            }

        races_count = len(driver_results)

        # DNF analysis (status_id in [3, 4, 5, 6] are mechanical failures)
        mechanical_dnf = driver_results["status_id"].isin([3, 4, 5, 6])
        # Accident/collision (status_id in [2, 7, 8, 9])
        accident_dnf = driver_results["status_id"].isin([2, 7, 8, 9])

        finish_rate = 1 - ((mechanical_dnf | accident_dnf).sum() / races_count)
        dnf_rate = 1 - finish_rate
        mechanical_dnf_rate = mechanical_dnf.sum() / races_count
        accident_dnf_rate = accident_dnf.sum() / races_count

        # Reliability score (0-100, higher is better)
        # Heavily weighted towards finishing races
        reliability_score = (
            finish_rate * 70  # 70% weight on finishing
            + (1 - mechanical_dnf_rate) * 20  # 20% weight on mechanical reliability
            + (1 - accident_dnf_rate) * 10  # 10% weight on avoiding accidents
        )

        metrics = {
            "races_count": races_count,
            "finish_rate": round(finish_rate, 3),
            "dnf_rate": round(dnf_rate, 3),
            "mechanical_dnf_rate": round(mechanical_dnf_rate, 3),
            "accident_dnf_rate": round(accident_dnf_rate, 3),
            "reliability_score": round(reliability_score, 2),
        }

        self.logger.debug("reliability_analyzed", driver_id=driver_id, **metrics)

        return metrics

    def analyze_team_reliability(
        self, race_results: pd.DataFrame, constructor_id: str
    ) -> dict[str, float]:
        """Analyze team reliability (DNF rates).

        Args:
            race_results: DataFrame with race results
            constructor_id: Constructor/team identifier

        Returns:
            Dictionary with team reliability metrics
        """
        team_results = race_results[
            race_results["constructor_id"] == constructor_id
        ].copy()
        team_results = team_results.sort_values("date").tail(self.window_size * 2)

        if len(team_results) == 0:
            return {
                "races_count": 0,
                "entries_count": 0,
                "finish_rate": 0.5,
                "dnf_rate": 0.5,
                "mechanical_dnf_rate": 0.25,
                "reliability_score": 50.0,
            }

        entries_count = len(team_results)
        races_count = team_results["season"].astype(str) + team_results["round"].astype(str)
        races_count = races_count.nunique()

        # DNF analysis
        mechanical_dnf = team_results["status_id"].isin([3, 4, 5, 6])
        accident_dnf = team_results["status_id"].isin([2, 7, 8, 9])

        finish_rate = 1 - ((mechanical_dnf | accident_dnf).sum() / entries_count)
        dnf_rate = 1 - finish_rate
        mechanical_dnf_rate = mechanical_dnf.sum() / entries_count

        # Team reliability score
        reliability_score = (
            finish_rate * 80  # 80% weight on finishing (more important for teams)
            + (1 - mechanical_dnf_rate) * 20  # 20% weight on mechanical reliability
        )

        metrics = {
            "races_count": races_count,
            "entries_count": entries_count,
            "finish_rate": round(finish_rate, 3),
            "dnf_rate": round(dnf_rate, 3),
            "mechanical_dnf_rate": round(mechanical_dnf_rate, 3),
            "reliability_score": round(reliability_score, 2),
        }

        self.logger.debug("team_reliability_analyzed", constructor_id=constructor_id, **metrics)

        return metrics


class TeammateComparisonAnalyzer:
    """Analyze head-to-head teammate comparisons.

    Calculates comparison metrics between teammates:
    - Head-to-head qualifying record
    - Head-to-head race finishes
    - Points difference
    - Average position difference
    - Consistency comparison
    """

    def __init__(self, season: Optional[str] = None):
        """Initialize teammate comparison analyzer.

        Args:
            season: Specific season to analyze (None for all seasons)
        """
        self.season = season
        self.logger = logger.bind(analyzer="teammate_comparison")

    def compare_teammates(
        self,
        race_results: pd.DataFrame,
        qualifying_results: pd.DataFrame,
        driver1_id: str,
        driver2_id: str,
    ) -> dict[str, float]:
        """Compare two teammates head-to-head.

        Args:
            race_results: DataFrame with race results
            qualifying_results: DataFrame with qualifying results
            driver1_id: First driver identifier
            driver2_id: Second driver identifier

        Returns:
            Dictionary with comparison metrics
        """
        # Filter for season if specified
        if self.season:
            race_results = race_results[race_results["season"] == self.season]
            qualifying_results = qualifying_results[
                qualifying_results["season"] == self.season
            ]

        # Get results for both drivers
        driver1_races = race_results[race_results["driver_id"] == driver1_id].copy()
        driver2_races = race_results[race_results["driver_id"] == driver2_id].copy()

        # Find races where both competed (same season/round)
        common_races = pd.merge(
            driver1_races[["season", "round", "position", "points"]],
            driver2_races[["season", "round", "position", "points"]],
            on=["season", "round"],
            suffixes=("_d1", "_d2"),
        )

        if len(common_races) == 0:
            self.logger.warning(
                "no_common_races",
                driver1_id=driver1_id,
                driver2_id=driver2_id,
            )
            return {
                "common_races": 0,
                "driver1_wins": 0,
                "driver2_wins": 0,
                "driver1_avg_position": 10.0,
                "driver2_avg_position": 10.0,
                "avg_position_diff": 0.0,
                "points_diff": 0.0,
                "driver1_consistency": 50.0,
                "driver2_consistency": 50.0,
            }

        # Head-to-head race finishes
        driver1_wins = (common_races["position_d1"] < common_races["position_d2"]).sum()
        driver2_wins = (common_races["position_d1"] > common_races["position_d2"]).sum()

        # Average positions
        driver1_avg_position = common_races["position_d1"].mean()
        driver2_avg_position = common_races["position_d2"].mean()
        avg_position_diff = driver1_avg_position - driver2_avg_position

        # Points difference
        driver1_points = common_races["points_d1"].sum()
        driver2_points = common_races["points_d2"].sum()
        points_diff = driver1_points - driver2_points

        # Consistency (inverse of std dev)
        driver1_std = common_races["position_d1"].std()
        driver2_std = common_races["position_d2"].std()
        driver1_consistency = max(0, 100 - (driver1_std * 5))
        driver2_consistency = max(0, 100 - (driver2_std * 5))

        # Qualifying comparison
        quali_comparison = self._compare_qualifying(
            qualifying_results, driver1_id, driver2_id
        )

        metrics = {
            "common_races": len(common_races),
            "driver1_wins": int(driver1_wins),
            "driver2_wins": int(driver2_wins),
            "driver1_avg_position": round(driver1_avg_position, 2),
            "driver2_avg_position": round(driver2_avg_position, 2),
            "avg_position_diff": round(avg_position_diff, 2),
            "points_diff": round(points_diff, 1),
            "driver1_points": round(driver1_points, 1),
            "driver2_points": round(driver2_points, 1),
            "driver1_consistency": round(driver1_consistency, 2),
            "driver2_consistency": round(driver2_consistency, 2),
            **quali_comparison,
        }

        self.logger.debug(
            "teammates_compared",
            driver1_id=driver1_id,
            driver2_id=driver2_id,
            **metrics,
        )

        return metrics

    def _compare_qualifying(
        self, qualifying_results: pd.DataFrame, driver1_id: str, driver2_id: str
    ) -> dict[str, int]:
        """Compare qualifying performance between teammates.

        Args:
            qualifying_results: DataFrame with qualifying results
            driver1_id: First driver identifier
            driver2_id: Second driver identifier

        Returns:
            Dictionary with qualifying comparison metrics
        """
        if self.season:
            qualifying_results = qualifying_results[
                qualifying_results["season"] == self.season
            ]

        driver1_quali = qualifying_results[
            qualifying_results["driver_id"] == driver1_id
        ].copy()
        driver2_quali = qualifying_results[
            qualifying_results["driver_id"] == driver2_id
        ].copy()

        # Find common qualifying sessions
        common_quali = pd.merge(
            driver1_quali[["season", "round", "position"]],
            driver2_quali[["season", "round", "position"]],
            on=["season", "round"],
            suffixes=("_d1", "_d2"),
        )

        if len(common_quali) == 0:
            return {
                "common_qualifying": 0,
                "driver1_quali_wins": 0,
                "driver2_quali_wins": 0,
            }

        driver1_quali_wins = (
            common_quali["position_d1"] < common_quali["position_d2"]
        ).sum()
        driver2_quali_wins = (
            common_quali["position_d1"] > common_quali["position_d2"]
        ).sum()

        return {
            "common_qualifying": len(common_quali),
            "driver1_quali_wins": int(driver1_quali_wins),
            "driver2_quali_wins": int(driver2_quali_wins),
        }


class PerformanceMetricsCalculator:
    """Main orchestrator for all performance metrics calculations.

    Coordinates all metric analyzers to generate comprehensive
    performance analysis reports.
    """

    def __init__(self, season: Optional[str] = None):
        """Initialize performance metrics calculator.

        Args:
            season: Specific season to analyze (None for all seasons)
        """
        self.season = season
        self.championship_analyzer = ChampionshipPointsAnalyzer(season=season)
        self.team_circuit_analyzer = TeamCircuitAnalyzer()
        self.qualifying_analyzer = QualifyingAnalyzer()
        self.dnf_analyzer = DNFReliabilityAnalyzer()
        self.teammate_analyzer = TeammateComparisonAnalyzer(season=season)
        self.logger = logger.bind(component="performance_metrics")

    def generate_driver_report(
        self,
        race_results: pd.DataFrame,
        qualifying_results: pd.DataFrame,
        driver_id: str,
    ) -> dict[str, dict]:
        """Generate comprehensive performance report for a driver.

        Args:
            race_results: DataFrame with race results
            qualifying_results: DataFrame with qualifying results
            driver_id: Driver identifier

        Returns:
            Dictionary with all performance metrics
        """
        self.logger.info("generating_driver_report", driver_id=driver_id)

        report = {
            "driver_id": driver_id,
            "season": self.season or "all",
            "championship_points": self.championship_analyzer.analyze_driver_points_trend(
                race_results, driver_id
            ),
            "qualifying_performance": self.qualifying_analyzer.analyze_driver_qualifying(
                race_results, qualifying_results, driver_id
            ),
            "reliability": self.dnf_analyzer.analyze_driver_reliability(
                race_results, driver_id
            ),
        }

        self.logger.info("driver_report_generated", driver_id=driver_id)

        return report

    def generate_team_report(
        self, race_results: pd.DataFrame, constructor_id: str
    ) -> dict[str, dict]:
        """Generate comprehensive performance report for a team.

        Args:
            race_results: DataFrame with race results
            constructor_id: Constructor/team identifier

        Returns:
            Dictionary with all team performance metrics
        """
        self.logger.info("generating_team_report", constructor_id=constructor_id)

        report = {
            "constructor_id": constructor_id,
            "season": self.season or "all",
            "reliability": self.dnf_analyzer.analyze_team_reliability(
                race_results, constructor_id
            ),
            "best_circuits": self.team_circuit_analyzer.get_best_circuits_for_team(
                race_results, constructor_id
            ).to_dict(orient="records"),
        }

        self.logger.info("team_report_generated", constructor_id=constructor_id)

        return report