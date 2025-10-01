"""Historical context provider for enriching race analysis with relevant historical data."""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import structlog

from f1_predict.analysis.base import BaseAnalyzer
from f1_predict.llm.base import BaseLLMProvider

logger = structlog.get_logger(__name__)


class HistoricalContextProvider(BaseAnalyzer):
    """Provide relevant historical context for race analysis.

    Retrieves and ranks historical facts, patterns, and statistics to enrich
    race previews and predictions with meaningful context.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        data_dir: Optional[Path] = None,
    ):
        """Initialize historical context provider.

        Args:
            llm_provider: LLM provider (not used directly, maintained for interface)
            data_dir: Directory containing historical race data
        """
        super().__init__(llm_provider)
        self.data_dir = data_dir or Path(__file__).parents[3] / "data" / "processed"
        self._load_historical_data()

    def _load_historical_data(self) -> None:
        """Load historical race data from processed files."""
        try:
            race_results_file = self.data_dir / "race_results.csv"
            if race_results_file.exists():
                self.race_results = pd.read_csv(race_results_file)
                self.logger.info(
                    "historical_data_loaded",
                    races=len(self.race_results),
                )
            else:
                self.logger.warning(
                    "historical_data_not_found", file=str(race_results_file)
                )
                self.race_results = pd.DataFrame()

        except Exception as e:
            self.logger.error("failed_to_load_historical_data", error=str(e))
            self.race_results = pd.DataFrame()

    async def generate(
        self,
        circuit_name: str,
        driver_names: Optional[list[str]] = None,
        team_names: Optional[list[str]] = None,
        year: Optional[int] = None,
        max_facts: int = 5,
    ) -> dict[str, Any]:
        """Generate historical context for a race.

        Args:
            circuit_name: Name of the circuit
            driver_names: List of driver names to focus on (optional)
            team_names: List of team names to focus on (optional)
            year: Current year for recency weighting (optional)
            max_facts: Maximum number of facts to return

        Returns:
            Dictionary containing historical context and facts

        Raises:
            ValueError: If circuit_name is empty
        """
        if not circuit_name:
            msg = "Circuit name is required"
            raise ValueError(msg)

        year = year or datetime.now().year

        self.logger.info(
            "generating_historical_context",
            circuit=circuit_name,
            drivers=driver_names,
            year=year,
        )

        context = {
            "circuit_name": circuit_name,
            "circuit_facts": self._get_circuit_facts(circuit_name, max_facts),
            "driver_facts": self._get_driver_facts(
                circuit_name, driver_names, max_facts
            )
            if driver_names
            else [],
            "team_facts": self._get_team_facts(circuit_name, team_names, max_facts)
            if team_names
            else [],
            "interesting_patterns": self._find_patterns(circuit_name),
            "relevant_milestones": self._find_milestones(
                circuit_name, driver_names, year
            ),
        }

        return self._add_metadata(context)

    def _get_circuit_facts(
        self, circuit_name: str, max_facts: int = 5
    ) -> list[dict[str, Any]]:
        """Get historical facts about the circuit.

        Args:
            circuit_name: Circuit name
            max_facts: Maximum facts to return

        Returns:
            List of circuit fact dictionaries with relevance scores
        """
        if self.race_results.empty:
            return []

        facts = []

        # Filter races at this circuit
        circuit_races = self.race_results[
            self.race_results["circuit_name"].str.contains(
                circuit_name, case=False, na=False
            )
        ]

        if circuit_races.empty:
            return facts

        # Most successful driver at circuit
        driver_wins = circuit_races[circuit_races["position"] == 1][
            "driver_name"
        ].value_counts()
        if not driver_wins.empty:
            top_driver = driver_wins.index[0]
            win_count = driver_wins.iloc[0]
            facts.append(
                {
                    "fact": f"{top_driver} has won {win_count} time{'s' if win_count > 1 else ''} at {circuit_name}",
                    "relevance_score": 0.95,
                    "category": "driver_record",
                    "data": {"driver": top_driver, "wins": int(win_count)},
                }
            )

        # Pole position conversion rate
        if "grid_position" in circuit_races.columns:
            pole_races = circuit_races[circuit_races["grid_position"] == 1]
            if len(pole_races) > 0:
                pole_wins = len(pole_races[pole_races["position"] == 1])
                conversion_rate = (pole_wins / len(pole_races)) * 100
                facts.append(
                    {
                        "fact": f"Pole sitter has won {conversion_rate:.0f}% of races at {circuit_name}",
                        "relevance_score": 0.85,
                        "category": "pole_conversion",
                        "data": {"conversion_rate": conversion_rate},
                    }
                )

        # Most successful team
        if "team_name" in circuit_races.columns:
            team_wins = circuit_races[circuit_races["position"] == 1][
                "team_name"
            ].value_counts()
            if not team_wins.empty:
                top_team = team_wins.index[0]
                team_win_count = team_wins.iloc[0]
                facts.append(
                    {
                        "fact": f"{top_team} has won {team_win_count} race{'s' if team_win_count > 1 else ''} at {circuit_name}",
                        "relevance_score": 0.80,
                        "category": "team_record",
                        "data": {"team": top_team, "wins": int(team_win_count)},
                    }
                )

        # Recent years data
        recent_years = (
            circuit_races["year"].max() if "year" in circuit_races.columns else None
        )
        if recent_years:
            facts.append(
                {
                    "fact": f"Last race at {circuit_name} was in {int(recent_years)}",
                    "relevance_score": 0.75,
                    "category": "recency",
                    "data": {"last_year": int(recent_years)},
                }
            )

        # Sort by relevance and limit
        facts.sort(key=lambda x: x["relevance_score"], reverse=True)
        return facts[:max_facts]

    def _get_driver_facts(
        self,
        circuit_name: str,
        driver_names: list[str],
        max_facts: int = 5,
    ) -> list[dict[str, Any]]:
        """Get historical facts about specific drivers at this circuit.

        Args:
            circuit_name: Circuit name
            driver_names: List of driver names
            max_facts: Maximum facts per driver

        Returns:
            List of driver-specific facts
        """
        if self.race_results.empty:
            return []

        facts = []
        circuit_races = self.race_results[
            self.race_results["circuit_name"].str.contains(
                circuit_name, case=False, na=False
            )
        ]

        for driver in driver_names:
            driver_races = circuit_races[
                circuit_races["driver_name"].str.contains(driver, case=False, na=False)
            ]

            if driver_races.empty:
                continue

            # Driver wins at circuit
            wins = len(driver_races[driver_races["position"] == 1])
            if wins > 0:
                facts.append(
                    {
                        "fact": f"{driver} has won {wins} time{'s' if wins > 1 else ''} at {circuit_name}",
                        "relevance_score": 0.90,
                        "category": "driver_wins",
                        "driver": driver,
                        "data": {"wins": wins},
                    }
                )

            # Average finish
            avg_finish = driver_races["position"].mean()
            facts.append(
                {
                    "fact": f"{driver}'s average finish at {circuit_name}: P{avg_finish:.1f}",
                    "relevance_score": 0.75,
                    "category": "driver_average",
                    "driver": driver,
                    "data": {"avg_finish": float(avg_finish)},
                }
            )

            # Podiums
            podiums = len(driver_races[driver_races["position"] <= 3])
            if podiums > 0:
                facts.append(
                    {
                        "fact": f"{driver} has {podiums} podium{'s' if podiums > 1 else ''} at {circuit_name}",
                        "relevance_score": 0.80,
                        "category": "driver_podiums",
                        "driver": driver,
                        "data": {"podiums": podiums},
                    }
                )

        facts.sort(key=lambda x: x["relevance_score"], reverse=True)
        return facts[:max_facts]

    def _get_team_facts(
        self,
        circuit_name: str,
        team_names: list[str],
        max_facts: int = 5,
    ) -> list[dict[str, Any]]:
        """Get historical facts about teams at this circuit.

        Args:
            circuit_name: Circuit name
            team_names: List of team names
            max_facts: Maximum facts per team

        Returns:
            List of team-specific facts
        """
        if self.race_results.empty or "team_name" not in self.race_results.columns:
            return []

        facts = []
        circuit_races = self.race_results[
            self.race_results["circuit_name"].str.contains(
                circuit_name, case=False, na=False
            )
        ]

        for team in team_names:
            team_races = circuit_races[
                circuit_races["team_name"].str.contains(team, case=False, na=False)
            ]

            if team_races.empty:
                continue

            # Team wins
            wins = len(team_races[team_races["position"] == 1])
            if wins > 0:
                facts.append(
                    {
                        "fact": f"{team} has won {wins} race{'s' if wins > 1 else ''} at {circuit_name}",
                        "relevance_score": 0.85,
                        "category": "team_wins",
                        "team": team,
                        "data": {"wins": wins},
                    }
                )

        facts.sort(key=lambda x: x["relevance_score"], reverse=True)
        return facts[:max_facts]

    def _find_patterns(self, circuit_name: str) -> list[str]:
        """Find interesting patterns at the circuit.

        Args:
            circuit_name: Circuit name

        Returns:
            List of pattern descriptions
        """
        if self.race_results.empty:
            return []

        patterns = []
        circuit_races = self.race_results[
            self.race_results["circuit_name"].str.contains(
                circuit_name, case=False, na=False
            )
        ]

        if circuit_races.empty:
            return patterns

        # Pole position win rate
        if "grid_position" in circuit_races.columns:
            pole_races = circuit_races[circuit_races["grid_position"] == 1]
            if len(pole_races) >= 5:
                pole_wins = len(pole_races[pole_races["position"] == 1])
                win_rate = (pole_wins / len(pole_races)) * 100
                if win_rate >= 60:
                    patterns.append(
                        f"Pole sitter has won {win_rate:.0f}% of races at {circuit_name}"
                    )

        # Recent dominance
        if "year" in circuit_races.columns:
            recent_races = circuit_races[
                circuit_races["year"] >= circuit_races["year"].max() - 3
            ]
            if not recent_races.empty and len(recent_races) >= 3:
                winner_counts = recent_races[recent_races["position"] == 1][
                    "driver_name"
                ].value_counts()
                if not winner_counts.empty and winner_counts.iloc[0] >= 2:
                    dominant_driver = winner_counts.index[0]
                    patterns.append(
                        f"{dominant_driver} has dominated recent races at {circuit_name}"
                    )

        return patterns[:3]

    def _find_milestones(
        self,
        circuit_name: str,
        driver_names: Optional[list[str]],
        year: int,
    ) -> list[str]:
        """Find relevant milestones for upcoming race.

        Args:
            circuit_name: Circuit name
            driver_names: Driver names to check
            year: Current year

        Returns:
            List of milestone descriptions
        """
        milestones = []

        if not self.race_results.empty and driver_names:
            for driver in driver_names:
                driver_races = self.race_results[
                    self.race_results["driver_name"].str.contains(
                        driver, case=False, na=False
                    )
                ]

                if not driver_races.empty:
                    total_wins = len(driver_races[driver_races["position"] == 1])
                    # Check for round number milestones
                    if total_wins in [49, 99, 149]:  # One win away from milestone
                        milestone = total_wins + 1
                        milestones.append(
                            f"{driver} could reach {milestone} career wins"
                        )

        return milestones[:3]
