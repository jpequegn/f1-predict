"""Momentum and form analysis for F1 performance.

Analyzes driver/team form trajectories, hot/cold streaks, and momentum indicators.
"""

from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class MomentumAnalyzer:
    """Analyze momentum and form trends in driver/team performance."""

    def __init__(self, min_races_for_form: int = 5):
        """Initialize momentum analyzer.

        Args:
            min_races_for_form: Minimum races to calculate form score
        """
        self.min_races_for_form = min_races_for_form
        self.logger = logger.bind(component="momentum_analyzer")

    def calculate_recent_form(
        self,
        performance_history: pd.Series,
        decay_factor: float = 0.7,
    ) -> dict[str, Any]:
        """Calculate recent form with exponential decay weighting.

        Recent races weighted more heavily than older races.

        Args:
            performance_history: Time-ordered performance values
            decay_factor: Weight decay (0.7 = older races 70% as important)

        Returns:
            Dictionary with form metrics
        """
        if len(performance_history) < self.min_races_for_form:
            return {
                "form_score": None,
                "warning": "insufficient_data",
                "races_available": len(performance_history),
            }

        try:
            # Recent races (last 5)
            recent = performance_history.iloc[-5:].values

            # Apply exponential decay weights
            weights = np.array([decay_factor ** (4 - i) for i in range(len(recent))])
            weights /= weights.sum()  # Normalize

            form_score = np.average(recent, weights=weights)

            return {
                "form_score": float(form_score),
                "recent_average": float(np.mean(recent)),
                "trend": "improving"
                if recent[-1] > np.mean(recent[:3])
                else "declining",
                "consistency": float(1 - np.std(recent) / (np.mean(recent) + 1e-6)),
                "races_analyzed": len(recent),
            }

        except Exception as e:
            self.logger.error("form_calculation_failed", error=str(e))
            raise

    def detect_hot_cold_streaks(
        self,
        performance_history: pd.Series,
        threshold: float = 0.75,
        min_streak_length: int = 3,
    ) -> dict[str, Any]:
        """Detect hot and cold performance streaks.

        Args:
            performance_history: Performance values
            threshold: Performance threshold (0-1) to classify as "hot"
            min_streak_length: Minimum races for streak

        Returns:
            Dictionary with streak analysis
        """
        try:
            # Normalize performance to 0-1 range
            min_perf = performance_history.min()
            max_perf = performance_history.max()
            normalized = (performance_history - min_perf) / (max_perf - min_perf + 1e-6)

            # Classify as hot (above threshold) or cold (below)
            is_hot = normalized > threshold

            # Find streaks
            streaks = []
            current_streak = None

            for i, hot in enumerate(is_hot):
                if hot:
                    if current_streak is None or current_streak["type"] == "hot":
                        if current_streak is None:
                            current_streak = {"type": "hot", "start": i, "length": 1}
                        else:
                            current_streak["length"] += 1
                    else:
                        if current_streak["length"] >= min_streak_length:
                            streaks.append(current_streak)
                        current_streak = {"type": "hot", "start": i, "length": 1}
                else:
                    if current_streak is None or current_streak["type"] == "cold":
                        if current_streak is None:
                            current_streak = {"type": "cold", "start": i, "length": 1}
                        else:
                            current_streak["length"] += 1
                    else:
                        if current_streak["length"] >= min_streak_length:
                            streaks.append(current_streak)
                        current_streak = {"type": "cold", "start": i, "length": 1}

            if current_streak and current_streak["length"] >= min_streak_length:
                streaks.append(current_streak)

            # Current streak
            current = (
                {"type": "hot" if is_hot.iloc[-1] else "cold", "length": 1}
                if len(is_hot) > 0
                else None
            )

            return {
                "current_streak": current,
                "all_streaks": streaks,
                "hot_races": int(is_hot.sum()),
                "cold_races": int((~is_hot).sum()),
            }

        except Exception as e:
            self.logger.error("streak_detection_failed", error=str(e))
            raise

    def identify_performance_inflection_points(
        self,
        performance_history: pd.Series,
        min_change: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Identify key turning points in performance.

        Args:
            performance_history: Time-ordered performance values
            min_change: Minimum acceleration to detect inflection

        Returns:
            List of inflection points with details
        """
        try:
            if len(performance_history) < 4:
                return []

            # Calculate acceleration (2nd derivative)
            first_diff = performance_history.diff()
            acceleration = first_diff.diff()

            inflection_points = []

            for i in range(1, len(acceleration)):
                # Sign change in acceleration indicates inflection
                if (
                    not np.isnan(acceleration.iloc[i])
                    and not np.isnan(acceleration.iloc[i - 1])
                ):
                    if acceleration.iloc[i] * acceleration.iloc[i - 1] < 0:
                        if abs(acceleration.iloc[i]) > min_change:
                            inflection_points.append(
                                {
                                    "race_index": i,
                                    "acceleration": float(acceleration.iloc[i]),
                                    "type": "peak" if acceleration.iloc[i] < 0 else "trough",
                                    "value": float(performance_history.iloc[i]),
                                }
                            )

            return inflection_points

        except Exception as e:
            self.logger.error("inflection_detection_failed", error=str(e))
            raise

    def analyze_circuit_type_performance(
        self,
        driver_data: pd.DataFrame,
        circuit_type_column: str,
        performance_column: str,
    ) -> dict[str, Any]:
        """Analyze driver performance by circuit type.

        Args:
            driver_data: DataFrame with circuit and performance data
            circuit_type_column: Column with circuit type (street, oval, etc.)
            performance_column: Column with performance metric

        Returns:
            Dictionary with performance by circuit type
        """
        try:
            circuit_performance = {}

            for circuit_type in driver_data[circuit_type_column].unique():
                mask = driver_data[circuit_type_column] == circuit_type
                performances = driver_data.loc[mask, performance_column]

                circuit_performance[str(circuit_type)] = {
                    "average": float(performances.mean()),
                    "best": float(performances.max()),
                    "worst": float(performances.min()),
                    "std_dev": float(performances.std()),
                    "n_races": int(performances.count()),
                }

            return circuit_performance

        except Exception as e:
            self.logger.error("circuit_analysis_failed", error=str(e))
            raise

    def estimate_team_development_rate(
        self,
        team_performance_history: pd.Series,
        window: int = 3,
    ) -> dict[str, Any]:
        """Estimate team's pace of development.

        Args:
            team_performance_history: Time-ordered team performance
            window: Window size for trend analysis

        Returns:
            Dictionary with development metrics
        """
        try:
            if len(team_performance_history) < window + 1:
                return {"error": "insufficient_data"}

            # Calculate pace of improvement in each window
            development_rates = []
            for i in range(window, len(team_performance_history)):
                window_data = team_performance_history.iloc[i - window : i]
                rate = (window_data.iloc[-1] - window_data.iloc[0]) / window
                development_rates.append(rate)

            development_rates = np.array(development_rates)

            return {
                "average_development_rate": float(np.mean(development_rates)),
                "recent_rate": float(development_rates[-1]),
                "trend": "accelerating"
                if development_rates[-1] > np.mean(development_rates[:-1])
                else "decelerating",
                "development_acceleration": float(np.diff(development_rates)[-1]),
            }

        except Exception as e:
            self.logger.error("development_rate_failed", error=str(e))
            raise

    def compare_momentum(
        self,
        driver1_history: pd.Series,
        driver2_history: pd.Series,
        recent_races: int = 5,
    ) -> dict[str, Any]:
        """Compare momentum between two drivers.

        Args:
            driver1_history: Driver 1 performance history
            driver2_history: Driver 2 performance history
            recent_races: Races to consider for momentum

        Returns:
            Dictionary with comparison
        """
        try:
            d1_recent = driver1_history.iloc[-recent_races:]
            d2_recent = driver2_history.iloc[-recent_races:]

            d1_slope = np.polyfit(np.arange(len(d1_recent)), d1_recent.values, 1)[0]
            d2_slope = np.polyfit(np.arange(len(d2_recent)), d2_recent.values, 1)[0]

            return {
                "driver1_momentum": float(d1_slope),
                "driver2_momentum": float(d2_slope),
                "leader": "driver1" if d1_slope > d2_slope else "driver2",
                "momentum_gap": float(abs(d1_slope - d2_slope)),
                "races_analyzed": recent_races,
            }

        except Exception as e:
            self.logger.error("momentum_comparison_failed", error=str(e))
            raise
