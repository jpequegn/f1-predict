"""Feature engineering for F1 prediction models.

This module provides feature calculators for various aspects of F1 racing:
- Driver form scores based on recent race performance
- Team reliability metrics
- Track-specific performance indicators
- Qualifying vs race performance gaps
- Weather impact features (placeholder for future data)
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class DriverFormCalculator:
    """Calculate driver form scores based on recent race performance.

    Form score is calculated using:
    - Recent race finishing positions (weighted by recency)
    - Points scored trend
    - Consistency (standard deviation of positions)
    - DNF rate
    """

    def __init__(self, window_size: int = 5, recency_weight: float = 0.7):
        """Initialize driver form calculator.

        Args:
            window_size: Number of recent races to consider for form
            recency_weight: Weight factor for more recent races (0-1)
        """
        self.window_size = window_size
        self.recency_weight = recency_weight
        self.logger = logger.bind(calculator="driver_form")

    def calculate_form_score(
        self,
        race_results: pd.DataFrame,
        driver_id: str,
        up_to_date: Optional[datetime] = None,
    ) -> float:
        """Calculate form score for a driver.

        Args:
            race_results: DataFrame with race results
            driver_id: Driver identifier
            up_to_date: Calculate form up to this date (for historical analysis)

        Returns:
            Form score between 0 and 100 (higher is better)
        """
        driver_races = race_results[race_results["driver_id"] == driver_id].copy()

        if up_to_date:
            driver_races = driver_races[driver_races["date"] <= up_to_date]

        driver_races = driver_races.sort_values("date").tail(self.window_size)

        if len(driver_races) == 0:
            self.logger.warning("no_races_found", driver_id=driver_id)
            return 50.0  # Neutral score

        # Calculate weighted position score (lower position = better)
        positions = driver_races["position"].values
        weights = self._calculate_recency_weights(len(positions))
        weights_array = pd.Series(weights)

        # Convert positions to scores (1st = 100, 20th = 0)
        position_scores = 100 * (1 - (positions - 1) / 20)
        position_scores = pd.Series(position_scores).clip(0, 100)

        weighted_position_score = (
            position_scores * weights_array
        ).sum() / weights_array.sum()

        # Calculate consistency penalty (higher std = lower consistency)
        consistency_penalty = positions.std() * 2  # Scale to 0-40 range
        consistency_score = max(0, 100 - consistency_penalty)

        # Calculate DNF penalty
        dnf_count = (driver_races["status_id"].isin([3, 4, 5, 6])).sum()
        dnf_rate = dnf_count / len(driver_races)
        dnf_penalty = dnf_rate * 30  # Up to 30 point penalty

        # Combined form score (60% position, 25% consistency, 15% reliability)
        form_score = (
            weighted_position_score * 0.60
            + consistency_score * 0.25
            + (100 - dnf_penalty) * 0.15
        )

        self.logger.debug(
            "form_calculated",
            driver_id=driver_id,
            races_analyzed=len(driver_races),
            form_score=round(form_score, 2),
        )

        return round(form_score, 2)

    def calculate_form_features(
        self, race_results: pd.DataFrame, up_to_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Calculate form scores for all drivers.

        Args:
            race_results: DataFrame with race results
            up_to_date: Calculate form up to this date

        Returns:
            DataFrame with driver_id and form_score columns
        """
        drivers = race_results["driver_id"].unique()
        form_scores = []

        for driver_id in drivers:
            score = self.calculate_form_score(race_results, driver_id, up_to_date)
            form_scores.append({"driver_id": driver_id, "form_score": score})

        return pd.DataFrame(form_scores)

    def _calculate_recency_weights(self, n: int) -> list[float]:
        """Calculate exponential recency weights."""
        return [self.recency_weight**i for i in range(n - 1, -1, -1)]


class TeamReliabilityCalculator:
    """Calculate team reliability metrics.

    Metrics include:
    - Finish rate (races completed vs DNF)
    - Average finishing position for completed races
    - Mechanical failure rate
    - Points consistency
    """

    def __init__(self, window_size: int = 10):
        """Initialize team reliability calculator.

        Args:
            window_size: Number of recent races to consider
        """
        self.window_size = window_size
        self.logger = logger.bind(calculator="team_reliability")

    def calculate_reliability_metrics(
        self,
        race_results: pd.DataFrame,
        constructor_id: str,
        up_to_date: Optional[datetime] = None,
    ) -> dict[str, float]:
        """Calculate reliability metrics for a team.

        Args:
            race_results: DataFrame with race results
            constructor_id: Constructor/team identifier
            up_to_date: Calculate metrics up to this date

        Returns:
            Dictionary with reliability metrics
        """
        team_races = race_results[
            race_results["constructor_id"] == constructor_id
        ].copy()

        if up_to_date:
            team_races = team_races[team_races["date"] <= up_to_date]

        team_races = team_races.sort_values("date").tail(self.window_size * 2)

        if len(team_races) == 0:
            self.logger.warning("no_races_found", constructor_id=constructor_id)
            return {
                "finish_rate": 0.5,
                "avg_position": 10.0,
                "mechanical_failure_rate": 0.5,
                "points_consistency": 50.0,
                "reliability_score": 50.0,
            }

        # Calculate finish rate (excluding accidents/driver errors)
        mechanical_failures = team_races["status_id"].isin([3, 4, 5, 6])
        finish_rate = 1 - (mechanical_failures.sum() / len(team_races))

        # Average position for completed races
        completed = team_races[~mechanical_failures]
        avg_position = completed["position"].mean() if len(completed) > 0 else 20.0

        # Mechanical failure rate
        mechanical_failure_rate = mechanical_failures.sum() / len(team_races)

        # Points consistency (std dev of points scored)
        points_std = completed["points"].std() if len(completed) > 0 else 0.0
        points_consistency = max(0, 100 - points_std * 5)  # Scale to 0-100

        # Overall reliability score
        reliability_score = (
            finish_rate * 40  # 40% weight on finishing races
            + (1 - avg_position / 20) * 30  # 30% weight on position
            + points_consistency * 0.30  # 30% weight on consistency
        )

        metrics = {
            "finish_rate": round(finish_rate, 3),
            "avg_position": round(avg_position, 2),
            "mechanical_failure_rate": round(mechanical_failure_rate, 3),
            "points_consistency": round(points_consistency, 2),
            "reliability_score": round(reliability_score, 2),
        }

        self.logger.debug(
            "reliability_calculated",
            constructor_id=constructor_id,
            races_analyzed=len(team_races),
            **metrics,
        )

        return metrics

    def calculate_reliability_features(
        self, race_results: pd.DataFrame, up_to_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Calculate reliability metrics for all teams.

        Args:
            race_results: DataFrame with race results
            up_to_date: Calculate metrics up to this date

        Returns:
            DataFrame with constructor_id and reliability metrics
        """
        constructors = race_results["constructor_id"].unique()
        reliability_data = []

        for constructor_id in constructors:
            metrics = self.calculate_reliability_metrics(
                race_results, constructor_id, up_to_date
            )
            metrics["constructor_id"] = constructor_id
            reliability_data.append(metrics)

        return pd.DataFrame(reliability_data)


class TrackPerformanceCalculator:
    """Calculate track-specific performance indicators.

    Metrics include:
    - Historical performance at specific circuits
    - Track type performance (street, high-speed, technical)
    - Weather adaptation (when data available)
    """

    def __init__(self, min_races: int = 2):
        """Initialize track performance calculator.

        Args:
            min_races: Minimum races at a circuit to calculate meaningful metrics
        """
        self.min_races = min_races
        self.logger = logger.bind(calculator="track_performance")

    def calculate_track_performance(
        self,
        race_results: pd.DataFrame,
        driver_id: str,
        circuit_id: str,
        up_to_date: Optional[datetime] = None,
    ) -> dict[str, float]:
        """Calculate driver performance at a specific circuit.

        Args:
            race_results: DataFrame with race results
            driver_id: Driver identifier
            circuit_id: Circuit identifier
            up_to_date: Calculate performance up to this date

        Returns:
            Dictionary with track performance metrics
        """
        driver_at_track = race_results[
            (race_results["driver_id"] == driver_id)
            & (race_results["circuit_id"] == circuit_id)
        ].copy()

        if up_to_date:
            driver_at_track = driver_at_track[driver_at_track["date"] <= up_to_date]

        if len(driver_at_track) < self.min_races:
            return {
                "avg_position": 10.0,
                "avg_points": 5.0,
                "best_position": 20,
                "races_at_track": len(driver_at_track),
                "track_performance_score": 50.0,
            }

        avg_position = driver_at_track["position"].mean()
        avg_points = driver_at_track["points"].mean()
        best_position = driver_at_track["position"].min()

        # Track performance score (0-100)
        position_score = 100 * (1 - (avg_position - 1) / 20)
        points_score = (avg_points / 25) * 100  # 25 points = perfect score

        track_performance_score = position_score * 0.6 + points_score * 0.4

        metrics = {
            "avg_position": round(avg_position, 2),
            "avg_points": round(avg_points, 2),
            "best_position": int(best_position),
            "races_at_track": len(driver_at_track),
            "track_performance_score": round(track_performance_score, 2),
        }

        self.logger.debug(
            "track_performance_calculated",
            driver_id=driver_id,
            circuit_id=circuit_id,
            **metrics,
        )

        return metrics

    def calculate_track_features(
        self,
        race_results: pd.DataFrame,
        circuit_id: str,
        up_to_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Calculate track performance for all drivers at a circuit.

        Args:
            race_results: DataFrame with race results
            circuit_id: Circuit identifier
            up_to_date: Calculate performance up to this date

        Returns:
            DataFrame with driver_id and track performance metrics
        """
        drivers = race_results["driver_id"].unique()
        track_data = []

        for driver_id in drivers:
            metrics = self.calculate_track_performance(
                race_results, driver_id, circuit_id, up_to_date
            )
            metrics["driver_id"] = driver_id
            metrics["circuit_id"] = circuit_id
            track_data.append(metrics)

        return pd.DataFrame(track_data)


class QualifyingRaceGapCalculator:
    """Calculate qualifying vs race performance gaps.

    Analyzes the difference between qualifying and race performance
    to identify drivers who:
    - Perform better in races (good racecraft)
    - Perform better in qualifying (one-lap pace)
    """

    def __init__(self, window_size: int = 10):
        """Initialize qualifying-race gap calculator.

        Args:
            window_size: Number of recent races to consider
        """
        self.window_size = window_size
        self.logger = logger.bind(calculator="quali_race_gap")

    def calculate_performance_gap(
        self,
        race_results: pd.DataFrame,
        qualifying_results: pd.DataFrame,
        driver_id: str,
        up_to_date: Optional[datetime] = None,
    ) -> dict[str, float]:
        """Calculate qualifying vs race performance gap.

        Args:
            race_results: DataFrame with race results
            qualifying_results: DataFrame with qualifying results
            driver_id: Driver identifier
            up_to_date: Calculate gap up to this date

        Returns:
            Dictionary with performance gap metrics
        """
        # Merge race and qualifying results
        merged = pd.merge(
            race_results[["season", "round", "driver_id", "position", "date"]],
            qualifying_results[["season", "round", "driver_id", "position"]],
            on=["season", "round", "driver_id"],
            suffixes=("_race", "_quali"),
        )

        driver_data = merged[merged["driver_id"] == driver_id].copy()

        if up_to_date:
            driver_data = driver_data[driver_data["date"] <= up_to_date]

        driver_data = driver_data.sort_values("date").tail(self.window_size)

        if len(driver_data) == 0:
            return {
                "avg_quali_position": 10.0,
                "avg_race_position": 10.0,
                "avg_position_gain": 0.0,
                "position_gain_consistency": 50.0,
                "racecraft_score": 50.0,
            }

        avg_quali_position = driver_data["position_quali"].mean()
        avg_race_position = driver_data["position_race"].mean()

        # Positive gain means improved in race
        driver_data["position_gain"] = (
            driver_data["position_quali"] - driver_data["position_race"]
        )
        avg_position_gain = driver_data["position_gain"].mean()

        # Consistency of gains/losses
        gain_std = driver_data["position_gain"].std()
        gain_consistency = max(0, 100 - gain_std * 10)  # Scale to 0-100

        # Racecraft score (ability to improve position in race)
        # Positive gain = better racecraft
        racecraft_score = 50 + (avg_position_gain * 5)  # Scale around 50
        racecraft_score = max(0, min(100, racecraft_score))

        metrics = {
            "avg_quali_position": round(avg_quali_position, 2),
            "avg_race_position": round(avg_race_position, 2),
            "avg_position_gain": round(avg_position_gain, 2),
            "position_gain_consistency": round(gain_consistency, 2),
            "racecraft_score": round(racecraft_score, 2),
        }

        self.logger.debug(
            "performance_gap_calculated",
            driver_id=driver_id,
            races_analyzed=len(driver_data),
            **metrics,
        )

        return metrics

    def calculate_gap_features(
        self,
        race_results: pd.DataFrame,
        qualifying_results: pd.DataFrame,
        up_to_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Calculate performance gaps for all drivers.

        Args:
            race_results: DataFrame with race results
            qualifying_results: DataFrame with qualifying results
            up_to_date: Calculate gaps up to this date

        Returns:
            DataFrame with driver_id and performance gap metrics
        """
        drivers = race_results["driver_id"].unique()
        gap_data = []

        for driver_id in drivers:
            metrics = self.calculate_performance_gap(
                race_results, qualifying_results, driver_id, up_to_date
            )
            metrics["driver_id"] = driver_id
            gap_data.append(metrics)

        return pd.DataFrame(gap_data)


class WeatherFeatureCalculator:
    """Calculate weather impact features.

    Placeholder for future weather data integration.
    Will analyze performance in:
    - Wet conditions
    - Variable conditions
    - Temperature extremes
    """

    def __init__(self):
        """Initialize weather feature calculator."""
        self.logger = logger.bind(calculator="weather_features")

    def calculate_weather_features(
        self, race_results: pd.DataFrame, weather_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Calculate weather impact features.

        Args:
            race_results: DataFrame with race results
            weather_data: Optional DataFrame with weather conditions

        Returns:
            DataFrame with driver_id and weather performance metrics
        """
        if weather_data is None:
            self.logger.warning("no_weather_data", status="using_placeholder")
            # Return placeholder features
            drivers = race_results["driver_id"].unique()
            return pd.DataFrame(
                {
                    "driver_id": drivers,
                    "wet_performance_score": 50.0,  # Neutral score
                    "variable_conditions_score": 50.0,
                    "temperature_adaptation_score": 50.0,
                }
            )

        # Future implementation when weather data is available
        self.logger.info("weather_features_calculated", drivers=len(weather_data))
        return weather_data


class FeatureEngineer:
    """Main feature engineering orchestrator.

    Coordinates all feature calculators to generate a complete
    feature set for model training.
    """

    def __init__(
        self,
        driver_form_window: int = 5,
        team_reliability_window: int = 10,
        quali_race_window: int = 10,
    ):
        """Initialize feature engineer.

        Args:
            driver_form_window: Window size for driver form calculation
            team_reliability_window: Window size for team reliability
            quali_race_window: Window size for quali-race gap calculation
        """
        self.driver_form = DriverFormCalculator(window_size=driver_form_window)
        self.team_reliability = TeamReliabilityCalculator(
            window_size=team_reliability_window
        )
        self.track_performance = TrackPerformanceCalculator()
        self.quali_race_gap = QualifyingRaceGapCalculator(window_size=quali_race_window)
        self.weather_features = WeatherFeatureCalculator()
        self.logger = logger.bind(component="feature_engineer")

    def generate_features(
        self,
        race_results: pd.DataFrame,
        qualifying_results: pd.DataFrame,
        circuit_id: Optional[str] = None,
        up_to_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Generate all features for model training.

        Args:
            race_results: DataFrame with race results
            qualifying_results: DataFrame with qualifying results
            circuit_id: Optional circuit ID for track-specific features
            up_to_date: Generate features up to this date

        Returns:
            DataFrame with all engineered features
        """
        self.logger.info("generating_features", circuit_id=circuit_id)

        # Calculate driver form
        form_features = self.driver_form.calculate_form_features(
            race_results, up_to_date
        )

        # Calculate team reliability (currently not included in final features)
        # reliability_features = self.team_reliability.calculate_reliability_features(
        #     race_results, up_to_date
        # )

        # Calculate quali-race gap
        gap_features = self.quali_race_gap.calculate_gap_features(
            race_results, qualifying_results, up_to_date
        )

        # Merge features
        features = form_features.merge(gap_features, on="driver_id", how="outer")

        # Add track-specific features if circuit specified
        if circuit_id:
            track_features = self.track_performance.calculate_track_features(
                race_results, circuit_id, up_to_date
            )
            features = features.merge(track_features, on="driver_id", how="left").drop(
                columns=["circuit_id"], errors="ignore"
            )

        # Add weather features (placeholder)
        weather_features = self.weather_features.calculate_weather_features(
            race_results
        )
        features = features.merge(weather_features, on="driver_id", how="left")

        self.logger.info("features_generated", num_drivers=len(features))

        return features

    def save_features(self, features: pd.DataFrame, output_path: Path) -> None:
        """Save generated features to file.

        Args:
            features: DataFrame with engineered features
            output_path: Path to save features (CSV or JSON)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".csv":
            features.to_csv(output_path, index=False)
        elif output_path.suffix == ".json":
            features.to_json(output_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")

        self.logger.info("features_saved", path=str(output_path), rows=len(features))
