"""F1-specific anomaly detection for race results."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Constants for anomaly detection thresholds
PODIUM_POSITIONS = (1, 2, 3)
MIN_HISTORY_FOR_ANOMALY_FLAG = 1
MIN_POSITION_THRESHOLD = 5
PODIUM_POSITIONS_COUNT = 3
MIN_HISTORY_FOR_ZSCORE = 3
PODIUM_IMPROVEMENT_SCORE_MULTIPLIER = 0.9
PODIUM_ZSCOPE_DIVISOR = 3.0
PODIUM_STANDARD_THRESHOLD = 10
Q_RACE_GAP_MAX_POSITION = 20
Q_RACE_GAP_THRESHOLD_WORSE = 8
Q_RACE_GAP_THRESHOLD_BETTER = 10
Q_RACE_GAP_BETTER_SCORE_MULTIPLIER = 0.7
DNF_CRITICAL_THRESHOLD = 0.5
DNF_CRITICAL_SCORE = 0.8
DNF_SIGNIFICANT_THRESHOLD = 0.25
DNF_SIGNIFICANT_SCORE = 0.6
DNF_STANDARD_SCORE = 0.3
ANOMALY_SCORE_THRESHOLD = 0.5
ANOMALY_SCORE_BOUNDS = (0.3, 0.5)
PODIUM_ANOMALY_SCORE_THRESHOLD = 0.3
Q_RACE_ANOMALY_SCORE_THRESHOLD = 0.3
DNF_ANOMALY_SCORE_THRESHOLD = 0.3


@dataclass
class DriverStats:
    """Statistics for a driver's historical performance."""

    driver_id: int
    driver_name: str
    avg_position: float = 0.0
    std_position: float = 0.0
    dnf_count: int = 0
    race_count: int = 0
    best_position: int = 20
    worst_position: int = 1
    positions: list[int] = field(default_factory=list)


class RaceAnomalyDetector:
    """F1-specific anomaly detection for race results.

    Detects domain-specific anomalies like unusual podium results,
    qualifying vs race performance gaps, and DNF patterns. Maintains
    driver history to identify deviations from typical performance.

    Key detections:
        - Podium anomalies: Significant performance deviations
        - Q-to-Race gaps: Qualifying vs race position changes
        - DNF patterns: Multiple consecutive DNFs
        - Career records: Best/worst finishes significantly unusual

    Example:
        >>> detector = RaceAnomalyDetector()
        >>> results = detector.detect(race_data)
        >>> anomalies = results[results['anomaly_flag']]
    """

    def __init__(self) -> None:
        """Initialize race anomaly detector."""
        self.logger = logger.bind(component="race_anomaly_detector")
        self.history: dict[int, DriverStats] = {}

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901, PLR0912
        """Detect F1-specific anomalies in race data.

        Analyzes driver performance against historical patterns to identify
        unusual results. Updates driver history with each call for evolving
        anomaly detection.

        Args:
            df: Race results DataFrame with columns:
                - driver_id: Unique driver identifier
                - driver_name: Driver name
                - position: Finishing position (0 for DNF)
                - qualifying_position: Qualifying position
                - status: 'Finished', 'DNF', 'Accident', etc.
                - season, round: Race timing information

        Returns:
            DataFrame with anomaly detection columns added:
            - anomaly_flag: Boolean indicating if row is anomalous
            - anomaly_score: Numeric score (0-1) indicating severity
            - anomaly_method: Detection method ('race_anomaly_detector')
        """
        try:
            result = df.copy()

            # Initialize anomaly columns
            result["anomaly_flag"] = False
            result["anomaly_score"] = 0.0
            result["anomaly_method"] = ""

            if result.empty:
                return result

            # Process each race result
            for idx, row in result.iterrows():
                anomaly_score = 0.0
                anomaly_reasons = []

                # Extract driver info
                driver_id = int(row.get("driver_id", 0))
                driver_name = str(row.get("driver_name", "Unknown"))
                position = int(row.get("position", 20))
                qual_position = int(row.get("qualifying_position", 20))
                status = str(row.get("status", "Unknown"))

                if driver_id == 0:
                    continue

                # Get or create driver stats
                if driver_id not in self.history:
                    self.history[driver_id] = DriverStats(
                        driver_id=driver_id,
                        driver_name=driver_name,
                    )

                stats = self.history[driver_id]

                # Check for unusual podium result (position 1-3)
                if position in PODIUM_POSITIONS:
                    podium_score = self._score_unusual_podium(stats, position)
                    if podium_score > PODIUM_ANOMALY_SCORE_THRESHOLD:
                        anomaly_score = max(anomaly_score, podium_score)
                        anomaly_reasons.append("podium_anomaly")

                # Check for qualifying vs race gap (any status except DNF/Did not start)
                if position > 0 and status not in ("DNF", "Did not start"):
                    gap_score = self._score_q_race_gap(qual_position, position)
                    if gap_score > Q_RACE_ANOMALY_SCORE_THRESHOLD:
                        anomaly_score = max(anomaly_score, gap_score)
                        anomaly_reasons.append("q_race_gap")

                # Check for DNF patterns
                if status in ("DNF", "Accident"):
                    dnf_score = self._score_dnf_pattern(stats)
                    if dnf_score > DNF_ANOMALY_SCORE_THRESHOLD:
                        anomaly_score = max(anomaly_score, dnf_score)
                        anomaly_reasons.append("dnf_pattern")

                # Update driver statistics
                self._update_driver_stats(stats, position, status)

                # Set anomaly flag if score exceeds threshold
                if anomaly_score > ANOMALY_SCORE_THRESHOLD:
                    result.at[idx, "anomaly_flag"] = True
                    result.at[idx, "anomaly_score"] = min(anomaly_score, 1.0)
                    result.at[idx, "anomaly_method"] = "race_anomaly_detector"
                elif anomaly_score > 0.0:
                    result.at[idx, "anomaly_score"] = anomaly_score

            self.logger.debug(
                "race_anomaly_detection_complete",
                total_rows=len(result),
                anomalies_detected=result["anomaly_flag"].sum(),
                drivers_analyzed=len(self.history),
            )

            return result

        except Exception as e:
            self.logger.error(
                "error_in_race_anomaly_detection",
                error=str(e),
                exc_info=True,
            )
            # Gracefully return data with no anomalies on error
            result = df.copy()
            result["anomaly_flag"] = False
            result["anomaly_score"] = 0.0
            result["anomaly_method"] = ""
            return result

    def _score_unusual_podium(self, stats: DriverStats, position: int) -> float:
        """Score podium result anomaly.

        Podium results (1-3) are unusual if significantly better than typical
        performance. Uses z-score to detect deviation from mean position.

        Args:
            stats: Driver historical statistics
            position: Current finishing position

        Returns:
            Anomaly score 0-1 (higher = more anomalous)
        """
        # Need at least one prior finish to compare
        if stats.race_count < MIN_HISTORY_FOR_ANOMALY_FLAG:
            return 0.0

        # Podium for driver typically worse-than-podium is anomalous
        if stats.race_count < MIN_HISTORY_FOR_ZSCORE:
            # Limited history: flag if average finish is much worse than podium
            if (
                stats.avg_position >= MIN_POSITION_THRESHOLD
                and position <= PODIUM_POSITIONS_COUNT
            ):
                # Driver typically finishes worse than podium positions
                improvement = min(
                    (stats.avg_position - position) / 10.0, 1.0
                )
                return improvement * PODIUM_IMPROVEMENT_SCORE_MULTIPLIER
            return 0.0

        # With sufficient history, use z-score
        z_score = abs(
            (position - stats.avg_position) / (stats.std_position + 0.1)
        )

        # Podium for worse-than-average driver is more anomalous
        if (
            stats.avg_position > PODIUM_STANDARD_THRESHOLD
            and position <= PODIUM_POSITIONS_COUNT
        ):
            # Big improvement from typical performance
            return min(z_score / PODIUM_ZSCOPE_DIVISOR, 1.0)

        return 0.0

    def _score_q_race_gap(self, qual_position: int, race_position: int) -> float:
        """Score qualifying vs race position gap anomaly.

        Large gaps between qualifying and race position indicate unusual
        race events (crash, incident, mechanical issues, or strong recovery).

        Args:
            qual_position: Qualifying position
            race_position: Race finishing position

        Returns:
            Anomaly score 0-1 (higher = larger gap)
        """
        gap = abs(int(qual_position) - int(race_position))

        # Normalize to 0-1 scale (max realistic gap is ~20)
        score = min(gap / Q_RACE_GAP_MAX_POSITION, 1.0)

        # Large drops (quali → finish worse) are more concerning
        if race_position > qual_position and gap >= Q_RACE_GAP_THRESHOLD_WORSE:
            return score
        # Large improvements are also anomalous
        if race_position < qual_position and gap >= Q_RACE_GAP_THRESHOLD_BETTER:
            return score * Q_RACE_GAP_BETTER_SCORE_MULTIPLIER

        return 0.0

    def _score_dnf_pattern(self, stats: DriverStats) -> float:
        """Score DNF (Did Not Finish) pattern anomaly.

        Multiple consecutive or frequent DNFs indicate unusual issues
        (reliability, driver performance, or external factors).

        Args:
            stats: Driver historical statistics

        Returns:
            Anomaly score 0-1 (higher = more unusual)
        """
        # Single DNF is not unusual
        if stats.dnf_count == 0:
            return 0.0

        if stats.dnf_count == 1:
            return 0.2  # Slight anomaly for first DNF

        # Multiple DNFs become increasingly anomalous
        dnf_rate = stats.dnf_count / max(stats.race_count, 1)

        if dnf_rate > DNF_CRITICAL_THRESHOLD:
            # >50% DNF rate is critical
            return DNF_CRITICAL_SCORE
        if dnf_rate > DNF_SIGNIFICANT_THRESHOLD:
            # >25% DNF rate is significant
            return DNF_SIGNIFICANT_SCORE
        return DNF_STANDARD_SCORE

    def _update_driver_stats(
        self, stats: DriverStats, position: int, status: str
    ) -> None:
        """Update driver statistics with new race result.

        Args:
            stats: Driver statistics to update
            position: Finishing position (0 for DNF)
            status: Finish status (Finished, DNF, etc.)
        """
        if status in ("DNF", "Accident"):
            stats.dnf_count += 1
            # Don't count DNF position in average
        elif position > 0:
            stats.positions.append(position)
            stats.best_position = min(stats.best_position, position)
            stats.worst_position = max(stats.worst_position, position)

        stats.race_count += 1

        # Recalculate statistics (keep only last 10 races for recency)
        if stats.positions:
            recent_positions = stats.positions[-10:]
            stats.avg_position = float(np.mean(recent_positions))
            stats.std_position = (
                float(np.std(recent_positions))
                if len(recent_positions) > 1
                else 1.0
            )
