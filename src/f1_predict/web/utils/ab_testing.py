"""A/B testing framework for model updates.

Provides:
- A/B test creation and management
- Traffic splitting and routing
- Performance comparison between models
- Statistical significance testing
- Gradual rollout (canary deployments)
"""

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
from scipy import stats
import structlog

logger = structlog.get_logger(__name__)


class TestStatus(Enum):
    """A/B test status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class TrafficAllocation(Enum):
    """Traffic allocation strategies."""

    EVEN_SPLIT = "even_split"  # 50/50
    WEIGHTED = "weighted"  # Custom weights
    CANARY = "canary"  # Gradual rollout (10% -> 25% -> 50% -> 100%)


@dataclass
class ABTestConfig:
    """Configuration for A/B test."""

    test_id: str
    control_model: str
    treatment_model: str
    traffic_allocation: str  # "even_split", "weighted", "canary"
    control_weight: float = 0.5
    treatment_weight: float = 0.5
    min_sample_size: int = 100
    significance_level: float = 0.05  # 95% confidence
    primary_metric: str = "accuracy"
    success_criteria: float = 0.02  # 2% improvement required
    duration_hours: int = 24
    auto_promote: bool = False  # Automatically promote if criteria met
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ABTestResult:
    """Result of A/B test."""

    test_id: str
    timestamp: float
    control_model: str
    treatment_model: str
    control_metrics: dict[str, float]
    treatment_metrics: dict[str, float]
    sample_size_control: int
    sample_size_treatment: int
    primary_metric: str
    control_value: float
    treatment_value: float
    difference: float
    percent_improvement: float
    p_value: float
    significant: bool
    confidence_interval: tuple[float, float]
    winner: Optional[str]  # "control", "treatment", or None
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["confidence_interval"] = list(data["confidence_interval"])
        return data


class ABTestingFramework:
    """Manages A/B testing for model comparisons."""

    def __init__(self, data_dir: Path | str = "data/monitoring"):
        """Initialize A/B testing framework.

        Args:
            data_dir: Directory for test storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tests_file = self.data_dir / "ab_tests.jsonl"
        self.results_file = self.data_dir / "ab_test_results.jsonl"
        self.logger = logger.bind(component="ab_testing")

    def create_test(self, config: ABTestConfig) -> bool:
        """Create a new A/B test.

        Args:
            config: Test configuration

        Returns:
            True if successful
        """
        try:
            with open(self.tests_file, "a") as f:
                f.write(json.dumps(config.to_dict()) + "\n")

            self.logger.info(
                "ab_test_created",
                test_id=config.test_id,
                control=config.control_model,
                treatment=config.treatment_model,
            )
            return True
        except Exception as e:
            self.logger.error("error_creating_ab_test", error=str(e))
            return False

    def get_assigned_model(self, test_id: str, user_id: str) -> Optional[str]:
        """Get assigned model for user (control or treatment).

        Uses consistent hashing so same user always gets same model.

        Args:
            test_id: Test ID
            user_id: User/prediction ID

        Returns:
            Model name ("control" or "treatment") or None
        """
        config = self._get_test_config(test_id)
        if not config:
            return None

        # Consistent hashing
        hash_value = hash(f"{test_id}_{user_id}") % 100

        if config["traffic_allocation"] == "even_split":
            return "control" if hash_value < 50 else "treatment"
        if config["traffic_allocation"] == "weighted":
            threshold = config["control_weight"] * 100
            return "control" if hash_value < threshold else "treatment"
        # canary
        # Implement gradual rollout
        phase = self._get_test_phase(test_id)
        thresholds = {0: 10, 1: 25, 2: 50, 3: 100}
        threshold = thresholds.get(phase, 100)
        return "control" if hash_value < (100 - threshold) else "treatment"

        return None

    def record_test_observation(
        self,
        test_id: str,
        model_group: str,  # "control" or "treatment"
        metrics: dict[str, float],
    ) -> None:
        """Record observation from test.

        Args:
            test_id: Test ID
            model_group: Which model group (control/treatment)
            metrics: Observed metrics
        """
        try:
            observation = {
                "timestamp": time.time(),
                "test_id": test_id,
                "model_group": model_group,
                "metrics": metrics,
            }

            observations_file = self.data_dir / f"ab_test_{test_id}_observations.jsonl"
            with open(observations_file, "a") as f:
                f.write(json.dumps(observation) + "\n")
        except Exception as e:
            self.logger.error("error_recording_observation", error=str(e))

    def analyze_test(self, test_id: str, config: ABTestConfig) -> Optional[ABTestResult]:
        """Analyze A/B test results.

        Args:
            test_id: Test ID
            config: Test configuration

        Returns:
            ABTestResult or None
        """
        try:
            observations_file = self.data_dir / f"ab_test_{test_id}_observations.jsonl"

            if not observations_file.exists():
                self.logger.warning("no_observations_for_test", test_id=test_id)
                return None

            control_metrics = []
            treatment_metrics = []

            # Load observations
            with open(observations_file) as f:
                for line in f:
                    if line.strip():
                        obs = json.loads(line)
                        metric_value = obs["metrics"].get(config.primary_metric, 0.0)

                        if obs["model_group"] == "control":
                            control_metrics.append(metric_value)
                        else:
                            treatment_metrics.append(metric_value)

            if not control_metrics or not treatment_metrics:
                self.logger.warning("insufficient_observations", test_id=test_id)
                return None

            # Calculate statistics
            control_mean = np.mean(control_metrics)
            treatment_mean = np.mean(treatment_metrics)
            control_std = np.std(control_metrics)
            treatment_std = np.std(treatment_metrics)

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(
                treatment_metrics, control_metrics, equal_var=False
            )

            significant = p_value < config.significance_level

            # Calculate confidence interval for difference
            se = np.sqrt(
                (control_std**2 / len(control_metrics))
                + (treatment_std**2 / len(treatment_metrics))
            )
            margin_of_error = 1.96 * se  # 95% CI
            difference = treatment_mean - control_mean
            ci = (difference - margin_of_error, difference + margin_of_error)

            # Percent improvement
            percent_improvement = (difference / (control_mean + 1e-10)) * 100

            # Determine winner
            winner = None
            recommendation = "Continue testing"

            if significant:
                if percent_improvement >= config.success_criteria:
                    winner = "treatment"
                    recommendation = "PROMOTE: Treatment model shows statistically significant improvement"
                elif percent_improvement <= -config.success_criteria:
                    winner = "control"
                    recommendation = "REJECT: Control model performs better"
                else:
                    winner = "treatment" if percent_improvement > 0 else "control"
                    recommendation = f"Significant but below success criteria ({percent_improvement:.2f}%)"
            else:
                recommendation = "No statistically significant difference detected"

            result = ABTestResult(
                test_id=test_id,
                timestamp=time.time(),
                control_model=config.control_model,
                treatment_model=config.treatment_model,
                control_metrics={
                    config.primary_metric: float(control_mean),
                    "std": float(control_std),
                },
                treatment_metrics={
                    config.primary_metric: float(treatment_mean),
                    "std": float(treatment_std),
                },
                sample_size_control=len(control_metrics),
                sample_size_treatment=len(treatment_metrics),
                primary_metric=config.primary_metric,
                control_value=float(control_mean),
                treatment_value=float(treatment_mean),
                difference=float(difference),
                percent_improvement=float(percent_improvement),
                p_value=float(p_value),
                significant=significant,
                confidence_interval=tuple(ci),
                winner=winner,
                recommendation=recommendation,
            )

            # Store result
            with open(self.results_file, "a") as f:
                f.write(json.dumps(result.to_dict()) + "\n")

            self.logger.info(
                "ab_test_analyzed",
                test_id=test_id,
                winner=winner,
                percent_improvement=percent_improvement,
            )

            return result
        except Exception as e:
            self.logger.error("error_analyzing_test", test_id=test_id, error=str(e))
            return None

    def get_test_results(self, test_id: Optional[str] = None) -> list[ABTestResult]:
        """Get test results.

        Args:
            test_id: Specific test ID or None for all

        Returns:
            List of test results
        """
        results = []
        try:
            if self.results_file.exists():
                with open(self.results_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if test_id is None or data["test_id"] == test_id:
                                results.append(ABTestResult(**data))
        except Exception as e:
            self.logger.error("error_loading_results", error=str(e))

        return sorted(results, key=lambda x: x.timestamp, reverse=True)

    def _get_test_config(self, test_id: str) -> Optional[dict]:
        """Get test configuration.

        Args:
            test_id: Test ID

        Returns:
            Config dictionary or None
        """
        try:
            if self.tests_file.exists():
                with open(self.tests_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if data["test_id"] == test_id:
                                return data
        except Exception as e:
            self.logger.error("error_loading_config", error=str(e))
        return None

    def _get_test_phase(self, test_id: str) -> int:
        """Get canary deployment phase for test.

        Args:
            test_id: Test ID

        Returns:
            Phase number (0-3)
        """
        config = self._get_test_config(test_id)
        if not config:
            return 0

        # Phase based on elapsed time
        created = config.get("created_at", time.time())
        elapsed_hours = (time.time() - created) / 3600
        duration_hours = config.get("duration_hours", 24)

        phase_duration = duration_hours / 4
        phase = min(int(elapsed_hours / phase_duration), 3)

        return phase
