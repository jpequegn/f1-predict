"""Counterfactual analysis for F1 race predictions.

This module provides tools for "what-if" analysis, enabling
exploration of alternative scenarios and their potential outcomes.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import structlog

from f1_predict.causal.dag import F1CausalDAG

logger = structlog.get_logger(__name__)


@dataclass
class Intervention:
    """Represents an intervention (change) to a variable."""

    variable: str
    original_value: Any
    new_value: Any
    description: str = ""

    def __str__(self) -> str:
        """Return string representation of intervention."""
        return f"{self.variable}: {self.original_value} -> {self.new_value}"


@dataclass
class CounterfactualResult:
    """Result of counterfactual analysis."""

    original_outcome: float
    counterfactual_outcome: float
    effect: float  # counterfactual - original
    interventions: list[Intervention]
    confidence_interval: Optional[tuple[float, float]] = None
    outcome_variable: str = ""
    method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def relative_change(self) -> float:
        """Calculate relative change in outcome."""
        if self.original_outcome == 0:
            return float("inf") if self.effect != 0 else 0.0
        return self.effect / abs(self.original_outcome)

    def summary(self) -> str:
        """Generate human-readable summary."""
        interventions_str = "; ".join(str(i) for i in self.interventions)
        cf_val = self.counterfactual_outcome
        return (
            f"Counterfactual Analysis:\n"
            f"  Interventions: {interventions_str}\n"
            f"  Original {self.outcome_variable}: {self.original_outcome:.2f}\n"
            f"  Counterfactual {self.outcome_variable}: {cf_val:.2f}\n"
            f"  Effect: {self.effect:+.2f} ({self.relative_change:+.1%})"
        )


class CounterfactualEngine:
    """Engine for generating and analyzing counterfactuals.

    Supports multiple approaches:
    - Structural equation modeling
    - Regression-based prediction
    - Nearest neighbor matching

    Example:
        >>> engine = CounterfactualEngine(dag)
        >>> result = engine.what_if(
        ...     observation=race_data.iloc[0],
        ...     interventions=[
        ...         Intervention("qualifying_position", 5, 1)
        ...     ],
        ...     outcome="race_position",
        ...     training_data=race_data,
        ... )
        >>> print(result.summary())
    """

    def __init__(
        self,
        dag: Optional[F1CausalDAG] = None,
        random_state: int = 42,
    ) -> None:
        """Initialize counterfactual engine.

        Args:
            dag: Causal DAG for structural modeling
            random_state: Random seed for reproducibility
        """
        self.dag = dag
        self.random_state = random_state
        self.logger = logger.bind(component="CounterfactualEngine")
        self._outcome_models: dict[str, Any] = {}

    def what_if(
        self,
        observation: pd.Series,
        interventions: list[Intervention],
        outcome: str,
        training_data: pd.DataFrame,
        method: str = "regression",
        n_bootstrap: int = 100,
    ) -> CounterfactualResult:
        """Generate counterfactual prediction for a single observation.

        Args:
            observation: Single observation to analyze
            interventions: List of interventions to apply
            outcome: Outcome variable to predict
            training_data: Historical data for model fitting
            method: Method for counterfactual generation
                   ("regression", "structural", "matching")
            n_bootstrap: Number of bootstrap samples for CI

        Returns:
            CounterfactualResult with predictions and effect
        """
        self.logger.info(
            "generating_counterfactual",
            outcome=outcome,
            n_interventions=len(interventions),
            method=method,
        )

        # Get original outcome
        original_outcome = float(observation[outcome])

        # Create counterfactual observation
        cf_observation = observation.copy()
        for intervention in interventions:
            intervention.original_value = observation[intervention.variable]
            cf_observation[intervention.variable] = intervention.new_value

        # Generate counterfactual outcome
        if method == "regression":
            cf_outcome, ci = self._regression_counterfactual(
                cf_observation, outcome, training_data, n_bootstrap
            )
        elif method == "structural":
            cf_outcome, ci = self._structural_counterfactual(
                observation, cf_observation, outcome, training_data, n_bootstrap
            )
        elif method == "matching":
            cf_outcome, ci = self._matching_counterfactual(
                cf_observation, outcome, training_data
            )
        else:
            msg = (
                f"Unknown method: {method}. "
                "Use 'regression', 'structural', or 'matching'"
            )
            raise ValueError(msg)

        effect = cf_outcome - original_outcome

        result = CounterfactualResult(
            original_outcome=original_outcome,
            counterfactual_outcome=cf_outcome,
            effect=effect,
            interventions=interventions,
            confidence_interval=ci,
            outcome_variable=outcome,
            method=method,
        )

        self.logger.info(
            "counterfactual_generated",
            original=original_outcome,
            counterfactual=cf_outcome,
            effect=effect,
        )

        return result

    def _regression_counterfactual(
        self,
        cf_observation: pd.Series,
        outcome: str,
        training_data: pd.DataFrame,
        n_bootstrap: int,
    ) -> tuple[float, Optional[tuple[float, float]]]:
        """Generate counterfactual using regression model.

        Args:
            cf_observation: Counterfactual observation
            outcome: Outcome variable
            training_data: Training data for model
            n_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (predicted outcome, confidence interval)
        """
        # Identify features (all columns except outcome)
        features = [c for c in training_data.columns if c != outcome]
        features = [f for f in features if f in cf_observation.index]

        x_train = training_data[features].values  # noqa: N806
        y_train = training_data[outcome].values

        # Fit model
        model = GradientBoostingRegressor(
            n_estimators=100,
            random_state=self.random_state,
        )
        model.fit(x_train, y_train)

        # Predict counterfactual
        x_cf = cf_observation[features].values.reshape(1, -1)  # noqa: N806
        cf_outcome = float(model.predict(x_cf)[0])

        # Bootstrap confidence interval
        rng = np.random.default_rng(self.random_state)
        bootstrap_predictions = []

        for _ in range(n_bootstrap):
            idx = rng.choice(len(y_train), size=len(y_train), replace=True)
            x_boot = x_train[idx]  # noqa: N806
            y_boot = y_train[idx]

            boot_model = GradientBoostingRegressor(
                n_estimators=50,
                random_state=self.random_state,
            )
            boot_model.fit(x_boot, y_boot)
            bootstrap_predictions.append(boot_model.predict(x_cf)[0])

        ci = (
            float(np.percentile(bootstrap_predictions, 2.5)),
            float(np.percentile(bootstrap_predictions, 97.5)),
        )

        return cf_outcome, ci

    def _structural_counterfactual(
        self,
        original: pd.Series,
        cf_observation: pd.Series,
        outcome: str,
        training_data: pd.DataFrame,
        n_bootstrap: int,
    ) -> tuple[float, Optional[tuple[float, float]]]:
        """Generate counterfactual using structural equations.

        Uses the DAG to propagate interventions through the causal structure.

        Args:
            original: Original observation
            cf_observation: Observation with interventions
            outcome: Outcome variable
            training_data: Training data
            n_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (predicted outcome, confidence interval)
        """
        if self.dag is None:
            # Fall back to regression if no DAG
            return self._regression_counterfactual(
                cf_observation, outcome, training_data, n_bootstrap
            )

        # Get topological order of nodes
        order = self._topological_sort()

        # Build structural equations for each node
        equations = self._fit_structural_equations(training_data, order)

        # Propagate intervention effects through the graph
        cf_values = cf_observation.copy()

        # Identify which variables were directly intervened on
        intervened_vars = {
            col for col in cf_observation.index if cf_observation[col] != original[col]
        }

        for node in order:
            if node in intervened_vars:
                # Keep intervened value
                continue

            if node not in equations:
                continue

            parents = self.dag.get_parents(node)
            if not parents:
                continue

            # Get parent values
            parent_vals = []
            for parent in parents:
                if parent in cf_values.index:
                    parent_vals.append(cf_values[parent])
                else:
                    parent_vals.append(0)

            if parent_vals:
                x_pred = np.array(parent_vals).reshape(1, -1)  # noqa: N806
                cf_values[node] = equations[node].predict(x_pred)[0]

        cf_outcome = float(cf_values[outcome]) if outcome in cf_values.index else 0.0

        # Simple bootstrap CI
        rng = np.random.default_rng(self.random_state)
        bootstrap_outcomes = []

        for _ in range(n_bootstrap):
            # Add noise to simulate uncertainty
            noise = rng.normal(0, 0.5)
            bootstrap_outcomes.append(cf_outcome + noise)

        ci = (
            float(np.percentile(bootstrap_outcomes, 2.5)),
            float(np.percentile(bootstrap_outcomes, 97.5)),
        )

        return cf_outcome, ci

    def _matching_counterfactual(
        self,
        cf_observation: pd.Series,
        outcome: str,
        training_data: pd.DataFrame,
        n_matches: int = 5,
    ) -> tuple[float, Optional[tuple[float, float]]]:
        """Generate counterfactual using nearest neighbor matching.

        Finds similar observations in training data and averages their outcomes.

        Args:
            cf_observation: Counterfactual observation
            outcome: Outcome variable
            training_data: Training data to match against
            n_matches: Number of nearest neighbors

        Returns:
            Tuple of (predicted outcome, confidence interval)
        """
        features = [c for c in training_data.columns if c != outcome]
        features = [f for f in features if f in cf_observation.index]

        x_train = training_data[features].values  # noqa: N806
        y_train = training_data[outcome].values
        x_cf = cf_observation[features].values.reshape(1, -1)  # noqa: N806

        # Standardize for distance calculation
        x_mean = x_train.mean(axis=0)  # noqa: N806
        x_std = x_train.std(axis=0) + 1e-8  # noqa: N806

        x_normalized = (x_train - x_mean) / x_std  # noqa: N806
        x_cf_normalized = (x_cf - x_mean) / x_std  # noqa: N806

        # Calculate distances
        distances = np.sqrt(((x_normalized - x_cf_normalized) ** 2).sum(axis=1))

        # Get nearest neighbors
        nearest_idx = np.argsort(distances)[:n_matches]
        matched_outcomes = y_train[nearest_idx]

        cf_outcome = float(matched_outcomes.mean())
        ci = (
            float(matched_outcomes.min()),
            float(matched_outcomes.max()),
        )

        return cf_outcome, ci

    def _topological_sort(self) -> list[str]:
        """Get topological ordering of DAG nodes.

        Returns:
            List of node names in topological order
        """
        if self.dag is None:
            return []

        in_degree: dict[str, int] = {node: 0 for node in self.dag.nodes}
        for edge in self.dag.edges.values():
            in_degree[edge.target] += 1

        queue = [node for node, degree in in_degree.items() if degree == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self.dag.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return order

    def _fit_structural_equations(
        self,
        data: pd.DataFrame,
        order: list[str],
    ) -> dict[str, Any]:
        """Fit structural equations for each node in the DAG.

        Args:
            data: Training data
            order: Topological order of nodes

        Returns:
            Dictionary mapping node names to fitted models
        """
        if self.dag is None:
            return {}

        equations: dict[str, Any] = {}

        for node in order:
            if node not in data.columns:
                continue

            parents = self.dag.get_parents(node)
            parents = [p for p in parents if p in data.columns]

            if not parents:
                continue

            x_train = data[parents].values  # noqa: N806
            y_train = data[node].values

            model = LinearRegression()
            model.fit(x_train, y_train)
            equations[node] = model

        return equations

    def scenario_analysis(
        self,
        base_data: pd.DataFrame,
        scenarios: list[dict[str, Any]],
        outcome: str,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """Analyze multiple scenarios across a dataset.

        Args:
            base_data: Base dataset to apply scenarios to
            scenarios: List of scenario definitions
                       Each scenario is a dict with variable: value pairs
            outcome: Outcome variable
            aggregation: How to aggregate results ("mean", "median", "sum")

        Returns:
            DataFrame with scenario comparison results
        """
        self.logger.info(
            "running_scenario_analysis",
            n_scenarios=len(scenarios),
            n_observations=len(base_data),
        )

        results = []

        # Baseline
        baseline_outcome = base_data[outcome].agg(aggregation)
        results.append(
            {
                "scenario": "baseline",
                f"{aggregation}_{outcome}": baseline_outcome,
                "change": 0.0,
                "relative_change": 0.0,
            }
        )

        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get("name", f"scenario_{i + 1}")
            interventions = {k: v for k, v in scenario.items() if k != "name"}

            # Apply scenario to all observations
            scenario_data = base_data.copy()
            for var, value in interventions.items():
                if var in scenario_data.columns:
                    scenario_data[var] = value

            # Predict outcomes
            features = [c for c in base_data.columns if c != outcome]
            features = [f for f in features if f in scenario_data.columns]

            # Fit model on original data
            x_train = base_data[features].values  # noqa: N806
            y_train = base_data[outcome].values

            model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
            )
            model.fit(x_train, y_train)

            # Predict scenario outcomes
            x_scenario = scenario_data[features].values  # noqa: N806
            scenario_outcomes = model.predict(x_scenario)

            scenario_agg = float(pd.Series(scenario_outcomes).agg(aggregation))
            change = scenario_agg - baseline_outcome
            rel_change = change / abs(baseline_outcome) if baseline_outcome != 0 else 0

            results.append(
                {
                    "scenario": scenario_name,
                    f"{aggregation}_{outcome}": scenario_agg,
                    "change": change,
                    "relative_change": rel_change,
                    **interventions,
                }
            )

        results_df = pd.DataFrame(results)

        self.logger.info(
            "scenario_analysis_complete",
            n_scenarios=len(scenarios),
        )

        return results_df

    def sensitivity_analysis(
        self,
        observation: pd.Series,
        variable: str,
        outcome: str,
        training_data: pd.DataFrame,
        value_range: Optional[tuple[float, float]] = None,
        n_points: int = 20,
    ) -> pd.DataFrame:
        """Analyze sensitivity of outcome to changes in a variable.

        Args:
            observation: Base observation
            variable: Variable to vary
            outcome: Outcome variable
            training_data: Training data
            value_range: Range of values to test (default: observed range)
            n_points: Number of points to evaluate

        Returns:
            DataFrame with sensitivity curve
        """
        self.logger.info(
            "running_sensitivity_analysis",
            variable=variable,
            outcome=outcome,
        )

        if value_range is None:
            value_range = (
                float(training_data[variable].min()),
                float(training_data[variable].max()),
            )

        test_values = np.linspace(value_range[0], value_range[1], n_points)
        results = []

        original_value = observation[variable]

        for val in test_values:
            cf_result = self.what_if(
                observation=observation,
                interventions=[Intervention(variable, original_value, val)],
                outcome=outcome,
                training_data=training_data,
                method="regression",
                n_bootstrap=50,
            )

            results.append(
                {
                    variable: val,
                    f"predicted_{outcome}": cf_result.counterfactual_outcome,
                    "effect": cf_result.effect,
                    "relative_change": cf_result.relative_change,
                }
            )

        sensitivity_df = pd.DataFrame(results)

        self.logger.info(
            "sensitivity_analysis_complete",
            n_points=n_points,
        )

        return sensitivity_df
