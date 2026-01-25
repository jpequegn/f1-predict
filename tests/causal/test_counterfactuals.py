"""Tests for counterfactual analysis engine."""

import numpy as np
import pandas as pd
import pytest

from f1_predict.causal.counterfactuals import (
    CounterfactualEngine,
    CounterfactualResult,
    Intervention,
)
from f1_predict.causal.dag import F1CausalDAG


class TestIntervention:
    """Test Intervention dataclass."""

    def test_create_intervention(self):
        """Test creating an intervention."""
        intervention = Intervention(
            variable="qualifying_position",
            original_value=5,
            new_value=1,
            description="What if driver started from pole?",
        )
        assert intervention.variable == "qualifying_position"
        assert intervention.original_value == 5
        assert intervention.new_value == 1

    def test_intervention_str(self):
        """Test string representation."""
        intervention = Intervention(
            variable="qualifying_position",
            original_value=10,
            new_value=3,
        )
        str_repr = str(intervention)
        assert "qualifying_position" in str_repr
        assert "10" in str_repr
        assert "3" in str_repr


class TestCounterfactualResult:
    """Test CounterfactualResult dataclass."""

    def test_create_result(self):
        """Test creating a counterfactual result."""
        intervention = Intervention("position", 5, 1)
        result = CounterfactualResult(
            original_outcome=5.0,
            counterfactual_outcome=2.0,
            effect=-3.0,
            interventions=[intervention],
            confidence_interval=(1.5, 2.5),
            outcome_variable="race_position",
            method="regression",
        )

        assert result.original_outcome == 5.0
        assert result.counterfactual_outcome == 2.0
        assert result.effect == -3.0

    def test_relative_change(self):
        """Test relative change calculation."""
        intervention = Intervention("x", 0, 1)
        result = CounterfactualResult(
            original_outcome=10.0,
            counterfactual_outcome=8.0,
            effect=-2.0,
            interventions=[intervention],
            outcome_variable="outcome",
            method="regression",
        )

        assert result.relative_change == -0.2  # -2/10

    def test_relative_change_zero_original(self):
        """Test relative change when original is zero."""
        intervention = Intervention("x", 0, 1)
        result = CounterfactualResult(
            original_outcome=0.0,
            counterfactual_outcome=5.0,
            effect=5.0,
            interventions=[intervention],
            outcome_variable="outcome",
            method="regression",
        )

        assert result.relative_change == float("inf")

    def test_summary(self):
        """Test summary generation."""
        intervention = Intervention("pit_stops", 2, 1)
        result = CounterfactualResult(
            original_outcome=5.0,
            counterfactual_outcome=3.0,
            effect=-2.0,
            interventions=[intervention],
            outcome_variable="race_position",
            method="regression",
        )

        summary = result.summary()
        assert "Counterfactual Analysis" in summary
        assert "race_position" in summary
        assert "5.00" in summary
        assert "3.00" in summary


class TestCounterfactualEngine:
    """Test CounterfactualEngine class."""

    @pytest.fixture
    def training_data(self):
        """Create training dataset."""
        np.random.seed(42)
        n = 100

        qualifying_position = np.random.randint(1, 21, n)
        driver_skill = np.random.normal(50, 10, n)
        team_performance = np.random.normal(100, 15, n)

        # Outcome: race position influenced by qualifying and other factors
        noise = np.random.normal(0, 2, n)
        race_position = (
            0.6 * qualifying_position
            - 0.1 * driver_skill
            - 0.05 * team_performance
            + 10
            + noise
        )
        race_position = np.clip(race_position, 1, 20)

        return pd.DataFrame(
            {
                "qualifying_position": qualifying_position,
                "driver_skill": driver_skill,
                "team_performance": team_performance,
                "race_position": race_position,
            }
        )

    @pytest.fixture
    def engine(self):
        """Create counterfactual engine."""
        return CounterfactualEngine()

    @pytest.fixture
    def engine_with_dag(self):
        """Create engine with F1 DAG."""
        dag = F1CausalDAG()
        dag.build_default_dag()
        return CounterfactualEngine(dag=dag)

    def test_what_if_regression(self, engine, training_data):
        """Test what-if analysis with regression method."""
        observation = training_data.iloc[0].copy()

        result = engine.what_if(
            observation=observation,
            interventions=[
                Intervention(
                    "qualifying_position", observation["qualifying_position"], 1
                )
            ],
            outcome="race_position",
            training_data=training_data,
            method="regression",
            n_bootstrap=20,
        )

        assert isinstance(result, CounterfactualResult)
        assert result.method == "regression"

    def test_what_if_structural(self, engine, training_data):
        """Test what-if analysis with structural method (no DAG - falls back to regression)."""
        observation = training_data.iloc[0].copy()

        # Without DAG, structural method falls back to regression
        result = engine.what_if(
            observation=observation,
            interventions=[
                Intervention(
                    "qualifying_position", observation["qualifying_position"], 1
                )
            ],
            outcome="race_position",
            training_data=training_data,
            method="structural",
            n_bootstrap=20,
        )

        assert isinstance(result, CounterfactualResult)
        assert result.method == "structural"

    def test_what_if_matching(self, engine, training_data):
        """Test what-if analysis with matching method."""
        observation = training_data.iloc[0].copy()

        result = engine.what_if(
            observation=observation,
            interventions=[
                Intervention(
                    "qualifying_position", observation["qualifying_position"], 3
                )
            ],
            outcome="race_position",
            training_data=training_data,
            method="matching",
        )

        assert isinstance(result, CounterfactualResult)
        assert result.method == "matching"

    def test_what_if_invalid_method(self, engine, training_data):
        """Test error on invalid method."""
        observation = training_data.iloc[0].copy()

        with pytest.raises(ValueError, match="Unknown method"):
            engine.what_if(
                observation=observation,
                interventions=[Intervention("qualifying_position", 5, 1)],
                outcome="race_position",
                training_data=training_data,
                method="invalid_method",
            )

    def test_multiple_interventions(self, engine, training_data):
        """Test what-if with multiple interventions."""
        observation = training_data.iloc[0].copy()

        result = engine.what_if(
            observation=observation,
            interventions=[
                Intervention(
                    "qualifying_position", observation["qualifying_position"], 1
                ),
                Intervention("driver_skill", observation["driver_skill"], 70),
            ],
            outcome="race_position",
            training_data=training_data,
            method="regression",
            n_bootstrap=20,
        )

        assert len(result.interventions) == 2

    def test_confidence_intervals(self, engine, training_data):
        """Test that confidence intervals are computed."""
        observation = training_data.iloc[0].copy()

        result = engine.what_if(
            observation=observation,
            interventions=[
                Intervention(
                    "qualifying_position", observation["qualifying_position"], 1
                )
            ],
            outcome="race_position",
            training_data=training_data,
            method="regression",
            n_bootstrap=50,
        )

        assert result.confidence_interval is not None
        lower, upper = result.confidence_interval
        assert lower <= upper


class TestScenarioAnalysis:
    """Test scenario analysis functionality."""

    @pytest.fixture
    def base_data(self):
        """Create base dataset for scenarios."""
        np.random.seed(42)
        n = 50

        return pd.DataFrame(
            {
                "qualifying_position": np.random.randint(1, 21, n),
                "pit_stops": np.random.randint(1, 4, n),
                "driver_skill": np.random.normal(50, 10, n),
                "race_position": np.random.randint(1, 21, n),
            }
        )

    @pytest.fixture
    def engine(self):
        """Create counterfactual engine."""
        return CounterfactualEngine()

    def test_scenario_analysis(self, engine, base_data):
        """Test running scenario analysis."""
        scenarios = [
            {"name": "one_stop", "pit_stops": 1},
            {"name": "two_stop", "pit_stops": 2},
            {"name": "three_stop", "pit_stops": 3},
        ]

        results = engine.scenario_analysis(
            base_data=base_data,
            scenarios=scenarios,
            outcome="race_position",
            aggregation="mean",
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 4  # baseline + 3 scenarios
        assert "scenario" in results.columns
        assert "baseline" in results["scenario"].values

    def test_scenario_analysis_different_aggregation(self, engine, base_data):
        """Test scenario analysis with different aggregation."""
        scenarios = [{"name": "test", "pit_stops": 2}]

        for agg in ["mean", "median", "sum"]:
            results = engine.scenario_analysis(
                base_data=base_data,
                scenarios=scenarios,
                outcome="race_position",
                aggregation=agg,
            )
            assert f"{agg}_race_position" in results.columns


class TestSensitivityAnalysis:
    """Test sensitivity analysis functionality."""

    @pytest.fixture
    def training_data(self):
        """Create training dataset."""
        np.random.seed(42)
        n = 100

        qualifying = np.random.randint(1, 21, n)
        skill = np.random.normal(50, 10, n)
        position = 0.6 * qualifying - 0.1 * skill + 10 + np.random.normal(0, 2, n)

        return pd.DataFrame(
            {
                "qualifying_position": qualifying,
                "driver_skill": skill,
                "race_position": np.clip(position, 1, 20),
            }
        )

    @pytest.fixture
    def engine(self):
        """Create counterfactual engine."""
        return CounterfactualEngine()

    def test_sensitivity_analysis(self, engine, training_data):
        """Test sensitivity analysis."""
        observation = training_data.iloc[0].copy()

        results = engine.sensitivity_analysis(
            observation=observation,
            variable="qualifying_position",
            outcome="race_position",
            training_data=training_data,
            value_range=(1, 10),
            n_points=5,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 5
        assert "qualifying_position" in results.columns
        assert "predicted_race_position" in results.columns
        assert "effect" in results.columns

    def test_sensitivity_analysis_auto_range(self, engine, training_data):
        """Test sensitivity analysis with auto-detected range."""
        observation = training_data.iloc[0].copy()

        results = engine.sensitivity_analysis(
            observation=observation,
            variable="qualifying_position",
            outcome="race_position",
            training_data=training_data,
            n_points=10,
        )

        assert len(results) == 10


class TestCounterfactualEngineEdgeCases:
    """Test edge cases for counterfactual engine."""

    @pytest.fixture
    def small_data(self):
        """Create small dataset."""
        return pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 4, 6, 8, 10],
                "outcome": [1, 2, 3, 4, 5],
            }
        )

    @pytest.fixture
    def engine(self):
        """Create engine."""
        return CounterfactualEngine()

    def test_structural_without_dag(self, engine, small_data):
        """Test structural method falls back to regression without DAG."""
        observation = small_data.iloc[0].copy()

        result = engine.what_if(
            observation=observation,
            interventions=[Intervention("x", 1, 3)],
            outcome="outcome",
            training_data=small_data,
            method="structural",
        )

        # Should not raise, falls back to regression
        assert result is not None

    def test_matching_small_n(self, engine, small_data):
        """Test matching with few neighbors available."""
        observation = small_data.iloc[0].copy()

        result = engine.what_if(
            observation=observation,
            interventions=[Intervention("x", 1, 3)],
            outcome="outcome",
            training_data=small_data,
            method="matching",
        )

        assert result is not None
