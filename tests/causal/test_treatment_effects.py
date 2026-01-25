"""Tests for treatment effect estimation."""

import numpy as np
import pandas as pd
import pytest

from f1_predict.causal.dag import F1CausalDAG
from f1_predict.causal.treatment_effects import (
    ATEResult,
    CATEResult,
    TreatmentEffectEstimator,
)


class TestATEResult:
    """Test ATEResult dataclass."""

    def test_create_ate_result(self):
        """Test creating ATE result."""
        result = ATEResult(
            treatment="pit_strategy",
            outcome="race_position",
            ate=2.5,
            std_error=0.3,
            confidence_interval=(1.9, 3.1),
            p_value=0.001,
            n_treated=50,
            n_control=100,
            method="regression",
            adjustment_set=["driver_skill"],
        )
        assert result.ate == 2.5
        assert result.std_error == 0.3
        assert result.n_treated == 50
        assert result.n_control == 100

    def test_significant_property(self):
        """Test significance detection."""
        significant = ATEResult(
            treatment="treatment",
            outcome="outcome",
            ate=2.5,
            std_error=0.3,
            confidence_interval=(1.9, 3.1),
            p_value=0.01,
            n_treated=50,
            n_control=100,
            method="regression",
            adjustment_set=[],
        )
        assert significant.significant is True

        not_significant = ATEResult(
            treatment="treatment",
            outcome="outcome",
            ate=0.5,
            std_error=0.8,
            confidence_interval=(-1.0, 2.0),
            p_value=0.15,
            n_treated=50,
            n_control=100,
            method="regression",
            adjustment_set=[],
        )
        assert not_significant.significant is False

    def test_summary(self):
        """Test summary generation."""
        result = ATEResult(
            treatment="treatment",
            outcome="outcome",
            ate=2.5,
            std_error=0.3,
            confidence_interval=(1.9, 3.1),
            p_value=0.01,
            n_treated=50,
            n_control=100,
            method="regression",
            adjustment_set=[],
        )
        summary = result.summary()
        assert "treatment" in summary
        assert "outcome" in summary
        assert "significant" in summary


class TestCATEResult:
    """Test CATEResult dataclass."""

    def test_create_cate_result(self):
        """Test creating CATE result."""
        result = CATEResult(
            treatment="pit_strategy",
            outcome="race_position",
            subgroup={"team": "Mercedes"},
            cate=1.5,
            std_error=0.4,
            confidence_interval=(0.7, 2.3),
            n_subgroup=25,
            method="regression",
        )
        assert result.cate == 1.5
        assert result.subgroup["team"] == "Mercedes"
        assert result.n_subgroup == 25

    def test_summary(self):
        """Test summary generation."""
        result = CATEResult(
            treatment="pit_strategy",
            outcome="race_position",
            subgroup={"team": "Mercedes"},
            cate=1.5,
            std_error=0.4,
            confidence_interval=(0.7, 2.3),
            n_subgroup=25,
            method="regression",
        )
        summary = result.summary()
        assert "Mercedes" in summary
        assert "1.5" in summary


class TestTreatmentEffectEstimator:
    """Test TreatmentEffectEstimator class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for treatment effect estimation."""
        np.random.seed(42)
        n = 200

        # Confounders
        driver_skill = np.random.normal(50, 10, n)
        team_budget = np.random.normal(100, 20, n)

        # Treatment (binary: 1 = aggressive pit strategy)
        # Treatment probability depends on confounders
        propensity = 1 / (1 + np.exp(-(0.02 * driver_skill + 0.01 * team_budget - 2)))
        treatment = (np.random.random(n) < propensity).astype(int)

        # Outcome (race position) - lower is better
        # True treatment effect = -2 (aggressive strategy improves position by 2)
        true_effect = -2
        noise = np.random.normal(0, 3, n)
        outcome = (
            20
            - 0.2 * driver_skill
            - 0.05 * team_budget
            + true_effect * treatment
            + noise
        )

        return pd.DataFrame(
            {
                "treatment": treatment,
                "outcome": outcome,
                "driver_skill": driver_skill,
                "team_budget": team_budget,
            }
        )

    @pytest.fixture
    def estimator(self):
        """Create estimator with default F1 DAG."""
        dag = F1CausalDAG()
        dag.build_default_dag()
        return TreatmentEffectEstimator(dag=dag)

    @pytest.fixture
    def simple_estimator(self):
        """Create estimator without DAG."""
        return TreatmentEffectEstimator()

    def test_estimate_ate_regression(self, simple_estimator, sample_data):
        """Test ATE estimation with regression adjustment."""
        result = simple_estimator.estimate_ate(
            data=sample_data,
            treatment="treatment",
            outcome="outcome",
            adjustment_set=["driver_skill", "team_budget"],
            method="regression",
        )

        assert isinstance(result, ATEResult)
        assert result.method == "regression"
        # True effect is -2, estimate should be close
        assert -4 < result.ate < 0
        assert (
            result.confidence_interval[0] < result.ate < result.confidence_interval[1]
        )

    def test_estimate_ate_ipw(self, simple_estimator, sample_data):
        """Test ATE estimation with inverse probability weighting."""
        result = simple_estimator.estimate_ate(
            data=sample_data,
            treatment="treatment",
            outcome="outcome",
            adjustment_set=["driver_skill", "team_budget"],
            method="ipw",
        )

        assert isinstance(result, ATEResult)
        assert result.method == "ipw"
        # IPW estimate should also be close to -2
        assert -5 < result.ate < 1

    def test_estimate_ate_doubly_robust(self, simple_estimator, sample_data):
        """Test ATE estimation with doubly robust method."""
        result = simple_estimator.estimate_ate(
            data=sample_data,
            treatment="treatment",
            outcome="outcome",
            adjustment_set=["driver_skill", "team_budget"],
            method="doubly_robust",
        )

        assert isinstance(result, ATEResult)
        assert result.method == "doubly_robust"
        # Doubly robust should be most accurate
        assert -4 < result.ate < 0

    def test_estimate_ate_invalid_method(self, simple_estimator, sample_data):
        """Test error on invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            simple_estimator.estimate_ate(
                data=sample_data,
                treatment="treatment",
                outcome="outcome",
                adjustment_set=["driver_skill"],
                method="invalid_method",
            )

    def test_estimate_cate(self, simple_estimator, sample_data):
        """Test CATE estimation for subgroups."""
        # Add a subgroup variable
        sample_data["high_skill"] = sample_data["driver_skill"] > 50

        result = simple_estimator.estimate_cate(
            data=sample_data,
            treatment="treatment",
            outcome="outcome",
            subgroup_vars=["high_skill"],
        )

        assert isinstance(result, list)
        assert len(result) > 0
        for subgroup_result in result:
            assert isinstance(subgroup_result, CATEResult)

    def test_estimate_heterogeneous_effects(self, simple_estimator, sample_data):
        """Test heterogeneous treatment effect estimation."""
        result = simple_estimator.estimate_heterogeneous_effects(
            data=sample_data,
            treatment="treatment",
            outcome="outcome",
            effect_modifiers=["driver_skill"],
        )

        assert isinstance(result, dict)
        assert "effect_modifiers" in result
        assert "driver_skill" in result["effect_modifiers"]

    def test_ate_result_counts(self, simple_estimator, sample_data):
        """Test that ATE result contains correct counts."""
        result = simple_estimator.estimate_ate(
            data=sample_data,
            treatment="treatment",
            outcome="outcome",
            adjustment_set=["driver_skill"],
            method="regression",
        )

        total = result.n_treated + result.n_control
        assert total == len(sample_data)

    def test_confidence_intervals(self, simple_estimator, sample_data):
        """Test that confidence intervals are computed."""
        result = simple_estimator.estimate_ate(
            data=sample_data,
            treatment="treatment",
            outcome="outcome",
            adjustment_set=["driver_skill"],
            method="regression",
        )

        assert result.confidence_interval is not None
        lower, upper = result.confidence_interval
        assert lower < upper


class TestTreatmentEffectEstimatorEdgeCases:
    """Test edge cases for treatment effect estimation."""

    @pytest.fixture
    def estimator(self):
        """Create simple estimator."""
        return TreatmentEffectEstimator()

    def test_no_adjustment_set(self, estimator):
        """Test estimation without adjustment set."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "treatment": np.random.binomial(1, 0.5, 100),
                "outcome": np.random.normal(10, 2, 100),
            }
        )

        result = estimator.estimate_ate(
            data=data,
            treatment="treatment",
            outcome="outcome",
            adjustment_set=[],
            method="regression",
        )

        assert result is not None
        assert result.ate is not None

    def test_continuous_treatment(self, estimator):
        """Test with continuous treatment variable."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "treatment": np.random.normal(10, 3, 100),
                "outcome": np.random.normal(5, 2, 100),
                "confounder": np.random.normal(0, 1, 100),
            }
        )

        # Should handle continuous treatment by median split
        result = estimator.estimate_ate(
            data=data,
            treatment="treatment",
            outcome="outcome",
            adjustment_set=["confounder"],
            method="regression",
        )

        assert result is not None
