"""Treatment effect estimation for F1 causal analysis.

This module provides methods to estimate the causal effect of
treatments (interventions) on race outcomes, including:
- Average Treatment Effect (ATE)
- Conditional Average Treatment Effect (CATE)
- Heterogeneous treatment effects
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_predict
import structlog

from f1_predict.causal.dag import F1CausalDAG

logger = structlog.get_logger(__name__)


@dataclass
class ATEResult:
    """Result of Average Treatment Effect estimation."""

    treatment: str
    outcome: str
    ate: float  # Average Treatment Effect
    std_error: float  # Standard error of estimate
    confidence_interval: tuple[float, float]  # 95% CI
    p_value: float  # Statistical significance
    n_treated: int  # Number of treated units
    n_control: int  # Number of control units
    method: str  # Estimation method used
    adjustment_set: list[str]  # Variables adjusted for
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        """Check if effect is statistically significant at 0.05 level."""
        return self.p_value < 0.05

    def summary(self) -> str:
        """Generate human-readable summary of results."""
        sig = "significant" if self.significant else "not significant"
        return (
            f"ATE of {self.treatment} on {self.outcome}: {self.ate:.4f} "
            f"(95% CI: [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}], "
            f"p={self.p_value:.4f}, {sig})"
        )


@dataclass
class CATEResult:
    """Result of Conditional Average Treatment Effect estimation."""

    treatment: str
    outcome: str
    subgroup: dict[str, Any]  # Subgroup definition
    cate: float  # Conditional Average Treatment Effect
    std_error: float
    confidence_interval: tuple[float, float]
    n_subgroup: int  # Number of units in subgroup
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary of results."""
        subgroup_str = ", ".join(f"{k}={v}" for k, v in self.subgroup.items())
        return (
            f"CATE for {subgroup_str}: {self.cate:.4f} "
            f"(95% CI: [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}])"
        )


class TreatmentEffectEstimator:
    """Estimator for treatment effects in F1 race data.

    Provides multiple methods for estimating causal effects:
    - Regression adjustment
    - Inverse probability weighting (IPW)
    - Doubly robust estimation
    - Propensity score matching

    Example:
        >>> estimator = TreatmentEffectEstimator(dag)
        >>> result = estimator.estimate_ate(
        ...     data=race_data,
        ...     treatment="qualifying_position",
        ...     outcome="race_position",
        ...     treatment_value=1,  # Starting from pole
        ...     control_value=10,   # Starting from P10
        ... )
        >>> print(result.summary())
    """

    def __init__(
        self,
        dag: Optional[F1CausalDAG] = None,
        random_state: int = 42,
    ) -> None:
        """Initialize treatment effect estimator.

        Args:
            dag: Causal DAG for identifying adjustment sets
            random_state: Random seed for reproducibility
        """
        self.dag = dag
        self.random_state = random_state
        self.logger = logger.bind(component="TreatmentEffectEstimator")

    def estimate_ate(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        treatment_value: Any = 1,
        control_value: Any = 0,
        method: str = "regression",
        adjustment_set: Optional[list[str]] = None,
        confidence_level: float = 0.95,
    ) -> ATEResult:
        """Estimate Average Treatment Effect.

        Args:
            data: DataFrame with treatment, outcome, and covariates
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            treatment_value: Value indicating treatment (default: 1)
            control_value: Value indicating control (default: 0)
            method: Estimation method ("regression", "ipw", "doubly_robust")
            adjustment_set: Variables to adjust for (uses DAG if not specified)
            confidence_level: Confidence level for interval (default: 0.95)

        Returns:
            ATEResult with effect estimate and statistics
        """
        self.logger.info(
            "estimating_ate",
            treatment=treatment,
            outcome=outcome,
            method=method,
        )

        # Get adjustment set from DAG if not specified
        if adjustment_set is None:
            if self.dag is not None:
                adjustment_set = self.dag.get_adjustment_set(treatment, outcome)
            else:
                adjustment_set = []

        # Filter to valid adjustment variables present in data
        adjustment_set = [v for v in adjustment_set if v in data.columns]

        # Split data into treatment and control groups
        treated_mask = data[treatment] == treatment_value
        control_mask = data[treatment] == control_value

        # If treatment is continuous, create binary indicator
        if not (treated_mask.any() and control_mask.any()):
            # Use median split for continuous treatment
            median_val = data[treatment].median()
            treated_mask = data[treatment] <= median_val
            control_mask = data[treatment] > median_val
            treatment_value = f"<= {median_val}"
            control_value = f"> {median_val}"

        n_treated = treated_mask.sum()
        n_control = control_mask.sum()

        if n_treated == 0 or n_control == 0:
            msg = "Insufficient data for treatment or control group"
            raise ValueError(msg)

        # Estimate ATE using specified method
        if method == "regression":
            ate, std_error = self._regression_adjustment(
                data, treatment, outcome, adjustment_set, treated_mask, control_mask
            )
        elif method == "ipw":
            ate, std_error = self._inverse_probability_weighting(
                data, treatment, outcome, adjustment_set, treated_mask, control_mask
            )
        elif method == "doubly_robust":
            ate, std_error = self._doubly_robust(
                data, treatment, outcome, adjustment_set, treated_mask, control_mask
            )
        else:
            msg = (
                f"Unknown method: {method}. Use 'regression', 'ipw', or 'doubly_robust'"
            )
            raise ValueError(msg)

        # Calculate confidence interval and p-value
        z = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = ate - z * std_error
        ci_upper = ate + z * std_error

        # Two-sided p-value
        z_stat = abs(ate / std_error) if std_error > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(z_stat))

        result = ATEResult(
            treatment=treatment,
            outcome=outcome,
            ate=ate,
            std_error=std_error,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            n_treated=n_treated,
            n_control=n_control,
            method=method,
            adjustment_set=adjustment_set,
            metadata={
                "treatment_value": treatment_value,
                "control_value": control_value,
                "confidence_level": confidence_level,
            },
        )

        self.logger.info(
            "ate_estimated",
            ate=ate,
            std_error=std_error,
            p_value=p_value,
            significant=result.significant,
        )

        return result

    def _regression_adjustment(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: list[str],
        treated_mask: pd.Series,
        control_mask: pd.Series,
    ) -> tuple[float, float]:
        """Estimate ATE using regression adjustment.

        Args:
            data: Full dataset
            treatment: Treatment variable name
            outcome: Outcome variable name
            adjustment_set: Covariates to adjust for
            treated_mask: Boolean mask for treated units
            control_mask: Boolean mask for control units

        Returns:
            Tuple of (ATE estimate, standard error)
        """
        # Create combined mask for valid observations
        valid_mask = treated_mask | control_mask
        df = data[valid_mask].copy()

        # Create binary treatment indicator
        df["_treatment_indicator"] = treated_mask[valid_mask].astype(int)

        # Build feature matrix
        if adjustment_set:
            features = ["_treatment_indicator"] + adjustment_set
        else:
            features = ["_treatment_indicator"]

        X = df[features].values
        y = df[outcome].values

        # Fit regression
        model = LinearRegression()
        model.fit(X, y)

        # ATE is the coefficient on treatment indicator
        ate = model.coef_[0]

        # Bootstrap for standard error
        std_error = self._bootstrap_std_error(
            df, features, outcome, model, n_bootstrap=100
        )

        return ate, std_error

    def _inverse_probability_weighting(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: list[str],
        treated_mask: pd.Series,
        control_mask: pd.Series,
    ) -> tuple[float, float]:
        """Estimate ATE using inverse probability weighting.

        Args:
            data: Full dataset
            treatment: Treatment variable name
            outcome: Outcome variable name
            adjustment_set: Covariates for propensity score
            treated_mask: Boolean mask for treated units
            control_mask: Boolean mask for control units

        Returns:
            Tuple of (ATE estimate, standard error)
        """
        valid_mask = treated_mask | control_mask
        df = data[valid_mask].copy()

        # Create treatment indicator
        T = treated_mask[valid_mask].astype(int).values
        Y = df[outcome].values

        # Estimate propensity score
        if adjustment_set:
            X = df[adjustment_set].values
            propensity_model = LogisticRegression(random_state=self.random_state)
            propensity_model.fit(X, T)
            propensity_scores = propensity_model.predict_proba(X)[:, 1]
        else:
            # Without covariates, use marginal probability
            propensity_scores = np.full(len(T), T.mean())

        # Clip propensity scores to avoid extreme weights
        propensity_scores = np.clip(propensity_scores, 0.01, 0.99)

        # IPW estimator
        weights_treated = T / propensity_scores
        weights_control = (1 - T) / (1 - propensity_scores)

        ate = (weights_treated * Y).sum() / weights_treated.sum() - (
            weights_control * Y
        ).sum() / weights_control.sum()

        # Estimate variance using influence function
        n = len(Y)
        influence = weights_treated * (
            Y - (weights_treated * Y).sum() / weights_treated.sum()
        ) - weights_control * (Y - (weights_control * Y).sum() / weights_control.sum())
        std_error = np.sqrt(np.var(influence) / n)

        return ate, std_error

    def _doubly_robust(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: list[str],
        treated_mask: pd.Series,
        control_mask: pd.Series,
    ) -> tuple[float, float]:
        """Estimate ATE using doubly robust estimation.

        Combines regression adjustment and IPW for robustness.
        Consistent if either the outcome model OR propensity model is correct.

        Args:
            data: Full dataset
            treatment: Treatment variable name
            outcome: Outcome variable name
            adjustment_set: Covariates
            treated_mask: Boolean mask for treated units
            control_mask: Boolean mask for control units

        Returns:
            Tuple of (ATE estimate, standard error)
        """
        valid_mask = treated_mask | control_mask
        df = data[valid_mask].copy()

        T = treated_mask[valid_mask].astype(int).values
        Y = df[outcome].values
        n = len(Y)

        if not adjustment_set:
            # Fall back to simple difference in means
            ate = Y[T == 1].mean() - Y[T == 0].mean()
            std_error = np.sqrt(
                np.var(Y[T == 1]) / (T == 1).sum() + np.var(Y[T == 0]) / (T == 0).sum()
            )
            return ate, std_error

        X = df[adjustment_set].values

        # Estimate propensity scores
        propensity_model = LogisticRegression(random_state=self.random_state)
        propensity_scores = cross_val_predict(
            propensity_model, X, T, cv=5, method="predict_proba"
        )[:, 1]
        propensity_scores = np.clip(propensity_scores, 0.01, 0.99)

        # Estimate outcome models
        outcome_model_1 = GradientBoostingRegressor(
            n_estimators=100, random_state=self.random_state
        )
        outcome_model_0 = GradientBoostingRegressor(
            n_estimators=100, random_state=self.random_state
        )

        # Fit outcome models on respective treatment groups
        outcome_model_1.fit(X[T == 1], Y[T == 1])
        outcome_model_0.fit(X[T == 0], Y[T == 0])

        # Predict potential outcomes
        mu_1 = outcome_model_1.predict(X)
        mu_0 = outcome_model_0.predict(X)

        # Doubly robust estimator
        dr_treated = T * (Y - mu_1) / propensity_scores + mu_1
        dr_control = (1 - T) * (Y - mu_0) / (1 - propensity_scores) + mu_0

        ate = dr_treated.mean() - dr_control.mean()

        # Variance estimation
        influence = dr_treated - dr_control - ate
        std_error = np.sqrt(np.var(influence) / n)

        return ate, std_error

    def _bootstrap_std_error(
        self,
        df: pd.DataFrame,
        features: list[str],
        outcome: str,
        model: Any,
        n_bootstrap: int = 100,
    ) -> float:
        """Estimate standard error using bootstrap.

        Args:
            df: Data to bootstrap
            features: Feature columns
            outcome: Outcome column
            model: Fitted model (for estimating coefficient)
            n_bootstrap: Number of bootstrap samples

        Returns:
            Bootstrap standard error estimate
        """
        np.random.seed(self.random_state)
        n = len(df)
        bootstrap_estimates = []

        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            df_boot = df.iloc[indices]

            X_boot = df_boot[features].values
            y_boot = df_boot[outcome].values

            boot_model = LinearRegression()
            boot_model.fit(X_boot, y_boot)
            bootstrap_estimates.append(boot_model.coef_[0])

        return np.std(bootstrap_estimates)

    def estimate_cate(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        subgroup_vars: list[str],
        treatment_value: Any = 1,
        control_value: Any = 0,
        method: str = "regression",
    ) -> list[CATEResult]:
        """Estimate Conditional Average Treatment Effects for subgroups.

        Args:
            data: DataFrame with treatment, outcome, and covariates
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            subgroup_vars: Variables defining subgroups
            treatment_value: Value indicating treatment
            control_value: Value indicating control
            method: Estimation method

        Returns:
            List of CATEResult for each subgroup
        """
        self.logger.info(
            "estimating_cate",
            treatment=treatment,
            outcome=outcome,
            subgroup_vars=subgroup_vars,
        )

        results = []

        # Get unique combinations of subgroup variables
        subgroups = data[subgroup_vars].drop_duplicates()

        for _, row in subgroups.iterrows():
            subgroup_def = row.to_dict()

            # Filter data to subgroup
            mask = pd.Series(True, index=data.index)
            for var, val in subgroup_def.items():
                mask &= data[var] == val

            subgroup_data = data[mask]

            if len(subgroup_data) < 10:  # Skip small subgroups
                continue

            try:
                ate_result = self.estimate_ate(
                    data=subgroup_data,
                    treatment=treatment,
                    outcome=outcome,
                    treatment_value=treatment_value,
                    control_value=control_value,
                    method=method,
                )

                cate_result = CATEResult(
                    treatment=treatment,
                    outcome=outcome,
                    subgroup=subgroup_def,
                    cate=ate_result.ate,
                    std_error=ate_result.std_error,
                    confidence_interval=ate_result.confidence_interval,
                    n_subgroup=len(subgroup_data),
                    method=method,
                )
                results.append(cate_result)

            except (ValueError, np.linalg.LinAlgError) as e:
                self.logger.warning(
                    "cate_estimation_failed",
                    subgroup=subgroup_def,
                    error=str(e),
                )

        self.logger.info("cate_estimated", n_subgroups=len(results))
        return results

    def estimate_heterogeneous_effects(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        effect_modifiers: list[str],
        treatment_value: Any = 1,
        control_value: Any = 0,
    ) -> dict[str, Any]:
        """Estimate how treatment effects vary across effect modifiers.

        Uses a causal forest-like approach to identify heterogeneity.

        Args:
            data: DataFrame with treatment, outcome, and covariates
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            effect_modifiers: Variables that may modify treatment effect
            treatment_value: Value indicating treatment
            control_value: Value indicating control

        Returns:
            Dictionary with heterogeneous effect estimates
        """
        self.logger.info(
            "estimating_heterogeneous_effects",
            treatment=treatment,
            outcome=outcome,
            effect_modifiers=effect_modifiers,
        )

        # Create treatment indicator
        treated_mask = data[treatment] == treatment_value
        control_mask = data[treatment] == control_value

        if not (treated_mask.any() and control_mask.any()):
            median_val = data[treatment].median()
            treated_mask = data[treatment] <= median_val
            control_mask = data[treatment] > median_val

        valid_mask = treated_mask | control_mask
        df = data[valid_mask].copy()
        T = treated_mask[valid_mask].astype(int).values
        Y = df[outcome].values

        results = {
            "treatment": treatment,
            "outcome": outcome,
            "effect_modifiers": {},
        }

        for modifier in effect_modifiers:
            if modifier not in df.columns:
                continue

            modifier_values = df[modifier].values

            # Fit model with interaction
            X = np.column_stack([T, modifier_values, T * modifier_values])
            model = LinearRegression()
            model.fit(X, Y)

            # Interaction coefficient indicates heterogeneity
            interaction_coef = model.coef_[2]

            # Test significance
            # Simple bootstrap for standard error
            n = len(Y)
            n_boot = 100
            boot_coefs = []
            np.random.seed(self.random_state)

            for _ in range(n_boot):
                idx = np.random.choice(n, size=n, replace=True)
                X_boot = X[idx]
                y_boot = Y[idx]
                boot_model = LinearRegression()
                boot_model.fit(X_boot, y_boot)
                boot_coefs.append(boot_model.coef_[2])

            std_error = np.std(boot_coefs)
            z_stat = abs(interaction_coef / std_error) if std_error > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(z_stat))

            results["effect_modifiers"][modifier] = {
                "interaction_coefficient": interaction_coef,
                "std_error": std_error,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "interpretation": (
                    f"Each unit increase in {modifier} "
                    f"{'increases' if interaction_coef > 0 else 'decreases'} "
                    f"the treatment effect by {abs(interaction_coef):.4f}"
                ),
            }

        self.logger.info(
            "heterogeneous_effects_estimated",
            n_modifiers=len(results["effect_modifiers"]),
        )

        return results
