"""Reporting and visualization for sensitivity analysis."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import json

from f1_predict.simulation.analysis.sensitivity_analyzer import SensitivityResult


@dataclass
class TornadoChartData:
    """Data for tornado chart visualization."""

    parameter_name: str
    drivers: Dict[str, Tuple[float, float]]  # driver_id -> (negative_impact, positive_impact)
    base_probabilities: Dict[str, float]  # driver_id -> base_prob


class SensitivityReport:
    """Generate reports and visualization data from sensitivity analysis."""

    def __init__(self, sensitivity_result: SensitivityResult):
        """Initialize report generator.

        Args:
            sensitivity_result: SensitivityResult from analysis
        """
        self.result = sensitivity_result

    def generate_summary_text(self) -> str:
        """Generate human-readable summary of sensitivity analysis.

        Returns:
            Formatted summary text
        """
        lines = []
        lines.append("=" * 70)
        lines.append("SENSITIVITY ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append("")

        lines.append(f"Parameter: {self.result.parameter_name}")
        lines.append(f"Type: {self.result.parameter_type.value}")
        lines.append(f"Parameter Values: {self.result.parameter_values}")
        lines.append("")

        lines.append("SUMMARY BY DRIVER:")
        lines.append("-" * 70)

        for driver_id in self.result.drivers:
            base_prob = self.result.base_result.get_winner_probability(driver_id)
            ci_low, ci_high = self.result.get_confidence_interval(driver_id)
            elasticity = self.result.get_elasticity(driver_id)
            sensitivity = self.result.get_sensitivity_metric(driver_id)
            tornado_neg, tornado_pos = self.result.get_tornado_value(driver_id)

            lines.append(f"\n{driver_id}:")
            lines.append(f"  Base Win Probability:      {base_prob:7.2%}")
            lines.append(f"  95% Confidence Interval:   [{ci_low:7.2%}, {ci_high:7.2%}]")
            lines.append(f"  Sensitivity Metric:        {sensitivity:7.2%}")
            lines.append(f"  Elasticity:                {elasticity:7.3f}")
            lines.append(f"  Tornado - Negative Impact: {tornado_neg:7.2%}")
            lines.append(f"  Tornado - Positive Impact: {tornado_pos:7.2%}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def get_sensitivity_table_data(self) -> List[Dict[str, Any]]:
        """Get sensitivity metrics as table data.

        Returns:
            List of dictionaries suitable for DataFrame conversion
        """
        table_data = []

        for driver_id in self.result.drivers:
            base_prob = self.result.base_result.get_winner_probability(driver_id)
            ci_low, ci_high = self.result.get_confidence_interval(driver_id)
            elasticity = self.result.get_elasticity(driver_id)
            sensitivity = self.result.get_sensitivity_metric(driver_id)

            table_data.append({
                "driver_id": driver_id,
                "base_probability": base_prob,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "ci_width": ci_high - ci_low,
                "elasticity": elasticity,
                "sensitivity_metric": sensitivity,
            })

        return table_data

    def get_tornado_chart_data(self) -> TornadoChartData:
        """Get data for tornado chart visualization.

        Returns:
            TornadoChartData with impacts for each driver
        """
        drivers_impact = {}
        base_probs = {}

        for driver_id in self.result.drivers:
            negative_impact, positive_impact = self.result.get_tornado_value(driver_id)
            drivers_impact[driver_id] = (negative_impact, positive_impact)

            base_prob = self.result.base_result.get_winner_probability(driver_id)
            base_probs[driver_id] = base_prob

        return TornadoChartData(
            parameter_name=self.result.parameter_name,
            drivers=drivers_impact,
            base_probabilities=base_probs,
        )

    def get_probability_curves(self) -> Dict[str, List[Tuple[float, float]]]:
        """Get probability curves for all drivers.

        Returns:
            Dict mapping driver_id to list of (parameter_value, probability) tuples
        """
        curves = {}

        for driver_id in self.result.drivers:
            probs = self.result.get_win_probability_by_parameter(driver_id)

            # Convert to sorted list of tuples
            curve = sorted([(param_val, prob) for param_val, prob in probs.items()])
            curves[driver_id] = curve

        return curves

    def get_podium_curves(self) -> Dict[str, List[Tuple[float, float]]]:
        """Get podium probability curves for all drivers.

        Returns:
            Dict mapping driver_id to list of (parameter_value, probability) tuples
        """
        curves = {}

        for driver_id in self.result.drivers:
            probs = self.result.get_podium_probability_by_parameter(driver_id)

            # Convert to sorted list of tuples
            curve = sorted([(param_val, prob) for param_val, prob in probs.items()])
            curves[driver_id] = curve

        return curves

    def get_most_sensitive_driver(self) -> Tuple[str, float]:
        """Get driver most sensitive to parameter variation.

        Returns:
            Tuple of (driver_id, sensitivity_metric)
        """
        sensitivities = {}

        for driver_id in self.result.drivers:
            sensitivity = self.result.get_sensitivity_metric(driver_id)
            sensitivities[driver_id] = sensitivity

        most_sensitive_id = max(sensitivities, key=sensitivities.get)
        return most_sensitive_id, sensitivities[most_sensitive_id]

    def get_least_sensitive_driver(self) -> Tuple[str, float]:
        """Get driver least sensitive to parameter variation.

        Returns:
            Tuple of (driver_id, sensitivity_metric)
        """
        sensitivities = {}

        for driver_id in self.result.drivers:
            sensitivity = self.result.get_sensitivity_metric(driver_id)
            sensitivities[driver_id] = sensitivity

        least_sensitive_id = min(sensitivities, key=sensitivities.get)
        return least_sensitive_id, sensitivities[least_sensitive_id]

    def get_elasticity_ranking(self) -> List[Tuple[str, float]]:
        """Rank drivers by elasticity (most to least elastic).

        Returns:
            Sorted list of (driver_id, elasticity) tuples
        """
        elasticities = {}

        for driver_id in self.result.drivers:
            elasticity = self.result.get_elasticity(driver_id)
            elasticities[driver_id] = elasticity

        # Sort by elasticity descending
        ranking = sorted(elasticities.items(), key=lambda x: abs(x[1]), reverse=True)
        return ranking

    def export_json(self) -> str:
        """Export report as JSON string.

        Returns:
            JSON string with all report data
        """
        summary = self.result.get_summary()

        # Add additional analysis
        summary["most_sensitive_driver"] = self.get_most_sensitive_driver()[0]
        summary["least_sensitive_driver"] = self.get_least_sensitive_driver()[0]
        summary["elasticity_ranking"] = [
            {"driver_id": d, "elasticity": e} for d, e in self.get_elasticity_ranking()
        ]

        # Add curves
        summary["win_probability_curves"] = {
            driver_id: [{"param_value": p, "probability": prob} for p, prob in curve]
            for driver_id, curve in self.get_probability_curves().items()
        }

        summary["podium_probability_curves"] = {
            driver_id: [{"param_value": p, "probability": prob} for p, prob in curve]
            for driver_id, curve in self.get_podium_curves().items()
        }

        return json.dumps(summary, indent=2)

    def get_key_findings(self) -> List[str]:
        """Generate key findings from sensitivity analysis.

        Returns:
            List of key finding strings
        """
        findings = []

        # Most sensitive driver
        most_sensitive_id, most_sensitive_val = self.get_most_sensitive_driver()
        findings.append(
            f"{most_sensitive_id} is most sensitive to {self.result.parameter_name} "
            f"(sensitivity metric: {most_sensitive_val:.2%})"
        )

        # Least sensitive driver
        least_sensitive_id, least_sensitive_val = self.get_least_sensitive_driver()
        findings.append(
            f"{least_sensitive_id} is least sensitive to {self.result.parameter_name} "
            f"(sensitivity metric: {least_sensitive_val:.2%})"
        )

        # Elasticity findings
        elasticity_ranking = self.get_elasticity_ranking()
        if elasticity_ranking:
            top_elastic = elasticity_ranking[0]
            findings.append(
                f"{top_elastic[0]} has highest elasticity ({top_elastic[1]:.3f}) - "
                "most responsive to parameter changes"
            )

        # Confidence interval findings
        for driver_id in self.result.drivers:
            ci_low, ci_high = self.result.get_confidence_interval(driver_id)
            ci_width = ci_high - ci_low
            findings.append(
                f"{driver_id} has 95% CI width of {ci_width:.2%} "
                f"(range: {ci_low:.2%} to {ci_high:.2%})"
            )

        return findings
