"""Unit tests for sensitivity analysis reports."""

import pytest
import json

from f1_predict.simulation.core.driver_state import DriverState
from f1_predict.simulation.core.race_state import CircuitContext
from f1_predict.simulation.engine.simulator import MonteCarloSimulator
from f1_predict.simulation.analysis.scenario_builder import ScenarioBuilder
from f1_predict.simulation.analysis.sensitivity_analyzer import (
    ParameterType,
    SensitivityAnalyzer,
    SensitivityResult,
)
from f1_predict.simulation.analysis.sensitivity_report import (
    SensitivityReport,
    TornadoChartData,
)


class TestSensitivityReport:
    """Test SensitivityReport class."""

    @pytest.fixture
    def sensitivity_result(self):
        """Create a sensitivity result for testing."""
        circuit = CircuitContext(circuit_name="Albert Park", total_laps=58)
        drivers = [
            DriverState("VER", "Max Verstappen", expected_lap_time=81.5),
            DriverState("HAM", "Lewis Hamilton", expected_lap_time=82.0),
        ]
        scenario = ScenarioBuilder("baseline", circuit).with_drivers(drivers).build()
        simulator = MonteCarloSimulator(circuit=circuit, random_state=42)

        analyzer = SensitivityAnalyzer(simulator, scenario, n_simulations=10)
        return analyzer.vary_driver_pace("VER", pace_deltas=[-1.0, -0.5, 0.0, 0.5, 1.0])

    def test_sensitivity_report_creation(self, sensitivity_result):
        """Test creating sensitivity report."""
        report = SensitivityReport(sensitivity_result)
        assert report.result == sensitivity_result

    def test_generate_summary_text(self, sensitivity_result):
        """Test generating summary text."""
        report = SensitivityReport(sensitivity_result)
        summary = report.generate_summary_text()

        assert isinstance(summary, str)
        assert "SENSITIVITY ANALYSIS REPORT" in summary
        assert "VER" in summary
        assert "HAM" in summary
        assert "Base Win Probability" in summary

    def test_get_sensitivity_table_data(self, sensitivity_result):
        """Test getting table data."""
        report = SensitivityReport(sensitivity_result)
        table_data = report.get_sensitivity_table_data()

        assert isinstance(table_data, list)
        assert len(table_data) == 2
        assert "driver_id" in table_data[0]
        assert "base_probability" in table_data[0]
        assert "ci_lower" in table_data[0]
        assert "ci_upper" in table_data[0]

    def test_get_tornado_chart_data(self, sensitivity_result):
        """Test getting tornado chart data."""
        report = SensitivityReport(sensitivity_result)
        tornado_data = report.get_tornado_chart_data()

        assert isinstance(tornado_data, TornadoChartData)
        assert tornado_data.parameter_name == sensitivity_result.parameter_name
        assert "VER" in tornado_data.drivers
        assert "HAM" in tornado_data.drivers
        assert "VER" in tornado_data.base_probabilities

    def test_get_probability_curves(self, sensitivity_result):
        """Test getting probability curves."""
        report = SensitivityReport(sensitivity_result)
        curves = report.get_probability_curves()

        assert isinstance(curves, dict)
        assert "VER" in curves
        assert "HAM" in curves

        # Each curve should be list of (parameter_value, probability) tuples
        ver_curve = curves["VER"]
        assert isinstance(ver_curve, list)
        assert len(ver_curve) > 0
        assert isinstance(ver_curve[0], tuple)
        assert len(ver_curve[0]) == 2

    def test_get_podium_curves(self, sensitivity_result):
        """Test getting podium probability curves."""
        report = SensitivityReport(sensitivity_result)
        curves = report.get_podium_curves()

        assert isinstance(curves, dict)
        assert "VER" in curves
        assert "HAM" in curves

    def test_get_most_sensitive_driver(self, sensitivity_result):
        """Test finding most sensitive driver."""
        report = SensitivityReport(sensitivity_result)
        driver_id, sensitivity = report.get_most_sensitive_driver()

        assert isinstance(driver_id, str)
        assert isinstance(sensitivity, float)
        assert sensitivity >= 0

    def test_get_least_sensitive_driver(self, sensitivity_result):
        """Test finding least sensitive driver."""
        report = SensitivityReport(sensitivity_result)
        driver_id, sensitivity = report.get_least_sensitive_driver()

        assert isinstance(driver_id, str)
        assert isinstance(sensitivity, float)
        assert sensitivity >= 0

    def test_get_elasticity_ranking(self, sensitivity_result):
        """Test elasticity ranking."""
        report = SensitivityReport(sensitivity_result)
        ranking = report.get_elasticity_ranking()

        assert isinstance(ranking, list)
        assert len(ranking) > 0

        # Each entry should be (driver_id, elasticity) tuple
        for driver_id, elasticity in ranking:
            assert isinstance(driver_id, str)
            assert isinstance(elasticity, float)

    def test_export_json(self, sensitivity_result):
        """Test exporting to JSON."""
        report = SensitivityReport(sensitivity_result)
        json_str = report.export_json()

        assert isinstance(json_str, str)

        # Parse JSON to verify it's valid
        data = json.loads(json_str)
        assert "parameter_name" in data
        assert "sensitivity_metrics" in data
        assert "most_sensitive_driver" in data
        assert "least_sensitive_driver" in data

    def test_get_key_findings(self, sensitivity_result):
        """Test generating key findings."""
        report = SensitivityReport(sensitivity_result)
        findings = report.get_key_findings()

        assert isinstance(findings, list)
        assert len(findings) > 0

        # All findings should be strings
        for finding in findings:
            assert isinstance(finding, str)
            assert len(finding) > 0


class TestTornadoChartData:
    """Test TornadoChartData class."""

    def test_tornado_chart_data_creation(self):
        """Test creating tornado chart data."""
        drivers_impact = {
            "VER": (0.05, 0.1),
            "HAM": (0.02, 0.08),
        }
        base_probs = {
            "VER": 0.5,
            "HAM": 0.3,
        }

        tornado_data = TornadoChartData(
            parameter_name="pace_VER",
            drivers=drivers_impact,
            base_probabilities=base_probs,
        )

        assert tornado_data.parameter_name == "pace_VER"
        assert len(tornado_data.drivers) == 2
        assert tornado_data.drivers["VER"] == (0.05, 0.1)
        assert tornado_data.base_probabilities["VER"] == 0.5
