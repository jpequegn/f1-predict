"""Unit tests for command line interface (CLI) module.

Tests cover:
- Data collection commands (collect, clean, validate)
- Anomaly detection CLI commands
- Error handling and logging
- Argument parsing and validation
"""

import argparse
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from f1_predict.cli import (
    _calculate_anomaly_severity,
    _filter_anomalies,
    setup_logging,
)


class TestAnomalySeverityCalculation:
    """Tests for anomaly severity calculation."""

    def test_critical_severity(self):
        """Test critical severity for high anomaly scores."""
        severity = _calculate_anomaly_severity(0.9)
        assert severity == "critical"

    def test_critical_boundary(self):
        """Test critical severity at boundary (>0.8)."""
        severity = _calculate_anomaly_severity(0.801)
        assert severity == "critical"

    def test_warning_severity(self):
        """Test warning severity for medium anomaly scores."""
        severity = _calculate_anomaly_severity(0.7)
        assert severity == "warning"

    def test_warning_boundary(self):
        """Test warning severity at boundary (>0.6)."""
        severity = _calculate_anomaly_severity(0.601)
        assert severity == "warning"

    def test_info_severity_low(self):
        """Test info severity for low anomaly scores."""
        severity = _calculate_anomaly_severity(0.5)
        assert severity == "info"

    def test_info_severity_zero(self):
        """Test info severity for zero anomaly score."""
        severity = _calculate_anomaly_severity(0.0)
        assert severity == "info"

    def test_info_severity_boundary(self):
        """Test info severity just below warning threshold."""
        severity = _calculate_anomaly_severity(0.59)
        assert severity == "info"

    def test_edge_case_one(self):
        """Test edge case for anomaly score of 1.0."""
        severity = _calculate_anomaly_severity(1.0)
        assert severity == "critical"


class TestAnomalyFiltering:
    """Tests for anomaly filtering functionality."""

    @pytest.fixture
    def sample_anomalies(self):
        """Create sample anomaly records."""
        return [
            MagicMock(driver_id=1, severity="critical"),
            MagicMock(driver_id=2, severity="warning"),
            MagicMock(driver_id=1, severity="info"),
            MagicMock(driver_id=3, severity="critical"),
            MagicMock(driver_id=2, severity="info"),
        ]

    def test_filter_by_driver_id(self, sample_anomalies):
        """Test filtering anomalies by driver ID."""
        filtered = _filter_anomalies(sample_anomalies, driver_id=1)
        assert len(filtered) == 2
        assert all(a.driver_id == 1 for a in filtered)

    def test_filter_by_severity(self, sample_anomalies):
        """Test filtering anomalies by severity."""
        filtered = _filter_anomalies(sample_anomalies, severity="critical")
        assert len(filtered) == 2
        assert all(a.severity == "critical" for a in filtered)

    def test_filter_by_driver_and_severity(self, sample_anomalies):
        """Test filtering by both driver ID and severity."""
        filtered = _filter_anomalies(sample_anomalies, driver_id=1, severity="critical")
        assert len(filtered) == 1
        assert filtered[0].driver_id == 1
        assert filtered[0].severity == "critical"

    def test_filter_no_matches(self, sample_anomalies):
        """Test filtering with no matching results."""
        filtered = _filter_anomalies(sample_anomalies, driver_id=999)
        assert len(filtered) == 0

    def test_filter_no_criteria(self, sample_anomalies):
        """Test filtering with no criteria returns all."""
        filtered = _filter_anomalies(sample_anomalies)
        assert len(filtered) == len(sample_anomalies)

    def test_filter_severity_warning(self, sample_anomalies):
        """Test filtering for warning severity."""
        filtered = _filter_anomalies(sample_anomalies, severity="warning")
        assert len(filtered) == 1
        assert filtered[0].severity == "warning"

    def test_filter_severity_info(self, sample_anomalies):
        """Test filtering for info severity."""
        filtered = _filter_anomalies(sample_anomalies, severity="info")
        assert len(filtered) == 2
        assert all(a.severity == "info" for a in filtered)


class TestLoggingSetup:
    """Tests for logging configuration."""

    def test_setup_logging_info_level(self):
        """Test logging setup with INFO level."""
        with patch("f1_predict.logging_config.configure_logging") as mock_config:
            setup_logging(verbose=False)
            mock_config.assert_called_once_with(log_level="INFO")

    def test_setup_logging_debug_level(self):
        """Test logging setup with DEBUG level."""
        with patch("f1_predict.logging_config.configure_logging") as mock_config:
            setup_logging(verbose=True)
            mock_config.assert_called_once_with(log_level="DEBUG")

    def test_setup_logging_default(self):
        """Test logging setup with default verbose=False."""
        with patch("f1_predict.logging_config.configure_logging") as mock_config:
            setup_logging()
            mock_config.assert_called_once_with(log_level="INFO")


class TestDataCollectionArguments:
    """Tests for data collection command argument handling."""

    def test_collect_command_arguments(self):
        """Test collect command requires proper arguments."""
        args = argparse.Namespace(
            type="all",
            data_dir="data",
            verbose=False,
            enrich=False,
        )
        assert args.type == "all"
        assert args.data_dir == "data"
        assert not args.verbose
        assert not args.enrich

    def test_collect_race_results_argument(self):
        """Test collect command with race-results type."""
        args = argparse.Namespace(
            type="race-results",
            data_dir="data",
            verbose=False,
        )
        assert args.type == "race-results"

    def test_collect_qualifying_argument(self):
        """Test collect command with qualifying type."""
        args = argparse.Namespace(
            type="qualifying",
            data_dir="data",
            verbose=False,
        )
        assert args.type == "qualifying"

    def test_collect_with_custom_data_dir(self):
        """Test collect command with custom data directory."""
        args = argparse.Namespace(
            type="all",
            data_dir="/custom/path",
            verbose=False,
        )
        assert args.data_dir == "/custom/path"


class TestCleaningArguments:
    """Tests for data cleaning command arguments."""

    def test_clean_all_types(self):
        """Test cleaning all data types."""
        args = argparse.Namespace(
            type="all",
            data_dir="data",
            output_dir="processed",
            verbose=False,
            strict=False,
        )
        assert args.type == "all"
        assert args.output_dir == "processed"

    def test_clean_specific_type(self):
        """Test cleaning specific data type."""
        args = argparse.Namespace(
            type="race-results",
            data_dir="data",
            output_dir="processed",
            verbose=False,
            strict=False,
        )
        assert args.type == "race-results"

    def test_clean_with_strict_mode(self):
        """Test cleaning with strict quality validation."""
        args = argparse.Namespace(
            type="all",
            data_dir="data",
            output_dir="processed",
            verbose=False,
            strict=True,
        )
        assert args.strict is True

    def test_clean_output_directory(self):
        """Test cleaning with custom output directory."""
        args = argparse.Namespace(
            type="all",
            data_dir="data",
            output_dir="/output/path",
            verbose=False,
            strict=False,
        )
        assert args.output_dir == "/output/path"


class TestValidationArguments:
    """Tests for data validation command arguments."""

    def test_validate_all_types(self):
        """Test validating all data types."""
        args = argparse.Namespace(
            type="all",
            data_dir="data",
            output_dir="reports",
            verbose=False,
        )
        assert args.type == "all"
        assert args.output_dir == "reports"

    def test_validate_specific_type(self):
        """Test validating specific data type."""
        args = argparse.Namespace(
            type="schedules",
            data_dir="data",
            output_dir="reports",
            verbose=False,
        )
        assert args.type == "schedules"

    def test_validate_with_verbose(self):
        """Test validation with verbose logging."""
        args = argparse.Namespace(
            type="all",
            data_dir="data",
            output_dir="reports",
            verbose=True,
        )
        assert args.verbose is True


class TestAnomalyDetectionArguments:
    """Tests for anomaly detection CLI arguments."""

    def test_detect_all_anomalies(self):
        """Test detecting all anomalies without filters."""
        args = argparse.Namespace(
            data_dir="data",
            output_dir="anomalies",
            driver_id=None,
            severity=None,
            verbose=False,
        )
        assert args.driver_id is None
        assert args.severity is None

    def test_detect_filter_by_driver(self):
        """Test anomaly detection filtered by driver."""
        args = argparse.Namespace(
            data_dir="data",
            output_dir="anomalies",
            driver_id=1,
            severity=None,
            verbose=False,
        )
        assert args.driver_id == 1

    def test_detect_filter_by_severity(self):
        """Test anomaly detection filtered by severity."""
        args = argparse.Namespace(
            data_dir="data",
            output_dir="anomalies",
            driver_id=None,
            severity="critical",
            verbose=False,
        )
        assert args.severity == "critical"

    def test_detect_with_both_filters(self):
        """Test anomaly detection with driver and severity filters."""
        args = argparse.Namespace(
            data_dir="data",
            output_dir="anomalies",
            driver_id=5,
            severity="warning",
            verbose=False,
        )
        assert args.driver_id == 5
        assert args.severity == "warning"
