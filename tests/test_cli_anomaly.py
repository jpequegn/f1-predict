"""Tests for anomaly analysis CLI commands."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from f1_predict import logging_config
from f1_predict.cli import (
    analyze_registry,
    create_parser,
    detect_anomalies,
    export_anomalies,
)
from f1_predict.data.anomaly_registry import AnomalyRecord, AnomalyRegistry


@pytest.fixture(scope="session", autouse=True)
def configure_logging_for_tests():
    """Configure logging for test environment."""
    logging_config.configure_logging(log_level="INFO")


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory for testing."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return str(data_dir)


@pytest.fixture
def temp_anomalies_dir(tmp_path):
    """Create temporary anomalies directory."""
    anomalies_dir = tmp_path / "anomalies"
    anomalies_dir.mkdir()
    return str(anomalies_dir)


@pytest.fixture
def sample_registry(temp_anomalies_dir):
    """Create a sample anomaly registry with test data."""
    registry = AnomalyRegistry(temp_anomalies_dir)

    # Add sample anomalies
    records = [
        AnomalyRecord(
            season=2024, race_round=1, driver_id=1, driver_name="Max",
            anomaly_type="podium_anomaly", anomaly_score=0.75, severity="warning"
        ),
        AnomalyRecord(
            season=2024, race_round=1, driver_id=2, driver_name="Lewis",
            anomaly_type="q_race_gap", anomaly_score=0.85, severity="critical"
        ),
        AnomalyRecord(
            season=2024, race_round=2, driver_id=1, driver_name="Max",
            anomaly_type="dnf_pattern", anomaly_score=0.6, severity="info"
        ),
    ]

    for record in records:
        registry.add_anomaly(record)

    registry.save()
    return registry


def test_create_parser_has_detect_anomalies_command():
    """Test that parser has detect-anomalies command."""
    parser = create_parser()
    assert parser is not None
    # Parse a sample command to verify it works
    args = parser.parse_args(["detect-anomalies", "--data-dir", "data"])
    assert args.command == "detect-anomalies"
    assert args.data_dir == "data"


def test_create_parser_has_analyze_registry_command():
    """Test that parser has analyze-registry command."""
    parser = create_parser()
    args = parser.parse_args(["analyze-registry", "--registry-dir", "data/anomalies"])
    assert args.command == "analyze-registry"
    assert args.registry_dir == "data/anomalies"


def test_create_parser_has_export_anomalies_command():
    """Test that parser has export-anomalies command."""
    parser = create_parser()
    args = parser.parse_args([
        "export-anomalies",
        "--registry-dir", "data/anomalies",
        "--output-dir", "exports"
    ])
    assert args.command == "export-anomalies"
    assert args.registry_dir == "data/anomalies"
    assert args.output_dir == "exports"


def test_detect_anomalies_with_mock(temp_data_dir):
    """Test detect-anomalies command execution."""
    # Create mock args
    args = Mock()
    args.data_dir = temp_data_dir
    args.output_dir = temp_data_dir
    args.severity = None
    args.verbose = False

    # Create necessary directory structure
    raw_dir = Path(temp_data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create sample race results CSV
    import csv
    race_file = raw_dir / "race_results.csv"
    with open(race_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "season", "round", "driver_id", "driver_name",
            "anomaly_flag", "anomaly_score", "anomaly_method"
        ])
        writer.writeheader()
        writer.writerow({
            "season": 2024,
            "round": 1,
            "driver_id": 1,
            "driver_name": "Max",
            "anomaly_flag": False,
            "anomaly_score": 0.0,
            "anomaly_method": "none"
        })

    # Mock the RaceAnomalyDetector to avoid actual detection
    with patch("f1_predict.cli.RaceAnomalyDetector") as mock_detector_class:
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector

        # Mock the detect method to return DataFrame with no anomalies
        import pandas as pd
        df = pd.read_csv(race_file)
        df["anomaly_flag"] = False
        mock_detector.detect.return_value = df

        detect_anomalies(args)

        # Verify detector was instantiated
        mock_detector_class.assert_called_once()


def test_analyze_registry_with_sample_data(sample_registry, capsys):
    """Test analyze-registry command with sample data."""
    args = Mock()
    args.registry_dir = sample_registry.storage_dir
    args.filter_driver = None
    args.filter_severity = None
    args.format = "text"
    args.verbose = False

    analyze_registry(args)

    captured = capsys.readouterr()
    assert "Total anomalies" in captured.out or len(captured.out) > 0


def test_export_anomalies_to_csv(sample_registry, tmp_path):
    """Test exporting anomalies to CSV format."""
    output_dir = tmp_path / "exports"
    output_dir.mkdir()

    args = Mock()
    args.registry_dir = sample_registry.storage_dir
    args.output_dir = str(output_dir)
    args.format = "csv"
    args.verbose = False

    export_anomalies(args)

    # Verify CSV file was created
    csv_file = output_dir / "anomalies.csv"
    assert csv_file.exists()


def test_export_anomalies_to_json(sample_registry, tmp_path):
    """Test exporting anomalies to JSON format."""
    output_dir = tmp_path / "exports"
    output_dir.mkdir()

    args = Mock()
    args.registry_dir = sample_registry.storage_dir
    args.output_dir = str(output_dir)
    args.format = "json"
    args.verbose = False

    export_anomalies(args)

    # Verify JSON file was created
    json_file = output_dir / "anomalies.json"
    assert json_file.exists()

    # Verify JSON is valid
    with open(json_file) as f:
        data = json.load(f)
    assert "anomalies" in data
    assert len(data["anomalies"]) == 3


def test_detect_anomalies_command_in_parser():
    """Test detect-anomalies command can be parsed with all options."""
    parser = create_parser()
    args = parser.parse_args([
        "detect-anomalies",
        "--data-dir", "data",
        "--output-dir", "data/anomalies",
        "--severity", "critical"
    ])
    assert args.command == "detect-anomalies"
    assert args.data_dir == "data"
    assert args.output_dir == "data/anomalies"
    assert args.severity == "critical"


def test_analyze_registry_command_with_filters():
    """Test analyze-registry command with filtering options."""
    parser = create_parser()
    args = parser.parse_args([
        "analyze-registry",
        "--registry-dir", "data/anomalies",
        "--filter-driver", "1",
        "--filter-severity", "warning"
    ])
    assert args.command == "analyze-registry"
    assert args.filter_driver == "1"
    assert args.filter_severity == "warning"


def test_export_anomalies_with_format_option():
    """Test export-anomalies command with format option."""
    parser = create_parser()
    args = parser.parse_args([
        "export-anomalies",
        "--registry-dir", "data/anomalies",
        "--output-dir", "exports",
        "--format", "json"
    ])
    assert args.command == "export-anomalies"
    assert args.format == "json"
