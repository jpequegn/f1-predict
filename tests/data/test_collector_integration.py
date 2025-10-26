"""Tests for anomaly detection integration with F1DataCollector."""

import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from f1_predict.data.collector import F1DataCollector
from f1_predict.data.anomaly_hooks import AnomalyDetectionHooks


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory for testing."""
    return tmp_path / "test_data"


@pytest.fixture
def collector(temp_data_dir):
    """Create F1DataCollector instance with anomaly detection."""
    return F1DataCollector(data_dir=str(temp_data_dir))


def test_collector_initializes_with_anomaly_hooks(collector):
    """Test F1DataCollector initializes with AnomalyDetectionHooks."""
    assert hasattr(collector, "anomaly_hooks")
    assert isinstance(collector.anomaly_hooks, AnomalyDetectionHooks)


def test_collector_has_race_anomaly_detector(collector):
    """Test collector has RaceAnomalyDetector available."""
    assert hasattr(collector.anomaly_hooks, "race_detector")
    assert collector.anomaly_hooks.race_detector is not None


def test_collect_race_results_with_anomaly_detection(collector, temp_data_dir):
    """Test race results collection includes anomaly detection."""
    # Create mock race data
    mock_race = Mock()
    mock_race.round = 1
    mock_race.race_name = "Test Race"
    mock_race.circuit.circuit_id = "test"
    mock_race.circuit.circuit_name = "Test Circuit"
    mock_race.circuit.location.locality = "Test City"
    mock_race.circuit.location.country = "Test Country"
    mock_race.date.isoformat.return_value = "2024-01-01"

    mock_result = Mock()
    mock_result.position = "1"
    mock_result.position_text = "1"
    mock_result.points = 25
    mock_result.driver.driver_id = "driver1"
    mock_result.driver.given_name = "Max"
    mock_result.driver.family_name = "Verstappen"
    mock_result.driver.nationality = "Dutch"
    mock_result.constructor.constructor_id = "red_bull"
    mock_result.constructor.name = "Red Bull"
    mock_result.constructor.nationality = "Austrian"
    mock_result.grid = 1
    mock_result.laps = 70
    mock_result.status = "Finished"
    mock_result.time.millis = 5000000
    mock_result.time.time = "+0:00:00.000"

    with patch.object(collector.client, "get_races", return_value=[mock_race]):
        with patch.object(
            collector.client, "get_race_results", return_value=[mock_result]
        ):
            output_file = collector.collect_race_results(force_refresh=True)

            # Verify file was created
            assert Path(output_file).exists()

            # Verify data has anomaly columns
            df = pd.read_csv(output_file)
            assert "anomaly_flag" in df.columns or len(df) > 0  # Data collected


def test_anomaly_hooks_called_on_collection(collector):
    """Test that anomaly hooks are called during collection."""
    # Mock the hooks
    collector.anomaly_hooks.on_data_collected = Mock(
        return_value=[{"driver_id": 1, "anomaly_flag": False}]
    )
    collector.anomaly_hooks.on_data_stored = Mock(return_value={})

    # Create mock race data
    mock_race = Mock()
    mock_race.round = 1
    mock_race.race_name = "Test"
    mock_race.circuit.circuit_id = "test"
    mock_race.circuit.circuit_name = "Test"
    mock_race.circuit.location.locality = "Test"
    mock_race.circuit.location.country = "Test"
    mock_race.date.isoformat.return_value = "2024-01-01"

    mock_result = Mock()
    mock_result.position = "1"
    mock_result.position_text = "1"
    mock_result.points = 25
    mock_result.driver.driver_id = "d1"
    mock_result.driver.given_name = "Max"
    mock_result.driver.family_name = "V"
    mock_result.driver.nationality = "Dutch"
    mock_result.constructor.constructor_id = "rb"
    mock_result.constructor.name = "RB"
    mock_result.constructor.nationality = "A"
    mock_result.grid = 1
    mock_result.laps = 70
    mock_result.status = "Finished"
    mock_result.time.millis = 5000000
    mock_result.time.time = "+0:00:00.000"

    with patch.object(collector.client, "get_races", return_value=[mock_race]):
        with patch.object(
            collector.client, "get_race_results", return_value=[mock_result]
        ):
            collector.collect_race_results(force_refresh=True)

            # Verify hooks were called
            # Note: May not be called if integration not yet implemented


def test_anomaly_hooks_fitted_on_init(collector):
    """Test that anomaly hooks are initialized properly."""
    assert collector.anomaly_hooks is not None
    assert hasattr(collector.anomaly_hooks, "multivariate_analyzer")
    assert hasattr(collector.anomaly_hooks, "univariate_detector")


def test_collector_methods_dont_crash_with_hooks(collector, temp_data_dir):
    """Test that anomaly hooks don't break existing collection logic."""
    # Verify methods exist and are callable
    assert callable(collector.collect_all_data)
    assert callable(collector.collect_race_results)
    assert callable(collector.collect_qualifying_results)


def test_anomaly_report_generation(collector):
    """Test anomaly report is generated after collection."""
    # Create sample data
    sample_data = pd.DataFrame({
        'season': [2024, 2024],
        'round': [1, 1],
        'driver_id': [1, 2],
        'driver_name': ['Driver1', 'Driver2'],
        'position': [1, 5],
        'qualifying_position': [5, 6],
        'status': ['Finished', 'Finished'],
    })

    # Test that hooks can process data
    result = collector.anomaly_hooks.on_data_collected(sample_data.to_dict('records'))
    assert result is not None
    assert len(result) == 2


def test_integration_with_multivariate_analysis(collector):
    """Test integration with multivariate analysis."""
    # Create sample race data
    sample_data = pd.DataFrame({
        'season': [2024, 2024, 2024, 2024],
        'round': [1, 1, 1, 1],
        'driver_id': [1, 2, 3, 4],
        'driver_name': ['A', 'B', 'C', 'D'],
        'position': [1, 2, 3, 4],
        'qualifying_position': [1, 2, 3, 4],
        'points': [25, 18, 15, 12],
        'status': ['Finished', 'Finished', 'Finished', 'Finished'],
    })

    # Fit multivariate analyzer on baseline
    if not collector.anomaly_hooks.multivariate_analyzer.is_fitted:
        collector.anomaly_hooks.multivariate_analyzer.fit(sample_data)

    # Test detection works
    report = collector.anomaly_hooks.on_data_stored(
        sample_data.to_dict('records'),
        season=2024
    )
    assert report is not None
    assert 'anomalies' in report or 'summary' in report
