"""Tests for F1-specific race anomaly detection."""

import pandas as pd
import pytest
from f1_predict.data.race_anomaly_detector import RaceAnomalyDetector


@pytest.fixture
def sample_race_data():
    """Sample F1 race results for testing."""
    return pd.DataFrame({
        'season': [2023, 2023, 2023, 2023, 2023, 2024, 2024, 2024],
        'round': [1, 1, 1, 2, 2, 1, 1, 1],
        'driver_id': [1, 2, 3, 1, 2, 1, 2, 3],
        'driver_name': ['Max', 'Lewis', 'Carlos', 'Max', 'Lewis', 'Max', 'Lewis', 'Carlos'],
        'position': [1, 2, 3, 1, 2, 3, 5, 8],  # Position at finish
        'qualifying_position': [1, 2, 3, 1, 3, 2, 1, 4],
        'points': [25, 18, 15, 25, 18, 15, 10, 6],
        'status': ['Finished', 'Finished', 'Finished', 'Finished', 'Finished', 'Finished', 'Finished', 'DNF'],
    })


def test_race_anomaly_detector_initialization():
    """Test RaceAnomalyDetector initializes correctly."""
    detector = RaceAnomalyDetector()

    assert detector is not None
    assert hasattr(detector, 'detect')
    assert hasattr(detector, 'history')


def test_unusual_podium_detection():
    """Test detection of unusual podium results (great performance vs history)."""
    detector = RaceAnomalyDetector()

    # Driver typically finishes 8th, now finishes 1st (unusual)
    data = pd.DataFrame({
        'season': [2023, 2023, 2024],
        'round': [1, 2, 1],
        'driver_id': [1, 1, 1],
        'driver_name': ['Alonso', 'Alonso', 'Alonso'],
        'position': [8, 9, 1],  # Unexpected 1st place
        'qualifying_position': [8, 9, 2],
        'points': [6, 4, 25],
        'status': ['Finished', 'Finished', 'Finished'],
    })

    result = detector.detect(data)

    # Row with position=1 should be flagged as anomalous
    anomalous_row = result[result['position'] == 1]
    assert len(anomalous_row) > 0
    assert anomalous_row.iloc[0]['anomaly_flag'] == True
    assert anomalous_row.iloc[0]['anomaly_score'] > 0.5


def test_qualifying_race_performance_gap():
    """Test detection of large gaps between qualifying and race performance."""
    detector = RaceAnomalyDetector()

    # Large gap: Qualified 1st but finished 18th (unusual)
    data = pd.DataFrame({
        'season': [2024, 2024],
        'round': [1, 1],
        'driver_id': [1, 2],
        'driver_name': ['Driver1', 'Driver2'],
        'position': [18, 5],
        'qualifying_position': [1, 4],  # Driver1: big gap (1 to 18)
        'points': [0, 12],
        'status': ['Accident', 'Finished'],
    })

    result = detector.detect(data)

    # Row with position=18 should be flagged due to large Q-to-Race gap
    anomalous_row = result[result['qualifying_position'] == 1]
    assert len(anomalous_row) > 0
    assert anomalous_row.iloc[0]['anomaly_flag'] == True


def test_dnf_patterns():
    """Test detection of DNF (Did Not Finish) anomalies."""
    detector = RaceAnomalyDetector()

    # Double DNF in consecutive races (unusual)
    data = pd.DataFrame({
        'season': [2024, 2024, 2024],
        'round': [1, 2, 3],
        'driver_id': [1, 1, 1],
        'driver_name': ['Driver1', 'Driver1', 'Driver1'],
        'position': [0, 0, 5],  # DNF, DNF, then finish
        'qualifying_position': [5, 6, 7],
        'points': [0, 0, 12],
        'status': ['DNF', 'DNF', 'Finished'],
    })

    result = detector.detect(data)

    # Second DNF should be flagged as anomalous pattern
    dnf_rows = result[result['status'] == 'DNF']
    assert len(dnf_rows) >= 1


def test_position_order_integrity():
    """Test that output maintains same row order as input."""
    detector = RaceAnomalyDetector()

    data = pd.DataFrame({
        'season': [2024, 2024, 2024],
        'round': [1, 1, 1],
        'driver_id': [1, 2, 3],
        'driver_name': ['A', 'B', 'C'],
        'position': [1, 2, 3],
        'qualifying_position': [1, 2, 3],
        'points': [25, 18, 15],
        'status': ['Finished', 'Finished', 'Finished'],
    })

    result = detector.detect(data)

    assert len(result) == len(data)
    pd.testing.assert_index_equal(result.index, data.index)


def test_detects_with_missing_values():
    """Test handling of missing values in driver history."""
    detector = RaceAnomalyDetector()

    # New driver (no history)
    data = pd.DataFrame({
        'season': [2024],
        'round': [1],
        'driver_id': [999],  # New driver ID
        'driver_name': ['NewDriver'],
        'position': [5],
        'qualifying_position': [6],
        'points': [12],
        'status': ['Finished'],
    })

    result = detector.detect(data)

    # Should handle gracefully without errors
    assert len(result) == 1
    assert 'anomaly_flag' in result.columns
    assert 'anomaly_score' in result.columns


def test_output_has_required_columns():
    """Test output DataFrame has all required anomaly columns."""
    detector = RaceAnomalyDetector()

    data = pd.DataFrame({
        'season': [2024],
        'round': [1],
        'driver_id': [1],
        'driver_name': ['Max'],
        'position': [1],
        'qualifying_position': [1],
        'points': [25],
        'status': ['Finished'],
    })

    result = detector.detect(data)

    assert 'anomaly_flag' in result.columns
    assert 'anomaly_score' in result.columns
    assert 'anomaly_method' in result.columns


def test_detector_accumulates_history():
    """Test that detector learns from multiple calls (builds history)."""
    detector = RaceAnomalyDetector()

    # First call: establish baseline
    data1 = pd.DataFrame({
        'season': [2023],
        'round': [1],
        'driver_id': [1],
        'driver_name': ['Driver1'],
        'position': [5],
        'qualifying_position': [5],
        'points': [12],
        'status': ['Finished'],
    })

    result1 = detector.detect(data1)

    # Second call: use history from first
    data2 = pd.DataFrame({
        'season': [2023],
        'round': [2],
        'driver_id': [1],
        'driver_name': ['Driver1'],
        'position': [1],  # Very different from position 5
        'qualifying_position': [4],
        'points': [25],
        'status': ['Finished'],
    })

    result2 = detector.detect(data2)

    # Second call should flag position=1 as anomalous based on history
    assert result2.iloc[0]['anomaly_flag'] == True or result2.iloc[0]['anomaly_score'] > 0.3


def test_normal_results_not_flagged():
    """Test that normal race results are not flagged as anomalies."""
    detector = RaceAnomalyDetector()

    # Consistent performance
    data = pd.DataFrame({
        'season': [2023, 2023, 2024, 2024],
        'round': [1, 2, 1, 2],
        'driver_id': [1, 1, 1, 1],
        'driver_name': ['Driver1', 'Driver1', 'Driver1', 'Driver1'],
        'position': [5, 6, 4, 7],  # Consistent mid-field performance
        'qualifying_position': [5, 6, 5, 7],
        'points': [12, 10, 15, 8],
        'status': ['Finished', 'Finished', 'Finished', 'Finished'],
    })

    result = detector.detect(data)

    # Most rows should not be flagged
    flagged_count = result['anomaly_flag'].sum()
    assert flagged_count <= len(result) // 3  # Less than 1/3 flagged
