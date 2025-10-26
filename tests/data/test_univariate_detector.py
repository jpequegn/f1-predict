"""Tests for univariate anomaly detection."""

import pandas as pd

from f1_predict.data.univariate_detector import UnivariateDetector


def test_detector_initialization():
    """Test UnivariateDetector initializes with correct defaults."""
    detector = UnivariateDetector()

    assert detector is not None
    assert detector.z_score_threshold == 3.0
    assert detector.use_iqr is True
    assert detector.enabled is True


def test_detector_custom_initialization():
    """Test UnivariateDetector with custom parameters."""
    detector = UnivariateDetector(z_score_threshold=2.5, use_iqr=False, enabled=False)

    assert detector.z_score_threshold == 2.5
    assert detector.use_iqr is False
    assert detector.enabled is False


def test_detect_z_score_anomalies():
    """Test Z-score based anomaly detection."""
    detector = UnivariateDetector(z_score_threshold=3.0, use_iqr=False)

    # Create data with one clear outlier
    # Normal points around 10, outlier at 100 gives Z-score > 3.0
    data = pd.DataFrame(
        {
            "position": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "points": [
                10,
                11,
                9,
                10,
                12,
                9,
                11,
                10,
                9,
                10,
                11,
                10,
                9,
                11,
                100,
            ],  # 100 is outlier
        }
    )

    result = detector.detect(data)

    # Check columns added
    assert "anomaly_flag" in result.columns
    assert "anomaly_score" in result.columns
    assert "anomaly_method" in result.columns
    assert "anomaly_confidence" in result.columns

    # Check that outlier is detected
    assert result.iloc[-1]["anomaly_flag"] == True
    assert result.iloc[-1]["anomaly_score"] > 0
    assert "z_score" in result.iloc[-1]["anomaly_method"]

    # Check that normal values are not flagged
    assert result.iloc[0]["anomaly_flag"] == False


def test_detect_iqr_anomalies():
    """Test IQR based anomaly detection."""
    detector = UnivariateDetector(use_iqr=True)

    # Create data with outliers beyond IQR bounds
    data = pd.DataFrame(
        {
            "lap_time": [90.5, 91.0, 90.8, 91.2, 90.9, 91.1, 150.0],  # 150.0 is outlier
            "position": [1, 2, 3, 4, 5, 6, 7],
        }
    )

    result = detector.detect(data)

    # Check that outlier is detected
    assert result.iloc[-1]["anomaly_flag"] == True
    assert "iqr" in result.iloc[-1]["anomaly_method"]


def test_detect_with_missing_values():
    """Test detection handles missing values gracefully."""
    detector = UnivariateDetector()

    data = pd.DataFrame(
        {
            "position": [1, 2, 3, None, 5],
            "points": [25, 18, 15, 12, 10],
        }
    )

    result = detector.detect(data)

    # Should not crash and should return same number of rows
    assert len(result) == len(data)
    assert "anomaly_flag" in result.columns


def test_categorical_columns_skipped():
    """Test that categorical/non-numeric columns are skipped."""
    detector = UnivariateDetector()

    data = pd.DataFrame(
        {
            "driver_name": ["Hamilton", "Verstappen", "Leclerc"],
            "team": ["Mercedes", "Red Bull", "Ferrari"],
            "points": [25, 18, 15],
        }
    )

    result = detector.detect(data)

    # Should complete without errors
    assert len(result) == len(data)
    assert "anomaly_flag" in result.columns


def test_disabled_detector_returns_unmodified():
    """Test that disabled detector returns data unchanged."""
    detector = UnivariateDetector(enabled=False)

    data = pd.DataFrame(
        {
            "points": [25, 18, 15, 2000],  # Has outlier
        }
    )

    result = detector.detect(data)

    # Should not add anomaly columns when disabled
    assert "anomaly_flag" not in result.columns


def test_empty_dataframe():
    """Test detection with empty DataFrame."""
    detector = UnivariateDetector()

    data = pd.DataFrame()

    result = detector.detect(data)

    # Should return empty DataFrame without crashing
    assert len(result) == 0


def test_single_row_dataframe():
    """Test detection with single row (cannot compute statistics)."""
    detector = UnivariateDetector()

    data = pd.DataFrame(
        {
            "points": [25],
        }
    )

    result = detector.detect(data)

    # Should handle gracefully
    assert len(result) == 1
    # Single row cannot be an anomaly
    if "anomaly_flag" in result.columns:
        assert result.iloc[0]["anomaly_flag"] == False


def test_all_identical_values():
    """Test detection when all values are identical (no variance)."""
    detector = UnivariateDetector()

    data = pd.DataFrame(
        {
            "points": [10, 10, 10, 10, 10],
        }
    )

    result = detector.detect(data)

    # Should handle zero variance gracefully
    assert len(result) == len(data)
    # No anomalies when all values identical
    if "anomaly_flag" in result.columns:
        assert result["anomaly_flag"].sum() == 0


def test_combined_z_score_and_iqr():
    """Test detection using both Z-score and IQR methods."""
    detector = UnivariateDetector(use_iqr=True)

    data = pd.DataFrame(
        {
            "points": [25, 18, 15, 12, 10, 8, 6, 4, 2, 2000],  # 2000 is extreme outlier
        }
    )

    result = detector.detect(data)

    # Extreme outlier should be caught by both methods
    assert result.iloc[-1]["anomaly_flag"] == True
    assert result.iloc[-1]["anomaly_score"] > 0


def test_anomaly_confidence_scores():
    """Test that confidence scores are calculated properly."""
    detector = UnivariateDetector()

    data = pd.DataFrame(
        {
            "points": [25, 18, 15, 12, 10, 8, 6, 4, 2, 2000],
        }
    )

    result = detector.detect(data)

    # Confidence should be between 0 and 1
    assert (result["anomaly_confidence"] >= 0).all()
    assert (result["anomaly_confidence"] <= 1).all()

    # Higher anomaly should have higher confidence
    outlier_confidence = result.iloc[-1]["anomaly_confidence"]
    normal_confidence = result.iloc[0]["anomaly_confidence"]
    assert outlier_confidence > normal_confidence
