# tests/data/test_anomaly_hooks.py
from f1_predict.data.anomaly_hooks import AnomalyDetectionHooks, AnomalyMetadata


def test_anomaly_metadata_to_dict():
    """Test AnomalyMetadata.to_dict() conversion."""
    metadata = AnomalyMetadata(
        anomaly_flag=True,
        anomaly_score=0.85,
        anomaly_method="isolation_forest",
        anomaly_confidence=0.92,
        features_involved=["position", "points"],
        explanation="Unusual performance detected",
    )
    result = metadata.to_dict()

    assert result["anomaly_flag"] is True
    assert result["anomaly_score"] == 0.85
    assert result["anomaly_method"] == "isolation_forest"
    assert result["anomaly_confidence"] == 0.92
    assert result["features_involved"] == ["position", "points"]
    assert result["explanation"] == "Unusual performance detected"


def test_hooks_initialization():
    """Test AnomalyDetectionHooks initializes with all detectors."""
    hooks = AnomalyDetectionHooks()

    assert hooks is not None
    assert hasattr(hooks, "univariate_detector")
    assert hasattr(hooks, "multivariate_analyzer")
    assert hasattr(hooks, "registry")


def test_on_data_collected_returns_data():
    """Test on_data_collected returns data with anomaly flags."""
    hooks = AnomalyDetectionHooks()

    data = [
        {"race_id": 1, "position": 1, "points": 25},
        {"race_id": 1, "position": 2, "points": 18},
    ]

    result = hooks.on_data_collected(data)

    assert result is not None
    assert len(result) == 2
    assert "_anomaly" in result[0]
    assert "anomaly_flag" in result[0]["_anomaly"]
    assert "anomaly_score" in result[0]["_anomaly"]


def test_on_data_stored_returns_report():
    """Test on_data_stored returns anomaly report."""
    hooks = AnomalyDetectionHooks()
    data = [{"race_id": 1, "position": 1}]
    report = hooks.on_data_stored(data, season=2024)

    assert "anomalies" in report
    assert "summary" in report
    assert isinstance(report["anomalies"], list)
    assert isinstance(report["summary"], dict)


def test_on_data_collected_handles_errors():
    """Test on_data_collected handles errors gracefully."""
    from unittest.mock import patch

    hooks = AnomalyDetectionHooks()

    # Mock AnomalyMetadata to raise exception
    with patch("f1_predict.data.anomaly_hooks.AnomalyMetadata") as mock_metadata:
        mock_metadata.side_effect = ValueError("Test error")
        data = [{"race_id": 1}]
        result = hooks.on_data_collected(data)
        # Should return original data despite error
        assert result == data


def test_on_data_stored_handles_errors():
    """Test on_data_stored handles errors gracefully."""
    from unittest.mock import patch

    hooks = AnomalyDetectionHooks()

    # Force an exception by patching dict constructor to fail
    original_dict = dict

    def failing_dict(*args, **kwargs):
        if kwargs.get("anomalies") is not None:
            raise ValueError("Test error in dict creation")
        return original_dict(*args, **kwargs)

    with patch("builtins.dict", side_effect=failing_dict):
        data = [{"race_id": 1}]
        result = hooks.on_data_stored(data, season=2024)
        # Should return fallback response
        assert "anomalies" in result
        assert "summary" in result

    # Also test normal graceful degradation with empty data
    result = hooks.on_data_stored([], season=2024)
    assert "anomalies" in result
    assert "summary" in result


def test_multivariate_analyzer_initialized():
    """Test that multivariate analyzer is properly initialized."""
    hooks = AnomalyDetectionHooks()

    assert hooks.multivariate_analyzer is not None
    assert hasattr(hooks.multivariate_analyzer, "fit")
    assert hasattr(hooks.multivariate_analyzer, "detect")
    assert hasattr(hooks.multivariate_analyzer, "is_fitted")


def test_on_data_stored_uses_multivariate_analyzer():
    """Test on_data_stored uses multivariate analyzer for detection."""
    hooks = AnomalyDetectionHooks()

    # Create realistic F1 data with some outliers
    data = [
        {"position": 1, "points": 25, "lap_time": 90.5},
        {"position": 2, "points": 18, "lap_time": 91.0},
        {"position": 3, "points": 15, "lap_time": 90.8},
        {"position": 4, "points": 12, "lap_time": 91.2},
        {"position": 5, "points": 10, "lap_time": 90.9},
        {"position": 6, "points": 8, "lap_time": 91.1},
        {"position": 7, "points": 6, "lap_time": 90.7},
        {"position": 8, "points": 4, "lap_time": 90.6},
        {"position": 9, "points": 2, "lap_time": 91.3},
        {"position": 10, "points": 1, "lap_time": 90.4},
        # Outlier - unrealistic performance
        {"position": 20, "points": 100, "lap_time": 200.0},
    ]

    report = hooks.on_data_stored(data, season=2024)

    # Verify report structure
    assert "anomalies" in report
    assert "summary" in report
    assert isinstance(report["anomalies"], list)

    # Verify summary contains expected keys
    summary = report["summary"]
    assert "total" in summary
    assert "critical" in summary
    assert "total_records" in summary
    assert "anomaly_rate" in summary

    # Verify multivariate analyzer was fitted
    assert hooks.multivariate_analyzer.is_fitted is True
