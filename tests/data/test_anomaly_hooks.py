# tests/data/test_anomaly_hooks.py
from f1_predict.data.anomaly_hooks import AnomalyDetectionHooks


def test_hooks_initialization():
    """Test AnomalyDetectionHooks initializes with all detectors."""
    hooks = AnomalyDetectionHooks()

    assert hooks is not None
    assert hasattr(hooks, 'univariate_detector')
    assert hasattr(hooks, 'multivariate_analyzer')
    assert hasattr(hooks, 'registry')


def test_on_data_collected_returns_data():
    """Test on_data_collected returns data with anomaly flags."""
    hooks = AnomalyDetectionHooks()

    data = [
        {'race_id': 1, 'position': 1, 'points': 25},
        {'race_id': 1, 'position': 2, 'points': 18},
    ]

    result = hooks.on_data_collected(data)

    assert result is not None
    assert len(result) == 2
    assert '_anomaly' in result[0]
    assert 'anomaly_flag' in result[0]['_anomaly']
    assert 'anomaly_score' in result[0]['_anomaly']
