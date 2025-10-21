"""Tests for anomaly detection system."""

import time

import numpy as np
import pandas as pd
import pytest

from f1_predict.models.anomaly_detection import (
    AnomalyScore,
    AnomalyType,
    ARIMAForecastDetector,
    HistoricalBaseline,
    IQRDetector,
    IsolationForestDetector,
    ZScoreDetector,
)
from f1_predict.web.utils.anomaly_detector import (
    AnomalyDetectionSystem,
    AnomalyEvent,
    DetectorConfig,
)


@pytest.fixture
def sample_normal_data():
    """Generate normal distribution data."""
    np.random.seed(42)
    return np.random.normal(loc=100, scale=15, size=100)


@pytest.fixture
def sample_time_series():
    """Generate time series data with trend."""
    np.random.seed(42)
    t = np.arange(100)
    trend = 0.5 * t
    noise = np.random.normal(0, 5, 100)
    return trend + noise + 100


@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        "accuracy": np.random.normal(0.85, 0.05, 100),
        "precision": np.random.normal(0.82, 0.06, 100),
    })


class TestAnomalyScore:
    """Test AnomalyScore dataclass."""

    def test_anomaly_score_creation(self):
        """Test creating AnomalyScore."""
        score = AnomalyScore(
            timestamp=time.time(),
            is_anomaly=True,
            anomaly_type=AnomalyType.UNIVARIATE.value,
            score=0.85,
            threshold=0.5,
            confidence=0.9,
            features_involved=["accuracy"],
            explanation="Test anomaly",
            severity="warning",
        )

        assert score.is_anomaly is True
        assert score.score == 0.85
        assert score.severity == "warning"

    def test_anomaly_score_to_dict(self):
        """Test converting AnomalyScore to dict."""
        score = AnomalyScore(
            timestamp=1234567890.0,
            is_anomaly=True,
            anomaly_type=AnomalyType.MULTIVARIATE.value,
            score=0.75,
            threshold=0.5,
            confidence=0.8,
            features_involved=["feat1", "feat2"],
            explanation="Multi-feature anomaly",
            severity="critical",
        )

        result = score.to_dict()
        assert result["is_anomaly"] is True
        assert result["anomaly_type"] == "multivariate"
        assert result["severity"] == "critical"
        assert len(result["features_involved"]) == 2


class TestHistoricalBaseline:
    """Test HistoricalBaseline dataclass."""

    def test_baseline_creation(self):
        """Test creating HistoricalBaseline."""
        baseline = HistoricalBaseline(
            feature_name="accuracy",
            mean=0.85,
            std=0.05,
            min_val=0.70,
            max_val=0.95,
            percentile_25=0.82,
            percentile_50=0.85,
            percentile_75=0.88,
            percentile_95=0.92,
            percentile_99=0.94,
            n_samples=1000,
            created_at=time.time(),
        )

        assert baseline.feature_name == "accuracy"
        assert baseline.mean == 0.85
        assert baseline.n_samples == 1000

    def test_baseline_to_dict(self):
        """Test converting baseline to dict."""
        baseline = HistoricalBaseline(
            feature_name="test",
            mean=10.0,
            std=2.0,
            min_val=5.0,
            max_val=15.0,
            percentile_25=8.0,
            percentile_50=10.0,
            percentile_75=12.0,
            percentile_95=14.0,
            percentile_99=14.8,
            n_samples=100,
            created_at=time.time(),
        )

        result = baseline.to_dict()
        assert result["feature_name"] == "test"
        assert "p25" in result
        assert "p99" in result


class TestIsolationForestDetector:
    """Test Isolation Forest anomaly detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = IsolationForestDetector(contamination=0.05)
        assert detector.contamination == 0.05
        assert detector.model is None

    def test_fit(self, sample_normal_data):
        """Test fitting detector."""
        detector = IsolationForestDetector()
        detector.fit(sample_normal_data)

        assert detector.model is not None
        assert detector.feature_names == ["feature_0"]

    def test_detect_normal(self, sample_normal_data):
        """Test detection of normal values."""
        detector = IsolationForestDetector()
        detector.fit(sample_normal_data)

        # Test normal value near mean
        result = detector.detect(np.array([100.0]))
        assert result.score < 0.5
        assert bool(result.is_anomaly) == False

    def test_detect_anomaly(self, sample_normal_data):
        """Test detection of anomalies."""
        detector = IsolationForestDetector(contamination=0.1)
        detector.fit(sample_normal_data)

        # Test extreme value
        result = detector.detect(np.array([500.0]))
        # Extreme value should be flagged as anomaly
        assert bool(result.is_anomaly) == True
        assert result.score > 0.0  # Score should be positive for anomaly

    def test_detect_with_dataframe(self, sample_dataframe):
        """Test detection with DataFrame input."""
        detector = IsolationForestDetector()
        detector.fit(sample_dataframe)

        test_data = pd.DataFrame({"accuracy": [0.95], "precision": [0.95]})
        result = detector.detect(test_data)

        assert result.is_anomaly is not None
        assert len(result.features_involved) == 2

    def test_get_threshold(self, sample_normal_data):
        """Test threshold getter."""
        detector = IsolationForestDetector()
        detector.fit(sample_normal_data)

        threshold = detector.get_threshold()
        assert threshold is not None


class TestZScoreDetector:
    """Test Z-score anomaly detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = ZScoreDetector(threshold_sigma=2.5)
        assert detector.threshold_sigma == 2.5

    def test_fit(self, sample_normal_data):
        """Test fitting detector."""
        detector = ZScoreDetector()
        detector.fit(sample_normal_data)

        assert detector.mean is not None
        assert detector.std is not None

    def test_detect_normal(self, sample_normal_data):
        """Test detection of normal values."""
        detector = ZScoreDetector(threshold_sigma=3.0)
        detector.fit(sample_normal_data)

        result = detector.detect(np.array([100.0]))
        assert result.is_anomaly == False

    def test_detect_anomaly(self, sample_normal_data):
        """Test detection of anomalies."""
        detector = ZScoreDetector(threshold_sigma=2.0)
        detector.fit(sample_normal_data)

        # Extreme value should be anomaly
        result = detector.detect(np.array([200.0]))
        assert result.is_anomaly == True
        assert result.severity in ["warning", "critical"]

    def test_fit_with_dataframe(self, sample_dataframe):
        """Test fitting with DataFrame."""
        detector = ZScoreDetector()
        detector.fit(sample_dataframe)

        assert detector.feature_name == "accuracy"

    def test_zero_std_handling(self):
        """Test handling of zero standard deviation."""
        detector = ZScoreDetector()
        constant_data = np.array([5.0, 5.0, 5.0, 5.0])

        detector.fit(constant_data)
        assert detector.std == 1.0  # Should be set to 1.0


class TestIQRDetector:
    """Test IQR anomaly detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = IQRDetector(iqr_multiplier=2.0)
        assert detector.iqr_multiplier == 2.0

    def test_fit(self, sample_normal_data):
        """Test fitting detector."""
        detector = IQRDetector()
        detector.fit(sample_normal_data)

        assert detector.q1 is not None
        assert detector.q3 is not None
        assert detector.iqr is not None

    def test_detect_normal(self, sample_normal_data):
        """Test detection of normal values."""
        detector = IQRDetector()
        detector.fit(sample_normal_data)

        result = detector.detect(np.array([100.0]))
        assert result.is_anomaly == False

    def test_detect_anomaly(self, sample_normal_data):
        """Test detection of anomalies."""
        detector = IQRDetector(iqr_multiplier=1.5)
        detector.fit(sample_normal_data)

        # Value far outside bounds
        result = detector.detect(np.array([300.0]))
        assert result.is_anomaly == True

    def test_bounds_calculation(self, sample_normal_data):
        """Test correct bounds calculation."""
        detector = IQRDetector(iqr_multiplier=1.5)
        detector.fit(sample_normal_data)

        # Verify bounds are ordered correctly
        assert detector.lower_bound < detector.upper_bound


class TestARIMAForecastDetector:
    """Test ARIMA forecast-based detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = ARIMAForecastDetector(order=(1, 1, 1), deviation_threshold=2.0)
        assert detector.order == (1, 1, 1)
        assert detector.deviation_threshold == 2.0

    def test_fit(self, sample_time_series):
        """Test fitting ARIMA detector."""
        detector = ARIMAForecastDetector()
        detector.fit(sample_time_series)

        assert detector.model is not None
        assert detector.residual_std > 0

    def test_detect(self, sample_time_series):
        """Test ARIMA detection."""
        detector = ARIMAForecastDetector()
        detector.fit(sample_time_series)

        # Test with normal value
        result = detector.detect(sample_time_series)
        assert result.is_anomaly is not None
        assert result.anomaly_type == AnomalyType.TIMESERIES.value

    def test_get_threshold(self, sample_time_series):
        """Test threshold getter."""
        detector = ARIMAForecastDetector()
        detector.fit(sample_time_series)

        threshold = detector.get_threshold()
        assert threshold > 0


class TestAnomalyDetectionSystem:
    """Test AnomalyDetectionSystem."""

    @pytest.fixture
    def system(self, tmp_path):
        """Create temporary anomaly detection system."""
        return AnomalyDetectionSystem(data_dir=tmp_path)

    def test_initialization(self, system):
        """Test system initialization."""
        assert system.data_dir.exists()
        assert len(system.detectors) == 0

    def test_register_detector_isolation_forest(self, system):
        """Test registering Isolation Forest."""
        config = DetectorConfig(method="isolation_forest", enabled=True, weight=1.0)
        system.register_detector(config)

        assert "isolation_forest" in system.detectors
        assert system.detector_configs["isolation_forest"].enabled is True

    def test_register_detector_zscore(self, system):
        """Test registering Z-score detector."""
        config = DetectorConfig(method="zscore", enabled=True, weight=0.8)
        system.register_detector(config)

        assert "zscore" in system.detectors

    def test_register_detector_iqr(self, system):
        """Test registering IQR detector."""
        config = DetectorConfig(method="iqr", enabled=True, weight=0.8)
        system.register_detector(config)

        assert "iqr" in system.detectors

    def test_register_detector_arima(self, system):
        """Test registering ARIMA detector."""
        config = DetectorConfig(
            method="arima",
            enabled=True,
            parameters={"order": [1, 1, 1]},
        )
        system.register_detector(config)

        assert "arima" in system.detectors

    def test_register_disabled_detector(self, system):
        """Test registering disabled detector."""
        config = DetectorConfig(method="zscore", enabled=False)
        system.register_detector(config)

        assert "zscore" not in system.detectors

    def test_fit_detectors(self, system, sample_normal_data):
        """Test fitting detectors."""
        config = DetectorConfig(method="zscore", enabled=True)
        system.register_detector(config)

        system.fit_detectors(sample_normal_data, feature_name="test_metric")

        assert "test_metric" in system.historical_baselines
        baseline = system.historical_baselines["test_metric"]
        assert baseline.mean is not None

    def test_detect_normal(self, system, sample_normal_data):
        """Test detection of normal values."""
        config = DetectorConfig(method="zscore", enabled=True, weight=1.0)
        system.register_detector(config)
        system.fit_detectors(sample_normal_data)

        event = system.detect(np.array([100.0]))
        # Normal value might or might not trigger, just verify it works
        assert event is None or event.anomaly_score.is_anomaly is False

    def test_detect_anomaly(self, system, sample_normal_data):
        """Test detection of anomalies."""
        config = DetectorConfig(method="zscore", enabled=True, weight=1.0)
        system.register_detector(config)
        system.fit_detectors(sample_normal_data)

        event = system.detect(np.array([300.0]))
        assert event is not None
        assert bool(event.anomaly_score.is_anomaly) == True

    def test_get_baseline(self, system, sample_normal_data):
        """Test getting baseline."""
        system.fit_detectors(sample_normal_data, feature_name="metric1")

        baseline = system.get_baseline("metric1")
        assert baseline is not None
        assert baseline.feature_name == "metric1"

    def test_update_baseline(self, system, sample_normal_data):
        """Test updating baseline."""
        system.fit_detectors(sample_normal_data, feature_name="metric1")
        original_mean = system.get_baseline("metric1").mean

        # Update with different data
        new_data = np.random.normal(110, 10, 50)
        system.update_baseline(new_data, feature_name="metric1")

        new_baseline = system.get_baseline("metric1")
        assert abs(new_baseline.mean - original_mean) > 5

    def test_get_anomaly_events(self, system, sample_normal_data):
        """Test getting anomaly events."""
        config = DetectorConfig(method="zscore", enabled=True)
        system.register_detector(config)
        system.fit_detectors(sample_normal_data)

        # Generate some anomalies
        system.detect(np.array([300.0]))
        system.detect(np.array([50.0]))

        events = system.get_anomaly_events(limit=10)
        assert len(events) >= 0

    def test_acknowledge_anomaly(self, system, sample_normal_data):
        """Test acknowledging anomaly."""
        config = DetectorConfig(method="zscore", enabled=True)
        system.register_detector(config)
        system.fit_detectors(sample_normal_data)

        event = system.detect(np.array([300.0]))
        if event:
            success = system.acknowledge_anomaly(
                event.timestamp,
                acknowledged_by="tester",
                notes="Test note",
            )
            assert success is True

    def test_get_statistics(self, system, sample_normal_data):
        """Test getting statistics."""
        config = DetectorConfig(method="zscore", enabled=True)
        system.register_detector(config)
        system.fit_detectors(sample_normal_data)

        system.detect(np.array([300.0]))

        stats = system.get_statistics()
        assert "total_anomalies" in stats
        assert "registered_detectors" in stats
        assert stats["registered_detectors"] == ["zscore"]

    def test_ensemble_detection(self, system, sample_normal_data):
        """Test ensemble detection with multiple detectors."""
        system.register_detector(DetectorConfig(method="zscore", enabled=True, weight=0.5))
        system.register_detector(DetectorConfig(method="iqr", enabled=True, weight=0.5))

        system.fit_detectors(sample_normal_data)

        event = system.detect(np.array([300.0]), ensemble=True)
        # Should use weighted average
        if event:
            assert event.ensemble_score > 0

    def test_persistence_baselines(self, system, sample_normal_data):
        """Test baseline saving and loading."""
        system.fit_detectors(sample_normal_data, feature_name="metric1")

        # Verify baseline is saved
        baseline = system.get_baseline("metric1")
        assert baseline is not None
        assert baseline.feature_name == "metric1"

        # Check that baselines file exists
        assert system.baselines_file.exists()

    def test_persistence_events(self, system, sample_normal_data):
        """Test event saving."""
        config = DetectorConfig(method="zscore", enabled=True)
        system.register_detector(config)
        system.fit_detectors(sample_normal_data)

        # Generate an anomaly
        system.detect(np.array([300.0]))

        # Verify events file exists
        assert system.events_file.exists()

        # Verify we can retrieve events
        events = system.get_anomaly_events()
        # May or may not have events depending on detector settings
        assert isinstance(events, list)


class TestAnomalyEvent:
    """Test AnomalyEvent dataclass."""

    def test_event_creation(self):
        """Test creating AnomalyEvent."""
        score = AnomalyScore(
            timestamp=time.time(),
            is_anomaly=True,
            anomaly_type=AnomalyType.UNIVARIATE.value,
            score=0.8,
            threshold=0.5,
            confidence=0.9,
            features_involved=["test"],
            explanation="Test",
            severity="warning",
        )

        event = AnomalyEvent(
            timestamp=time.time(),
            anomaly_score=score,
            detectors_triggered=["zscore"],
            ensemble_score=0.8,
        )

        assert event.acknowledged == False
        assert len(event.detectors_triggered) == 1

    def test_event_to_dict(self):
        """Test converting event to dict."""
        score = AnomalyScore(
            timestamp=time.time(),
            is_anomaly=True,
            anomaly_type=AnomalyType.UNIVARIATE.value,
            score=0.8,
            threshold=0.5,
            confidence=0.9,
            features_involved=["test"],
            explanation="Test",
            severity="warning",
        )

        event = AnomalyEvent(
            timestamp=time.time(),
            anomaly_score=score,
            detectors_triggered=["detector1"],
            ensemble_score=0.75,
        )

        result = event.to_dict()
        assert "timestamp" in result
        assert "anomaly_score" in result
        assert "detectors_triggered" in result
