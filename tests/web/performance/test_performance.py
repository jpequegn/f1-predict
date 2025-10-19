"""Performance tests for web interface."""

import time

import pytest

from f1_predict.web.utils.performance import (
    CacheMonitor,
    PerformanceMetrics,
    PerformanceTracker,
    track_performance,
)


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics."""

    def test_create_metric(self):
        """Test creating a performance metric."""
        metric = PerformanceMetrics(
            operation_name="test_op",
            duration_ms=100.0,
        )

        assert metric.operation_name == "test_op"
        assert metric.duration_ms == 100.0
        assert metric.status == "success"

    def test_metric_is_slow(self):
        """Test checking if metric is slow."""
        slow_metric = PerformanceMetrics(
            operation_name="test_op",
            duration_ms=1500.0,
        )
        fast_metric = PerformanceMetrics(
            operation_name="test_op",
            duration_ms=500.0,
        )

        assert slow_metric.is_slow(threshold_ms=1000.0)
        assert not fast_metric.is_slow(threshold_ms=1000.0)


class TestPerformanceTracker:
    """Tests for PerformanceTracker."""

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = PerformanceTracker()

        assert len(tracker.metrics) == 0
        assert tracker.max_metrics == 1000

    def test_record_metric(self):
        """Test recording a metric."""
        tracker = PerformanceTracker()
        metric = PerformanceMetrics(
            operation_name="test_op",
            duration_ms=100.0,
        )

        tracker.record(metric)

        assert len(tracker.metrics) == 1
        assert tracker.metrics[0].operation_name == "test_op"

    def test_get_average_duration(self):
        """Test getting average duration."""
        tracker = PerformanceTracker()

        for i in range(5):
            metric = PerformanceMetrics(
                operation_name="test_op",
                duration_ms=float(100 * (i + 1)),
            )
            tracker.record(metric)

        avg = tracker.get_average_duration("test_op")
        assert avg == 300.0

    def test_get_success_rate(self):
        """Test getting success rate."""
        tracker = PerformanceTracker()

        # Add 3 successful metrics
        for _ in range(3):
            tracker.record(
                PerformanceMetrics(
                    operation_name="test_op",
                    duration_ms=100.0,
                    status="success",
                )
            )

        # Add 1 failed metric
        tracker.record(
            PerformanceMetrics(
                operation_name="test_op",
                duration_ms=100.0,
                status="error",
            )
        )

        rate = tracker.get_success_rate("test_op")
        assert rate == 75.0

    def test_get_slow_operations(self):
        """Test getting slow operations."""
        tracker = PerformanceTracker()

        # Add mix of fast and slow operations
        tracker.record(
            PerformanceMetrics(
                operation_name="test_op",
                duration_ms=500.0,
            )
        )
        tracker.record(
            PerformanceMetrics(
                operation_name="test_op",
                duration_ms=1500.0,
            )
        )
        tracker.record(
            PerformanceMetrics(
                operation_name="test_op",
                duration_ms=2000.0,
            )
        )

        slow = tracker.get_slow_operations(threshold_ms=1000.0)
        assert len(slow) == 2

    def test_get_stats(self):
        """Test getting overall stats."""
        tracker = PerformanceTracker()

        for i in range(3):
            tracker.record(
                PerformanceMetrics(
                    operation_name="test_op",
                    duration_ms=100.0 * (i + 1),
                    status="success",
                )
            )

        stats = tracker.get_stats()

        assert stats["total_operations"] == 3
        assert stats["average_duration_ms"] == 200.0
        assert stats["success_rate"] == 100.0

    def test_metrics_list_bounded(self):
        """Test that metrics list is bounded."""
        tracker = PerformanceTracker(max_metrics=100)

        # Add more than max metrics
        for _ in range(150):
            tracker.record(
                PerformanceMetrics(
                    operation_name="test_op",
                    duration_ms=100.0,
                )
            )

        assert len(tracker.metrics) <= 100


class TestCacheMonitor:
    """Tests for CacheMonitor."""

    def test_cache_monitor_initialization(self):
        """Test cache monitor initialization."""
        monitor = CacheMonitor()

        assert monitor.hits == 0
        assert monitor.misses == 0

    def test_record_cache_hit(self):
        """Test recording cache hit."""
        monitor = CacheMonitor()

        monitor.record_hit()
        monitor.record_hit()

        assert monitor.hits == 2

    def test_record_cache_miss(self):
        """Test recording cache miss."""
        monitor = CacheMonitor()

        monitor.record_miss()
        monitor.record_miss()
        monitor.record_miss()

        assert monitor.misses == 3

    def test_get_hit_rate(self):
        """Test getting hit rate."""
        monitor = CacheMonitor()

        # 3 hits, 1 miss
        for _ in range(3):
            monitor.record_hit()
        monitor.record_miss()

        rate = monitor.get_hit_rate()
        assert rate == 75.0

    def test_hit_rate_empty(self):
        """Test hit rate when empty."""
        monitor = CacheMonitor()

        rate = monitor.get_hit_rate()
        assert rate == 0.0

    def test_reset_cache_monitor(self):
        """Test resetting cache monitor."""
        monitor = CacheMonitor()

        monitor.record_hit()
        monitor.record_miss()
        monitor.reset()

        assert monitor.hits == 0
        assert monitor.misses == 0

    def test_get_cache_stats(self):
        """Test getting cache stats."""
        monitor = CacheMonitor()

        for _ in range(8):
            monitor.record_hit()
        for _ in range(2):
            monitor.record_miss()

        stats = monitor.get_stats()

        assert stats["hits"] == 8
        assert stats["misses"] == 2
        assert stats["total_accesses"] == 10
        assert stats["hit_rate"] == 80.0


class TestPerformanceTracking:
    """Tests for performance tracking context manager."""

    def test_track_performance_success(self):
        """Test tracking successful operation."""
        with track_performance("test_op", threshold_ms=1000.0):
            time.sleep(0.01)

        # Should not raise

    def test_track_performance_slow_operation(self):
        """Test tracking slow operation."""
        with track_performance("slow_op", threshold_ms=50.0):
            time.sleep(0.1)

        # Should log warning but not raise

    def test_track_performance_error(self):
        """Test tracking failed operation."""
        with pytest.raises(ValueError, match="Test error"):  # noqa: SIM117
            with track_performance("error_op", threshold_ms=1000.0):
                raise ValueError("Test error")

    def test_performance_context_manager_timing(self):
        """Test that context manager correctly measures time."""
        start = time.time()

        with track_performance("timed_op", threshold_ms=1000.0):
            time.sleep(0.05)

        elapsed = time.time() - start

        assert elapsed >= 0.05


