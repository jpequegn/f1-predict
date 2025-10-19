"""Performance monitoring and metrics tracking utilities.

Provides:
- Operation timing and performance tracking
- Cache hit rate monitoring
- Memory usage monitoring
- Performance metrics aggregation
- Performance alerts and warnings
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

try:
    import streamlit as st
except ImportError:
    st = None


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    operation_name: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)
    status: str = "success"
    error: Optional[str] = None

    def is_slow(self, threshold_ms: float = 1000.0) -> bool:
        """Check if operation exceeded threshold."""
        return self.duration_ms > threshold_ms


class PerformanceTracker:
    """Tracks performance metrics for operations."""

    def __init__(self, max_metrics: int = 1000):
        """Initialize performance tracker.

        Args:
            max_metrics: Maximum metrics to store
        """
        self.max_metrics = max_metrics
        self.metrics: list[PerformanceMetrics] = []
        self.logger = logger.bind(component="performance_tracker")

    def record(self, metric: PerformanceMetrics) -> None:
        """Record a performance metric.

        Args:
            metric: Performance metric to record
        """
        self.metrics.append(metric)

        # Keep list size bounded
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics :]

    def get_average_duration(self, operation_name: str) -> Optional[float]:
        """Get average duration for operation.

        Args:
            operation_name: Name of operation

        Returns:
            Average duration in milliseconds or None
        """
        durations = [
            m.duration_ms
            for m in self.metrics
            if m.operation_name == operation_name and m.status == "success"
        ]

        return sum(durations) / len(durations) if durations else None

    def get_success_rate(self, operation_name: str) -> float:
        """Get success rate for operation.

        Args:
            operation_name: Name of operation

        Returns:
            Success rate as percentage (0.0-100.0)
        """
        operation_metrics = [m for m in self.metrics if m.operation_name == operation_name]

        if not operation_metrics:
            return 100.0

        successes = len([m for m in operation_metrics if m.status == "success"])
        return (successes / len(operation_metrics)) * 100

    def get_slow_operations(self, threshold_ms: float = 1000.0) -> list[PerformanceMetrics]:
        """Get operations that exceeded threshold.

        Args:
            threshold_ms: Threshold in milliseconds

        Returns:
            List of slow operations
        """
        return [m for m in self.metrics if m.is_slow(threshold_ms)]

    def get_stats(self) -> dict[str, Any]:
        """Get overall performance statistics.

        Returns:
            Dictionary with performance stats
        """
        if not self.metrics:
            return {"total_operations": 0}

        total_time = sum(m.duration_ms for m in self.metrics)
        slow_count = len(self.get_slow_operations())

        return {
            "total_operations": len(self.metrics),
            "total_time_ms": total_time,
            "average_duration_ms": total_time / len(self.metrics),
            "slow_operations_count": slow_count,
            "slow_operations_percentage": (slow_count / len(self.metrics)) * 100,
            "success_rate": (
                len([m for m in self.metrics if m.status == "success"]) / len(self.metrics)
            )
            * 100,
        }


@contextmanager
def track_performance(operation_name: str, threshold_ms: float = 1000.0):
    """Context manager to track operation performance.

    Args:
        operation_name: Name of operation
        threshold_ms: Warning threshold in milliseconds

    Yields:
        Allows operation to execute

    Example:
        with track_performance("load_data"):
            data = load_race_data()
    """
    start_time = time.time()
    metric = None

    try:
        yield
        duration_ms = (time.time() - start_time) * 1000
        metric = PerformanceMetrics(
            operation_name=operation_name,
            duration_ms=duration_ms,
            status="success",
        )

        if duration_ms > threshold_ms:
            logger.warning(
                f"Slow operation detected: {operation_name}",
                duration_ms=duration_ms,
                threshold_ms=threshold_ms,
            )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        metric = PerformanceMetrics(
            operation_name=operation_name,
            duration_ms=duration_ms,
            status="error",
            error=str(e),
        )
        logger.error(
            f"Operation failed: {operation_name}",
            error=str(e),
            duration_ms=duration_ms,
        )
        raise

    finally:
        if metric and st:
            # Store in session state if available
            if "performance_metrics" not in st.session_state:
                st.session_state.performance_metrics = []

            st.session_state.performance_metrics.append(metric)


class CacheMonitor:
    """Monitors cache hit rates."""

    def __init__(self):
        """Initialize cache monitor."""
        self.hits = 0
        self.misses = 0
        self.logger = logger.bind(component="cache_monitor")

    def record_hit(self) -> None:
        """Record cache hit."""
        self.hits += 1

    def record_miss(self) -> None:
        """Record cache miss."""
        self.misses += 1

    def get_hit_rate(self) -> float:
        """Get cache hit rate as percentage.

        Returns:
            Hit rate (0.0-100.0)
        """
        total = self.hits + self.misses

        if total == 0:
            return 0.0

        return (self.hits / total) * 100

    def reset(self) -> None:
        """Reset counters."""
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_accesses": self.hits + self.misses,
            "hit_rate": self.get_hit_rate(),
        }


def get_performance_tracker() -> PerformanceTracker:
    """Get or create performance tracker in session state.

    Returns:
        Performance tracker instance
    """
    if "performance_tracker" not in st.session_state:
        st.session_state.performance_tracker = PerformanceTracker()

    return st.session_state.performance_tracker


def get_cache_monitor() -> CacheMonitor:
    """Get or create cache monitor in session state.

    Returns:
        Cache monitor instance
    """
    if "cache_monitor" not in st.session_state:
        st.session_state.cache_monitor = CacheMonitor()

    return st.session_state.cache_monitor


def display_performance_metrics(tracker: PerformanceTracker) -> None:
    """Display performance metrics in UI.

    Args:
        tracker: Performance tracker instance
    """
    if not st:
        return

    stats = tracker.get_stats()

    if stats["total_operations"] == 0:
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Operations",
            stats["total_operations"],
        )

    with col2:
        st.metric(
            "Avg Duration",
            f"{stats['average_duration_ms']:.0f}ms",
        )

    with col3:
        st.metric(
            "Slow Ops",
            f"{stats['slow_operations_percentage']:.1f}%",
        )

    with col4:
        st.metric(
            "Success Rate",
            f"{stats['success_rate']:.1f}%",
        )


def display_cache_metrics(monitor: CacheMonitor) -> None:
    """Display cache metrics in UI.

    Args:
        monitor: Cache monitor instance
    """
    if not st:
        return

    stats = monitor.get_stats()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Cache Hits", stats["hits"])

    with col2:
        st.metric("Cache Misses", stats["misses"])

    with col3:
        st.metric("Hit Rate", f"{stats['hit_rate']:.1f}%")
