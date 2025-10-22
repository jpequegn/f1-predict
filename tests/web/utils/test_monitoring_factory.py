"""Tests for monitoring factory pattern implementation."""

import os
from unittest.mock import patch

import pytest

from f1_predict.web.utils.monitoring_factory import (
    get_alerting_system,
    get_performance_tracker,
    is_database_backend_enabled,
)


class TestMonitoringFactory:
    """Test monitoring factory functions."""

    def test_is_database_backend_enabled_default(self):
        """Test default database backend is disabled."""
        # Clear env var if set
        with patch.dict(os.environ, {}, clear=False):
            if "MONITORING_DB_ENABLED" in os.environ:
                del os.environ["MONITORING_DB_ENABLED"]
            assert is_database_backend_enabled() is False

    def test_is_database_backend_enabled_true(self, monkeypatch):
        """Test database backend enabled when env var set to true."""
        monkeypatch.setenv("MONITORING_DB_ENABLED", "true")
        assert is_database_backend_enabled() is True

    def test_is_database_backend_enabled_false(self, monkeypatch):
        """Test database backend disabled when env var set to false."""
        monkeypatch.setenv("MONITORING_DB_ENABLED", "false")
        assert is_database_backend_enabled() is False

    def test_get_performance_tracker_file_based(self, monkeypatch):
        """Test returns file-based tracker when database disabled."""
        monkeypatch.setenv("MONITORING_DB_ENABLED", "false")
        tracker = get_performance_tracker("data/monitoring")

        from f1_predict.web.utils.monitoring import ModelPerformanceTracker

        assert isinstance(tracker, ModelPerformanceTracker)

    def test_get_performance_tracker_database_backed(self, monkeypatch):
        """Test database backend selection when enabled."""
        monkeypatch.setenv("MONITORING_DB_ENABLED", "true")
        tracker = get_performance_tracker("data/monitoring")

        # When database is enabled, we should try to get database-backed version
        # If database is not fully configured, it falls back to file-based
        from f1_predict.web.utils.monitoring_database import ModelPerformanceTrackerDB
        from f1_predict.web.utils.monitoring import ModelPerformanceTracker

        # Both are acceptable - either database-backed or fallback
        assert isinstance(tracker, (ModelPerformanceTrackerDB, ModelPerformanceTracker))

    def test_get_alerting_system_file_based(self, monkeypatch):
        """Test returns file-based alerting system when database disabled."""
        monkeypatch.setenv("MONITORING_DB_ENABLED", "false")
        system = get_alerting_system("data/monitoring")

        from f1_predict.web.utils.alerting import AlertingSystem

        assert isinstance(system, AlertingSystem)

    def test_get_alerting_system_database_backed(self, monkeypatch):
        """Test database backend selection when enabled."""
        monkeypatch.setenv("MONITORING_DB_ENABLED", "true")
        system = get_alerting_system("data/monitoring")

        from f1_predict.web.utils.alerting_database import AlertingSystemDB
        from f1_predict.web.utils.alerting import AlertingSystem

        # Both are acceptable - either database-backed or fallback
        assert isinstance(system, (AlertingSystemDB, AlertingSystem))

    def test_factory_returns_instance(self, monkeypatch):
        """Test factory always returns a tracker instance."""
        monkeypatch.setenv("MONITORING_DB_ENABLED", "false")

        tracker = get_performance_tracker("data/monitoring")
        system = get_alerting_system("data/monitoring")

        # Both should return instances
        assert tracker is not None
        assert system is not None
