"""Factory for monitoring system implementations.

Provides transparent switching between file-based and database-backed
monitoring systems based on environment configuration.
"""

import os
from pathlib import Path
from typing import Any, Union

import structlog

logger = structlog.get_logger(__name__)


def get_performance_tracker(
    data_dir: Union[Path, str] = "data/monitoring",
) -> Any:
    """Get performance tracker instance.

    Returns database-backed tracker if MONITORING_DB_ENABLED=true,
    otherwise returns file-based tracker.

    Args:
        data_dir: Directory for file-based storage (ignored if database backend)

    Returns:
        ModelPerformanceTracker or ModelPerformanceTrackerDB instance
    """
    db_enabled = os.getenv("MONITORING_DB_ENABLED", "false").lower() == "true"

    if db_enabled:
        try:
            from f1_predict.web.utils.monitoring_database import (
                ModelPerformanceTrackerDB,
            )

            logger.info("performance_tracker_backend", backend="database")
            return ModelPerformanceTrackerDB(data_dir=data_dir)
        except Exception as e:
            logger.warning(
                "database_backend_initialization_failed",
                error=str(e),
                fallback="file-based",
            )
            # Fall through to file-based
    else:
        logger.info("performance_tracker_backend", backend="file-based")

    # Fall back to file-based implementation
    from f1_predict.web.utils.monitoring import ModelPerformanceTracker

    return ModelPerformanceTracker(data_dir=data_dir)


def get_alerting_system(
    data_dir: Union[Path, str] = "data/monitoring",
    channel_config: Any = None,
) -> Any:
    """Get alerting system instance.

    Returns database-backed alerting system if MONITORING_DB_ENABLED=true,
    otherwise returns file-based alerting system.

    Args:
        data_dir: Directory for file-based storage (ignored if database backend)
        channel_config: Alert channel configuration

    Returns:
        AlertingSystem or AlertingSystemDB instance
    """
    db_enabled = os.getenv("MONITORING_DB_ENABLED", "false").lower() == "true"

    if db_enabled:
        try:
            from f1_predict.web.utils.alerting_database import AlertingSystemDB

            logger.info("alerting_system_backend", backend="database")
            return AlertingSystemDB(data_dir=data_dir, channel_config=channel_config)
        except Exception as e:
            logger.warning(
                "database_backend_initialization_failed",
                error=str(e),
                fallback="file-based",
            )
            # Fall through to file-based
    else:
        logger.info("alerting_system_backend", backend="file-based")

    # Fall back to file-based implementation
    from f1_predict.web.utils.alerting import AlertingSystem

    return AlertingSystem(data_dir=data_dir, channel_config=channel_config)


def is_database_backend_enabled() -> bool:
    """Check if database backend is enabled.

    Returns:
        True if MONITORING_DB_ENABLED=true, False otherwise
    """
    return os.getenv("MONITORING_DB_ENABLED", "false").lower() == "true"
