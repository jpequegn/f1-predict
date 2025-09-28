"""Structured logging configuration for F1 Predict.

This module provides structured logging using structlog for better observability
and debugging in development, testing, and production environments.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from structlog.typing import EventDict, Processor


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    enable_colors: bool = True,
) -> None:
    """Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ('json' or 'console')
        log_file: Optional log file path
        enable_colors: Whether to enable colored output for console format
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure structlog processors
    processors = _get_processors(log_format, enable_colors)

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.WriteLoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    _configure_stdlib_logging(numeric_level, log_file)

    # Set up log file directory if needed
    if log_file:
        _ensure_log_directory(log_file)


def _get_processors(log_format: str, enable_colors: bool) -> List[Processor]:
    """Get the appropriate processors for the given format."""
    # Common processors for all formats
    processors: List[Processor] = [
        # Add timestamp
        structlog.processors.TimeStamper(fmt="ISO"),

        # Add logger name and level
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,

        # Add positional arguments
        structlog.stdlib.PositionalArgumentsFormatter(),

        # Process stack info
        structlog.processors.StackInfoRenderer(),

        # Add exception info
        structlog.processors.format_exc_info,

        # Add custom context processors
        _add_process_info,
        _add_correlation_id,
    ]

    if log_format.lower() == "json":
        # JSON format for production/structured logging
        processors.extend([
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ])
    else:
        # Console format for development
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=enable_colors),
        ])

    return processors


def _configure_stdlib_logging(level: int, log_file: Optional[str]) -> None:
    """Configure the standard library logging."""
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove default handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(message)s"  # structlog handles the formatting
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        root_logger.addHandler(file_handler)

    # Configure third-party loggers
    _configure_third_party_loggers()


def _configure_third_party_loggers() -> None:
    """Configure logging levels for third-party libraries."""
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # ML libraries
    logging.getLogger("sklearn").setLevel(logging.WARNING)
    logging.getLogger("xgboost").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)


def _add_process_info(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add process information to log entries."""
    event_dict["process_id"] = os.getpid()
    event_dict["thread_id"] = None  # Could add threading.get_ident() if needed
    return event_dict


def _add_correlation_id(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add correlation ID for request tracing."""
    # In a web application, this would typically come from request context
    # For now, we'll add a placeholder
    if "correlation_id" not in event_dict:
        event_dict["correlation_id"] = getattr(logger, "_correlation_id", None)
    return event_dict


def _ensure_log_directory(log_file: str) -> None:
    """Ensure the log file directory exists."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, **context: Any) -> structlog.BoundLogger:
    """Get a structured logger with optional context.

    Args:
        name: Logger name (typically __name__)
        **context: Additional context to bind to the logger

    Returns:
        Bound structlog logger with context
    """
    logger = structlog.get_logger(name)
    if context:
        logger = logger.bind(**context)
    return logger


def setup_request_logging(correlation_id: str) -> structlog.BoundLogger:
    """Set up logging for a request with correlation ID.

    Args:
        correlation_id: Unique identifier for the request

    Returns:
        Logger bound with request context
    """
    return get_logger("f1_predict.request", correlation_id=correlation_id)


def log_function_call(
    logger: structlog.BoundLogger,
    function_name: str,
    args: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[float] = None,
) -> None:
    """Log a function call with parameters and timing.

    Args:
        logger: Structured logger instance
        function_name: Name of the function being called
        args: Function arguments to log (excluding sensitive data)
        duration_ms: Function execution time in milliseconds
    """
    log_data = {
        "event": "function_call",
        "function": function_name,
    }

    if args:
        # Filter out sensitive information
        safe_args = {k: v for k, v in args.items() if not _is_sensitive_key(k)}
        log_data["args"] = safe_args

    if duration_ms is not None:
        log_data["duration_ms"] = duration_ms

    logger.info("Function executed", **log_data)


def log_ml_metrics(
    logger: structlog.BoundLogger,
    model_name: str,
    metrics: Dict[str, float],
    dataset_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Log machine learning model metrics.

    Args:
        logger: Structured logger instance
        model_name: Name of the ML model
        metrics: Model performance metrics
        dataset_info: Information about the dataset used
    """
    log_data = {
        "event": "ml_metrics",
        "model": model_name,
        "metrics": metrics,
    }

    if dataset_info:
        log_data["dataset"] = dataset_info

    logger.info("ML model metrics", **log_data)


def log_data_quality(
    logger: structlog.BoundLogger,
    dataset_name: str,
    quality_metrics: Dict[str, Any],
    issues: Optional[List[str]] = None,
) -> None:
    """Log data quality assessment results.

    Args:
        logger: Structured logger instance
        dataset_name: Name of the dataset
        quality_metrics: Data quality metrics
        issues: List of data quality issues found
    """
    log_data = {
        "event": "data_quality",
        "dataset": dataset_name,
        "quality_metrics": quality_metrics,
    }

    if issues:
        log_data["issues"] = issues

    logger.info("Data quality assessment", **log_data)


def _is_sensitive_key(key: str) -> bool:
    """Check if a key contains sensitive information."""
    sensitive_patterns = [
        "password", "token", "key", "secret", "credential",
        "auth", "session", "cookie", "private"
    ]
    key_lower = key.lower()
    return any(pattern in key_lower for pattern in sensitive_patterns)


# Performance monitoring decorator
def log_performance(logger: Optional[structlog.BoundLogger] = None):
    """Decorator to log function performance metrics."""
    import functools
    import time

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            func_logger = logger or get_logger(func.__module__)

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000

                log_function_call(
                    func_logger,
                    func.__name__,
                    {"args_count": len(args), "kwargs_count": len(kwargs)},
                    duration_ms
                )

                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                func_logger.error(
                    "Function failed",
                    function=func.__name__,
                    duration_ms=duration_ms,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise

        return wrapper
    return decorator