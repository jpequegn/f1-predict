"""Configuration management for production deployment.

Handles environment-based configuration for different deployment modes.
"""

from dataclasses import dataclass
import os
from typing import Optional


@dataclass
class ServerConfig:
    """Server configuration."""

    host: str = "0.0.0.0"  # noqa: S104 - Intentional for containerized deployment
    port: int = 8501
    enable_cors: bool = False
    enable_xsrf_protection: bool = True
    max_upload_size_mb: int = 200
    enable_websocket_compression: bool = True


@dataclass
class CacheConfig:
    """Cache configuration."""

    ttl_seconds: int = 3600
    max_entries: int = 10000
    enable_redis: bool = False
    redis_url: Optional[str] = None


@dataclass
class PerformanceConfig:
    """Performance configuration."""

    max_workers: int = 4
    enable_metrics: bool = True
    slow_operation_threshold_ms: float = 1000.0
    cache_monitoring_enabled: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_file_logging: bool = False
    log_file: Optional[str] = None


@dataclass
class SecurityConfig:
    """Security configuration."""

    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    require_https: bool = False
    api_key_validation: bool = False


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""

    enable_error_tracking: bool = True
    enable_user_analytics: bool = False
    enable_performance_logging: bool = True
    sentry_dsn: Optional[str] = None


class Config:
    """Main configuration class."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        self.env = os.getenv("APP_ENV", "development")

        self.server = ServerConfig(
            host=os.getenv("SERVER_HOST", "0.0.0.0"),  # noqa: S104 - Intentional for containerized deployment
            port=int(os.getenv("SERVER_PORT", "8501")),
            enable_cors=os.getenv("ENABLE_CORS", "false").lower() == "true",
            max_upload_size_mb=int(
                os.getenv("MAX_UPLOAD_SIZE_MB", "200")
            ),
        )

        self.cache = CacheConfig(
            ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
            max_entries=int(os.getenv("CACHE_MAX_ENTRIES", "10000")),
            enable_redis=os.getenv("ENABLE_REDIS", "false").lower() == "true",
            redis_url=os.getenv("REDIS_URL"),
        )

        self.performance = PerformanceConfig(
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            enable_metrics=os.getenv("ENABLE_METRICS", "true").lower()
            == "true",
            slow_operation_threshold_ms=float(
                os.getenv("SLOW_OPERATION_THRESHOLD_MS", "1000.0")
            ),
        )

        self.logging = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            enable_file_logging=os.getenv("ENABLE_FILE_LOGGING", "false").lower()
            == "true",
            log_file=os.getenv("LOG_FILE"),
        )

        self.security = SecurityConfig(
            enable_rate_limiting=os.getenv("ENABLE_RATE_LIMITING", "true").lower()
            == "true",
            rate_limit_requests=int(
                os.getenv("RATE_LIMIT_REQUESTS", "100")
            ),
            rate_limit_window_seconds=int(
                os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")
            ),
            require_https=os.getenv("REQUIRE_HTTPS", "false").lower()
            == "true",
        )

        self.monitoring = MonitoringConfig(
            enable_error_tracking=os.getenv("ENABLE_ERROR_TRACKING", "true").lower()
            == "true",
            enable_user_analytics=os.getenv("ENABLE_USER_ANALYTICS", "false").lower()
            == "true",
            enable_performance_logging=os.getenv(
                "ENABLE_PERFORMANCE_LOGGING", "true"
            ).lower()
            == "true",
            sentry_dsn=os.getenv("SENTRY_DSN"),
        )

    def is_production(self) -> bool:
        """Check if running in production.

        Returns:
            True if production environment
        """
        return self.env.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development.

        Returns:
            True if development environment
        """
        return self.env.lower() == "development"

    def __repr__(self) -> str:
        """String representation of config."""
        return f"Config(env={self.env})"


def get_config() -> Config:
    """Get global configuration instance.

    Returns:
        Configuration instance
    """
    if not hasattr(get_config, "_instance"):
        get_config._instance = Config()

    return get_config._instance


# Export singleton instance
config = get_config()
