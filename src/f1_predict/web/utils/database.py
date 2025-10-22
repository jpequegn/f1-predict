"""Database configuration and session management for monitoring system.

Provides connection pooling, session factories, and configuration management
for both PostgreSQL (production) and SQLite (development/testing).
"""

from collections.abc import Generator
from contextlib import contextmanager
import os

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool
import structlog

logger = structlog.get_logger(__name__)


class DatabaseConfig:
    """Database configuration from environment variables."""

    def __init__(self) -> None:
        """Initialize database configuration."""
        self.db_type = os.getenv("MONITORING_DB_TYPE", "sqlite").lower()
        self.enabled = os.getenv("MONITORING_DB_ENABLED", "false").lower() == "true"

        if self.db_type == "postgresql":
            self.host = os.getenv("MONITORING_DB_HOST", "localhost")
            self.port = int(os.getenv("MONITORING_DB_PORT", "5432"))
            self.user = os.getenv("MONITORING_DB_USER", "postgres")
            self.password = os.getenv("MONITORING_DB_PASSWORD", "")
            self.database = os.getenv("MONITORING_DB_NAME", "f1_predict_monitoring")
            self.url = (
                f"postgresql+psycopg2://{self.user}:{self.password}@"
                f"{self.host}:{self.port}/{self.database}"
            )
        else:  # sqlite
            db_path = os.getenv(
                "MONITORING_DB_PATH", "data/monitoring/monitoring.db"
            )
            self.url = f"sqlite:///{db_path}"

        self.pool_size = int(os.getenv("MONITORING_DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("MONITORING_DB_MAX_OVERFLOW", "20"))
        self.echo = os.getenv("MONITORING_DB_ECHO", "false").lower() == "true"


class DatabaseManager:
    """Manages database connections and session lifecycle."""

    _instance = None
    _config = None
    _engine = None
    _session_factory = None

    def __new__(cls) -> "DatabaseManager":
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, config: DatabaseConfig | None = None) -> None:
        """Initialize database manager with configuration.

        Args:
            config: DatabaseConfig instance. If None, creates from environment.
        """
        if cls._config is not None:
            logger.info("database_already_initialized")
            return

        cls._config = config or DatabaseConfig()

        if not cls._config.enabled:
            logger.info("database_disabled", db_type=cls._config.db_type)
            return

        # Create engine with appropriate pool configuration
        if cls._config.db_type == "postgresql":
            engine_kwargs = {
                "poolclass": QueuePool,
                "pool_size": cls._config.pool_size,
                "max_overflow": cls._config.max_overflow,
                "pool_pre_ping": True,  # Verify connections before using
            }
        else:  # SQLite
            engine_kwargs = {
                "poolclass": NullPool,  # SQLite doesn't benefit from pooling
            }

        cls._engine = create_engine(
            cls._config.url,
            echo=cls._config.echo,
            **engine_kwargs,
        )

        # Set up SQLite pragma for concurrent access
        if cls._config.db_type == "sqlite":

            @event.listens_for(Engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):  # noqa: ARG001
                """Enable write-ahead logging for SQLite."""
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()

        # Create session factory
        cls._session_factory = sessionmaker(
            bind=cls._engine,
            expire_on_commit=False,
        )

        logger.info(
            "database_initialized",
            db_type=cls._config.db_type,
            url=cls._config.url.split("@")[-1] if "@" in cls._config.url else cls._config.url,
        )

    @classmethod
    def get_engine(cls) -> Engine:
        """Get the SQLAlchemy engine.

        Returns:
            SQLAlchemy Engine instance.

        Raises:
            RuntimeError: If database not initialized.
        """
        if cls._engine is None:
            raise RuntimeError(
                "Database not initialized. Call DatabaseManager.initialize() first."
            )
        return cls._engine

    @classmethod
    def get_session(cls) -> Session:
        """Create a new database session.

        Returns:
            SQLAlchemy Session instance.

        Raises:
            RuntimeError: If database not initialized.
        """
        if cls._session_factory is None:
            raise RuntimeError(
                "Database not initialized. Call DatabaseManager.initialize() first."
            )
        return cls._session_factory()

    @classmethod
    @contextmanager
    def session_scope(cls) -> Generator[Session, None, None]:
        """Context manager for database sessions.

        Ensures proper session cleanup and transaction handling.

        Yields:
            SQLAlchemy Session instance.

        Example:
            with DatabaseManager.session_scope() as session:
                result = session.query(Model).first()
        """
        session = cls.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("session_error", error=str(e))
            raise
        finally:
            session.close()

    @classmethod
    def health_check(cls) -> bool:
        """Check database connectivity.

        Returns:
            True if database is healthy, False otherwise.
        """
        try:
            with cls.session_scope() as session:
                session.execute("SELECT 1")
            logger.info("database_health_check_passed")
            return True
        except Exception as e:
            logger.error("database_health_check_failed", error=str(e))
            return False

    @classmethod
    def create_all_tables(cls) -> None:
        """Create all database tables.

        Must import all models before calling this method.
        """
        if cls._engine is None:
            raise RuntimeError(
                "Database not initialized. Call DatabaseManager.initialize() first."
            )

        # Import here to avoid circular imports
        from f1_predict.web.utils.database_models import Base  # noqa: F401

        Base.metadata.create_all(cls._engine)
        logger.info("database_tables_created")

    @classmethod
    def drop_all_tables(cls) -> None:
        """Drop all database tables.

        WARNING: This is destructive. Use only in testing.
        """
        if cls._engine is None:
            raise RuntimeError(
                "Database not initialized. Call DatabaseManager.initialize() first."
            )

        # Import here to avoid circular imports
        from f1_predict.web.utils.database_models import Base  # noqa: F401

        Base.metadata.drop_all(cls._engine)
        logger.info("database_tables_dropped")

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if database is enabled.

        Returns:
            True if database is enabled via configuration.
        """
        return cls._config is not None and cls._config.enabled
