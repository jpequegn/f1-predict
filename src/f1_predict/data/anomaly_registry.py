"""Anomaly detection registry for persistence and querying."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class AnomalyRecord:
    """Record of a detected anomaly with context and severity."""

    season: int
    race_round: int
    driver_id: int
    driver_name: str
    anomaly_type: str
    anomaly_score: float
    severity: str
    explanation: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary for serialization.

        Returns:
            Dictionary representation of the anomaly record
        """
        data = asdict(self)
        # Convert datetime objects to ISO format strings for JSON serialization
        if isinstance(data["timestamp"], datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnomalyRecord":
        """Create record from dictionary.

        Args:
            data: Dictionary representation of anomaly record

        Returns:
            AnomalyRecord instance
        """
        # Handle timestamp string parsing
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class AnomalyRegistry:
    """Registry for managing anomaly detection records with persistence.

    Maintains a collection of detected anomalies with support for filtering,
    persistence to JSON files, and statistical summaries.

    Example:
        >>> registry = AnomalyRegistry(storage_dir="data/anomalies")
        >>> record = AnomalyRecord(season=2024, race_round=1, ...)
        >>> registry.add_anomaly(record)
        >>> critical = registry.get_anomalies(severity="critical")
    """

    def __init__(self, storage_dir: str, fail_on_error: bool = False) -> None:
        """Initialize anomaly registry.

        Args:
            storage_dir: Directory path for storing anomaly data
            fail_on_error: If True, raise exceptions on save/load errors.
                          If False, log errors and continue gracefully.
        """
        self.storage_dir = Path(storage_dir)
        self.anomalies: list[AnomalyRecord] = []
        self.logger = logger.bind(component="anomaly_registry")
        self.fail_on_error = fail_on_error

        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def add_anomaly(self, record: AnomalyRecord) -> None:
        """Add an anomaly record to the registry.

        Args:
            record: AnomalyRecord instance to add

        Raises:
            TypeError: If record is not an AnomalyRecord instance
            ValueError: If anomaly_score is not in valid range [0, 1]
        """
        if not isinstance(record, AnomalyRecord):
            msg = f"Expected AnomalyRecord, got {type(record).__name__}"  # type: ignore[unreachable]
            raise TypeError(msg)

        # Type guard: isinstance check above ensures record is AnomalyRecord
        if not 0.0 <= record.anomaly_score <= 1.0:
            msg = f"Anomaly score must be in [0, 1], got {record.anomaly_score}"
            raise ValueError(msg)

        self.anomalies.append(record)
        try:
            self.logger.debug(
                "anomaly_added",
                season=record.season,
                driver_id=record.driver_id,
                severity=record.severity,
                total_anomalies=len(self.anomalies),
            )
        except Exception:  # noqa: BLE001
            # Gracefully handle logging failures in test environments
            pass

    def get_anomalies(
        self,
        season: Optional[int] = None,
        driver_id: Optional[int] = None,
        severity: Optional[str] = None,
    ) -> list[AnomalyRecord]:
        """Retrieve anomalies with optional filtering.

        Args:
            season: Filter by F1 season (optional)
            driver_id: Filter by driver ID (optional)
            severity: Filter by severity level (optional)

        Returns:
            List of matching AnomalyRecord instances
        """
        results = self.anomalies

        if season is not None:
            results = [r for r in results if r.season == season]

        if driver_id is not None:
            results = [r for r in results if r.driver_id == driver_id]

        if severity is not None:
            results = [r for r in results if r.severity == severity]

        return results

    def get_summary(self) -> dict[str, Any]:
        """Generate summary statistics for anomalies.

        Returns:
            Dictionary with summary statistics including:
            - total_anomalies: Total count of anomalies
            - by_severity: Count breakdown by severity level
            - by_type: Count breakdown by anomaly type
            - by_driver: Count breakdown by driver ID
        """
        summary: dict[str, Any] = {
            "total_anomalies": len(self.anomalies),
            "by_severity": {},
            "by_type": {},
            "by_driver": {},
        }

        for record in self.anomalies:
            # Count by severity
            if record.severity not in summary["by_severity"]:
                summary["by_severity"][record.severity] = 0
            summary["by_severity"][record.severity] += 1

            # Count by type
            if record.anomaly_type not in summary["by_type"]:
                summary["by_type"][record.anomaly_type] = 0
            summary["by_type"][record.anomaly_type] += 1

            # Count by driver
            if record.driver_id not in summary["by_driver"]:
                summary["by_driver"][record.driver_id] = 0
            summary["by_driver"][record.driver_id] += 1

        try:
            self.logger.debug(
                "summary_generated",
                total_anomalies=summary["total_anomalies"],
                severity_levels=len(summary["by_severity"]),
            )
        except Exception:  # noqa: BLE001
            # Gracefully handle logging failures in test environments
            pass

        return summary

    def save(self) -> None:
        """Save anomaly registry to JSON file.

        Persists all anomalies to a JSON file in the storage directory.
        Uses atomic writes to prevent file corruption on write failures.

        Raises:
            Exception: If fail_on_error is True and write fails
        """
        try:
            from tempfile import NamedTemporaryFile

            output_file = self.storage_dir / "anomalies.json"

            # Convert records to dictionaries for JSON serialization
            data = {
                "anomalies": [r.to_dict() for r in self.anomalies],
                "saved_at": datetime.now(tz=timezone.utc).isoformat(),
                "total": len(self.anomalies),
            }

            # Atomic write: write to temp file first, then rename
            with NamedTemporaryFile(
                mode="w",
                dir=self.storage_dir,
                delete=False,
                suffix=".tmp",
            ) as tmp:
                json.dump(data, tmp, indent=2)
                tmp_path = Path(tmp.name)

            # Atomic rename replaces existing file atomically
            tmp_path.replace(output_file)

            try:
                self.logger.info(
                    "registry_saved",
                    file=str(output_file),
                    total_anomalies=len(self.anomalies),
                )
            except Exception:  # noqa: BLE001
                # Gracefully handle logging failures in test environments
                pass

        except Exception as e:
            try:
                self.logger.error(
                    "error_saving_registry",
                    error=str(e),
                    exc_info=True,
                )
            except Exception:  # noqa: BLE001
                # Gracefully handle logging failures in test environments
                pass
            # Clean up temp file if it exists
            if "tmp_path" in locals():
                tmp_path.unlink(missing_ok=True)
            if self.fail_on_error:
                raise

    def load(self) -> None:
        """Load anomaly registry from JSON file.

        Loads all anomalies from the JSON file in the storage directory.
        If file doesn't exist, registry remains empty. Corrupted individual
        records are skipped with a warning.

        Raises:
            Exception: If fail_on_error is True and any error occurs
        """
        try:
            input_file = self.storage_dir / "anomalies.json"

            if not input_file.exists():
                try:
                    self.logger.debug("registry_file_not_found", file=str(input_file))
                except Exception:  # noqa: BLE001
                    # Gracefully handle logging failures in test environments
                    pass
                return

            with open(input_file) as f:
                data = json.load(f)

            # Load anomalies with per-record error handling
            loaded: list[AnomalyRecord] = []
            skipped = 0
            for record_data in data.get("anomalies", []):
                try:
                    loaded.append(AnomalyRecord.from_dict(record_data))
                except Exception as e:  # noqa: BLE001
                    skipped += 1
                    try:
                        self.logger.warning(
                            "skipping_invalid_record",
                            record=record_data,
                            error=str(e),
                        )
                    except Exception:  # noqa: BLE001
                        # Gracefully handle logging failures in test environments
                        pass

            self.anomalies = loaded

            try:
                self.logger.info(
                    "registry_loaded",
                    file=str(input_file),
                    total_anomalies=len(self.anomalies),
                    skipped_records=skipped,
                )
            except Exception:  # noqa: BLE001
                # Gracefully handle logging failures in test environments
                pass

        except Exception as e:
            try:
                self.logger.error(
                    "error_loading_registry",
                    error=str(e),
                    exc_info=True,
                )
            except Exception:  # noqa: BLE001
                # Gracefully handle logging failures in test environments
                pass
            if self.fail_on_error:
                raise

    def clear(self) -> None:
        """Clear all anomalies from the registry.

        Removes all anomaly records from memory (does not delete persisted files).
        """
        self.anomalies.clear()
        try:
            self.logger.debug("registry_cleared")
        except Exception:  # noqa: BLE001
            # Gracefully handle logging failures in test environments
            pass
