"""JSON to Database migration tool for monitoring system.

Migrates existing JSONL files to the new database backend while maintaining
data integrity and allowing rollback if needed.
"""

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import structlog

from f1_predict.web.utils.database import DatabaseManager
from f1_predict.web.utils.database_repositories import (
    AlertRepository,
    AlertRuleRepository,
    HealthSnapshotRepository,
    PredictionRepository,
)

logger = structlog.get_logger(__name__)


class MonitoringMigrationTool:
    """Tool for migrating monitoring data from JSON files to database."""

    def __init__(self, data_dir: Path | str = "data/monitoring") -> None:
        """Initialize migration tool.

        Args:
            data_dir: Directory containing JSONL files to migrate
        """
        self.data_dir = Path(data_dir)
        self.logger = logger.bind(component="migration_tool")

        if not DatabaseManager.is_enabled():
            raise RuntimeError(
                "Database backend not enabled. Set MONITORING_DB_ENABLED=true"
            )

    def migrate_predictions(self, batch_size: int = 1000) -> dict[str, Any]:
        """Migrate predictions from JSONL to database.

        Args:
            batch_size: Number of records to insert per batch

        Returns:
            Migration statistics dictionary
        """
        predictions_file = self.data_dir / "predictions.jsonl"

        if not predictions_file.exists():
            self.logger.info("predictions_file_not_found", path=predictions_file)
            return {"file_found": False, "records_migrated": 0}

        stats = {"records_migrated": 0, "errors": 0, "file_found": True}

        try:
            batch = []

            with open(predictions_file) as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        batch.append(data)

                        # Migrate batch
                        if len(batch) >= batch_size:
                            stats["records_migrated"] += self._insert_prediction_batch(
                                batch
                            )
                            batch = []

                    except json.JSONDecodeError as e:
                        self.logger.warning(
                            "json_parse_error",
                            file=predictions_file,
                            line=line_num,
                            error=str(e),
                        )
                        stats["errors"] += 1

                # Migrate remaining records
                if batch:
                    stats["records_migrated"] += self._insert_prediction_batch(batch)

            self.logger.info(
                "predictions_migration_complete",
                records_migrated=stats["records_migrated"],
                errors=stats["errors"],
            )
            return stats

        except Exception as e:
            self.logger.error("predictions_migration_failed", error=str(e))
            stats["errors"] += 1
            return stats

    def _insert_prediction_batch(self, batch: list[dict[str, Any]]) -> int:
        """Insert a batch of predictions.

        Args:
            batch: List of prediction dictionaries

        Returns:
            Number of successfully inserted records
        """
        try:
            with DatabaseManager.session_scope() as session:
                repo = PredictionRepository(session)

                # Convert ISO format timestamps to datetime
                for pred in batch:
                    if isinstance(pred.get("timestamp"), str):
                        pred["timestamp"] = datetime.fromisoformat(
                            pred["timestamp"]
                        )
                    # Map old field names to new
                    if "metadata" in pred:
                        pred["extra_metadata"] = pred.pop("metadata")

                return repo.batch_create(batch)

        except Exception as e:
            self.logger.error("batch_insert_failed", batch_size=len(batch), error=str(e))
            return 0

    def migrate_health_snapshots(self) -> dict[str, Any]:
        """Migrate health snapshots from JSONL to database.

        Returns:
            Migration statistics dictionary
        """
        snapshots_file = self.data_dir / "health_snapshots.jsonl"

        if not snapshots_file.exists():
            self.logger.info(
                "health_snapshots_file_not_found", path=snapshots_file
            )
            return {"file_found": False, "records_migrated": 0}

        stats = {"records_migrated": 0, "errors": 0, "file_found": True}

        try:
            with open(snapshots_file) as f, DatabaseManager.session_scope() as session:  # noqa: SIM117
                repo = HealthSnapshotRepository(session)

                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)

                        # Convert timestamp
                        if isinstance(data.get("timestamp"), str):
                            data["timestamp"] = datetime.fromisoformat(
                                data["timestamp"]
                            )

                        repo.create(**data)
                        stats["records_migrated"] += 1

                    except json.JSONDecodeError as e:
                        self.logger.warning(
                            "json_parse_error",
                            file=snapshots_file,
                            line=line_num,
                            error=str(e),
                        )
                        stats["errors"] += 1

            self.logger.info(
                "health_snapshots_migration_complete",
                records_migrated=stats["records_migrated"],
                errors=stats["errors"],
            )
            return stats

        except Exception as e:
            self.logger.error("health_snapshots_migration_failed", error=str(e))
            stats["errors"] += 1
            return stats

    def migrate_alerts(self) -> dict[str, Any]:
        """Migrate alerts from JSONL to database.

        Returns:
            Migration statistics dictionary
        """
        alerts_file = self.data_dir / "alerts.jsonl"

        if not alerts_file.exists():
            self.logger.info("alerts_file_not_found", path=alerts_file)
            return {"file_found": False, "records_migrated": 0}

        stats = {"records_migrated": 0, "errors": 0, "file_found": True}

        try:
            with open(alerts_file) as f, DatabaseManager.session_scope() as session:  # noqa: SIM117
                repo = AlertRepository(session)

                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)

                        # Convert timestamps
                        for ts_field in ["timestamp", "acknowledged_at"]:
                            if isinstance(data.get(ts_field), str):
                                data[ts_field] = datetime.fromisoformat(
                                    data[ts_field]
                                )

                        repo.create(**data)
                        stats["records_migrated"] += 1

                    except json.JSONDecodeError as e:
                        self.logger.warning(
                            "json_parse_error",
                            file=alerts_file,
                            line=line_num,
                            error=str(e),
                        )
                        stats["errors"] += 1

            self.logger.info(
                "alerts_migration_complete",
                records_migrated=stats["records_migrated"],
                errors=stats["errors"],
            )
            return stats

        except Exception as e:
            self.logger.error("alerts_migration_failed", error=str(e))
            stats["errors"] += 1
            return stats

    def migrate_alert_rules(self) -> dict[str, Any]:
        """Migrate alert rules from JSON to database.

        Returns:
            Migration statistics dictionary
        """
        rules_file = self.data_dir / "alert_rules.json"

        if not rules_file.exists():
            self.logger.info("alert_rules_file_not_found", path=rules_file)
            return {"file_found": False, "records_migrated": 0}

        stats = {"records_migrated": 0, "errors": 0, "file_found": True}

        try:
            with open(rules_file) as f:
                rules_data = json.load(f)

            rules_list = rules_data if isinstance(rules_data, list) else [rules_data]

            with DatabaseManager.session_scope() as session:
                repo = AlertRuleRepository(session)

                for rule in rules_list:
                    try:
                        # Convert timestamp fields
                        for ts_field in ["last_triggered"]:
                            if isinstance(rule.get(ts_field), str):
                                rule[ts_field] = datetime.fromisoformat(
                                    rule[ts_field]
                                )

                        repo.create(**rule)
                        stats["records_migrated"] += 1

                    except Exception as e:
                        self.logger.warning(
                            "rule_insert_failed",
                            rule_id=rule.get("rule_id"),
                            error=str(e),
                        )
                        stats["errors"] += 1

            self.logger.info(
                "alert_rules_migration_complete",
                records_migrated=stats["records_migrated"],
                errors=stats["errors"],
            )
            return stats

        except Exception as e:
            self.logger.error("alert_rules_migration_failed", error=str(e))
            stats["errors"] += 1
            return stats

    def migrate_all(self) -> dict[str, Any]:
        """Migrate all monitoring data from files to database.

        Returns:
            Combined migration statistics from all operations
        """
        self.logger.info("migration_started", data_dir=self.data_dir)

        start_time = time.time()

        all_stats = {
            "predictions": self.migrate_predictions(),
            "health_snapshots": self.migrate_health_snapshots(),
            "alerts": self.migrate_alerts(),
            "alert_rules": self.migrate_alert_rules(),
            "duration_seconds": time.time() - start_time,
        }

        total_migrated = sum(
            s.get("records_migrated", 0) for s in all_stats.values() if isinstance(s, dict)
        )
        total_errors = sum(
            s.get("errors", 0) for s in all_stats.values() if isinstance(s, dict)
        )

        self.logger.info(
            "migration_completed",
            total_records=total_migrated,
            total_errors=total_errors,
            duration=all_stats["duration_seconds"],
        )

        return all_stats
