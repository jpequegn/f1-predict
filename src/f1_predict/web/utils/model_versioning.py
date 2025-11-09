"""Model versioning and rollback system.

Provides:
- Model versioning and metadata tracking
- Model registry and storage management
- Rollback to previous model versions
- Model comparison and validation
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a model version."""

    model_version: str
    timestamp: float
    model_type: str  # "ensemble", "xgboost", "lightgbm", etc.
    model_path: str
    parent_version: Optional[str] = None
    status: str = "active"  # "active", "candidate", "deprecated", "failed"
    metrics: dict[str, float] = field(default_factory=dict)
    description: str = ""
    tags: list[str] = field(default_factory=list)
    created_by: str = "system"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ModelRegistry:
    """Manages model versions and metadata."""

    def __init__(self, registry_dir: Path | str = "models/registry"):
        """Initialize model registry.

        Args:
            registry_dir: Directory for model storage
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "metadata.jsonl"
        self.logger = logger.bind(component="model_registry")
        self.active_model: Optional[str] = None
        self._load_active_model()

    def register_model(
        self,
        model_type: str,
        model_path: Path | str,
        metrics: Optional[dict[str, float]] = None,
        parent_version: Optional[str] = None,
        description: str = "",
        tags: Optional[list[str]] = None,
        created_by: str = "system",
    ) -> ModelMetadata:
        """Register a new model version.

        Args:
            model_type: Type of model
            model_path: Path to saved model
            metrics: Performance metrics
            parent_version: Parent model version (for tracking lineage)
            description: Model description
            tags: List of tags
            created_by: User who created model

        Returns:
            ModelMetadata for the registered model
        """
        # Generate version ID
        model_version = f"{model_type}_v{int(time.time() * 1000)}"

        # Copy model to registry
        model_path = Path(model_path)
        if model_path.exists():
            dest_path = self.registry_dir / model_version
            if model_path.is_dir():
                shutil.copytree(model_path, dest_path, dirs_exist_ok=True)
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(model_path, dest_path)
        else:
            dest_path = self.registry_dir / model_version
            dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata = ModelMetadata(
            model_version=model_version,
            timestamp=time.time(),
            model_type=model_type,
            model_path=str(dest_path),
            parent_version=parent_version,
            metrics=metrics or {},
            description=description,
            tags=tags or [],
            created_by=created_by,
        )

        # Store metadata
        with open(self.metadata_file, "a") as f:
            f.write(json.dumps(metadata.to_dict()) + "\n")

        self.logger.info(
            "model_registered",
            model_version=model_version,
            model_type=model_type,
            metrics=metrics,
        )

        return metadata

    def get_model_metadata(self, model_version: str) -> Optional[ModelMetadata]:
        """Get metadata for a model version.

        Args:
            model_version: Model version ID

        Returns:
            ModelMetadata or None if not found
        """
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if data["model_version"] == model_version:
                                return ModelMetadata(**data)
        except Exception as e:
            self.logger.error("error_loading_metadata", error=str(e))
        return None

    def list_model_versions(
        self,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[ModelMetadata]:
        """List model versions.

        Args:
            model_type: Filter by model type
            status: Filter by status
            limit: Maximum versions to return

        Returns:
            List of ModelMetadata objects
        """
        versions = []
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if (
                                (model_type is None or data["model_type"] == model_type)
                                and (status is None or data["status"] == status)
                            ):
                                versions.append(ModelMetadata(**data))
        except Exception as e:
            self.logger.error("error_listing_versions", error=str(e))

        return sorted(versions, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_active_model(self) -> Optional[ModelMetadata]:
        """Get currently active model.

        Returns:
            Active model metadata or None
        """
        if self.active_model:
            return self.get_model_metadata(self.active_model)
        return None

    def activate_model(self, model_version: str) -> bool:
        """Activate a model version.

        Args:
            model_version: Model version to activate

        Returns:
            True if successful
        """
        metadata = self.get_model_metadata(model_version)
        if not metadata:
            self.logger.error("model_not_found", model_version=model_version)
            return False

        # Deactivate current model
        if self.active_model:
            current = self.get_model_metadata(self.active_model)
            if current:
                current.status = "inactive"
                self._update_metadata(current)

        # Activate new model
        metadata.status = "active"
        self._update_metadata(metadata)
        self.active_model = model_version

        # Save to file
        active_file = self.registry_dir / "active_model.txt"
        with open(active_file, "w") as f:
            f.write(model_version)

        self.logger.info("model_activated", model_version=model_version)
        return True

    def set_model_status(self, model_version: str, status: str) -> bool:
        """Set model status.

        Args:
            model_version: Model version
            status: New status

        Returns:
            True if successful
        """
        metadata = self.get_model_metadata(model_version)
        if not metadata:
            return False

        metadata.status = status
        self._update_metadata(metadata)

        self.logger.info(
            "model_status_updated",
            model_version=model_version,
            status=status,
        )
        return True

    def compare_models(
        self, version1: str, version2: str
    ) -> Optional[dict[str, Any]]:
        """Compare two model versions.

        Args:
            version1: First model version
            version2: Second model version

        Returns:
            Comparison dictionary or None
        """
        meta1 = self.get_model_metadata(version1)
        meta2 = self.get_model_metadata(version2)

        if not meta1 or not meta2:
            self.logger.error("model_not_found_for_comparison")
            return None

        # Compare metrics
        metric_comparison = {}
        all_metrics = set(meta1.metrics.keys()) | set(meta2.metrics.keys())

        for metric in all_metrics:
            val1 = meta1.metrics.get(metric, 0.0)
            val2 = meta2.metrics.get(metric, 0.0)
            diff = val2 - val1
            pct_change = (diff / (val1 + 1e-10)) * 100

            metric_comparison[metric] = {
                "version1": val1,
                "version2": val2,
                "difference": diff,
                "percent_change": pct_change,
                "improved": diff > 0,
            }

        return {
            "version1": version1,
            "version2": version2,
            "metrics_comparison": metric_comparison,
            "version1_created": datetime.fromtimestamp(meta1.timestamp).isoformat(),
            "version2_created": datetime.fromtimestamp(meta2.timestamp).isoformat(),
            "version1_status": meta1.status,
            "version2_status": meta2.status,
        }


    def rollback_to_version(self, model_version: str) -> bool:
        """Rollback to a previous model version.

        Args:
            model_version: Model version to rollback to

        Returns:
            True if successful
        """
        metadata = self.get_model_metadata(model_version)
        if not metadata:
            self.logger.error("model_not_found_for_rollback", model_version=model_version)
            return False

        # Activate the previous version
        self.activate_model(model_version)

        self.logger.warning(
            "model_rollback_performed",
            model_version=model_version,
            timestamp=datetime.now().isoformat(),
        )

        return True

    def delete_model_version(self, model_version: str) -> bool:
        """Delete a model version.

        Args:
            model_version: Model version to delete

        Returns:
            True if successful
        """
        try:
            metadata = self.get_model_metadata(model_version)
            if not metadata:
                return False

            # Delete model files
            model_path = Path(metadata.model_path)
            if model_path.exists():
                if model_path.is_dir():
                    shutil.rmtree(model_path)
                else:
                    model_path.unlink()

            # Mark as deleted in metadata
            metadata.status = "deleted"
            self._update_metadata(metadata)

            self.logger.info(
                "model_version_deleted",
                model_version=model_version,
            )
            return True
        except Exception as e:
            self.logger.error("error_deleting_model", error=str(e))
            return False

    def get_model_lineage(self, model_version: str) -> list[ModelMetadata]:
        """Get lineage of a model version.

        Args:
            model_version: Model version to trace

        Returns:
            List of models in lineage
        """
        lineage = []
        current = self.get_model_metadata(model_version)

        while current:
            lineage.append(current)
            if current.parent_version:
                current = self.get_model_metadata(current.parent_version)
            else:
                break

        return lineage

    def _update_metadata(self, metadata: ModelMetadata) -> None:
        """Update metadata for a model.

        Args:
            metadata: Updated metadata
        """
        try:
            versions = self.list_model_versions(limit=10000)

            with open(self.metadata_file, "w") as f:
                for v in versions:
                    if v.model_version == metadata.model_version:
                        f.write(json.dumps(metadata.to_dict()) + "\n")
                    else:
                        f.write(json.dumps(v.to_dict()) + "\n")
        except Exception as e:
            self.logger.error("error_updating_metadata", error=str(e))

    def _load_active_model(self) -> None:
        """Load active model from file."""
        try:
            active_file = self.registry_dir / "active_model.txt"
            if active_file.exists():
                with open(active_file) as f:
                    self.active_model = f.read().strip()
        except Exception as e:
            self.logger.error("error_loading_active_model", error=str(e))
