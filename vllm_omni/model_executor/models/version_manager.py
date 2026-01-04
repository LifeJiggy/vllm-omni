"""
Model Version Manager for Real-time Model Switching

This module provides version management capabilities including version history tracking,
metadata storage, rollback support, and version comparison.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from vllm_omni.config.model import OmniModelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class VersionHistory:
    """Represents the version history for a model."""
    model_id: str
    versions: List['ModelVersion'] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def add_version(self, version: 'ModelVersion'):
        """Add a version to history."""
        self.versions.append(version)
        self.versions.sort(key=lambda v: v.created_at, reverse=True)
        self.last_updated = time.time()

    def get_version(self, version: str) -> Optional['ModelVersion']:
        """Get a specific version."""
        for v in self.versions:
            if v.version == version:
                return v
        return None

    def get_latest_version(self) -> Optional['ModelVersion']:
        """Get the latest version."""
        return self.versions[0] if self.versions else None

    def remove_version(self, version: str) -> bool:
        """Remove a version from history."""
        for i, v in enumerate(self.versions):
            if v.version == version:
                self.versions.pop(i)
                self.last_updated = time.time()
                return True
        return False


@dataclass
class VersionComparison:
    """Result of comparing two model versions."""
    version1: str
    version2: str
    differences: Dict[str, Any]
    compatibility_score: float  # 0.0 to 1.0, higher is more compatible
    breaking_changes: List[str]
    recommendations: List[str]


class VersionStorage:
    """Abstract base class for version storage backends."""

    def save_version_history(self, history: VersionHistory) -> bool:
        """Save version history."""
        raise NotImplementedError

    def load_version_history(self, model_id: str) -> Optional[VersionHistory]:
        """Load version history for a model."""
        raise NotImplementedError

    def list_models(self) -> List[str]:
        """List all models with version history."""
        raise NotImplementedError

    def delete_version_history(self, model_id: str) -> bool:
        """Delete version history for a model."""
        raise NotImplementedError


class FileVersionStorage(VersionStorage):
    """File-based version storage."""

    def __init__(self, storage_path: str = "./model_versions"):
        """
        Initialize file-based storage.

        Args:
            storage_path: Directory to store version files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_history_file(self, model_id: str) -> Path:
        """Get the file path for a model's version history."""
        return self.storage_path / f"{model_id}_versions.json"

    def save_version_history(self, history: VersionHistory) -> bool:
        """Save version history to file."""
        try:
            file_path = self._get_history_file(history.model_id)
            data = {
                "model_id": history.model_id,
                "versions": [
                    {
                        "version": v.version,
                        "config": {
                            "model": v.config.model,
                            "model_arch": v.config.model_arch,
                            "model_stage": v.config.model_stage,
                            "stage_id": v.config.stage_id,
                        },
                        "created_at": v.created_at,
                        "status": v.status,
                        "metadata": v.metadata
                    }
                    for v in history.versions
                ],
                "created_at": history.created_at,
                "last_updated": history.last_updated
            }

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            return True
        except Exception as e:
            logger.error(f"Failed to save version history for {history.model_id}: {e}")
            return False

    def load_version_history(self, model_id: str) -> Optional[VersionHistory]:
        """Load version history from file."""
        try:
            file_path = self._get_history_file(model_id)
            if not file_path.exists():
                return None

            with open(file_path, 'r') as f:
                data = json.load(f)

            # Reconstruct VersionHistory
            history = VersionHistory(
                model_id=data["model_id"],
                created_at=data["created_at"],
                last_updated=data["last_updated"]
            )

            for v_data in data["versions"]:
                # Reconstruct OmniModelConfig (simplified)
                config = OmniModelConfig(
                    model=v_data["config"]["model"],
                    model_arch=v_data["config"]["model_arch"],
                    model_stage=v_data["config"]["model_stage"],
                    stage_id=v_data["config"]["stage_id"]
                )

                version = ModelVersion(
                    model_id=model_id,
                    version=v_data["version"],
                    config=config,
                    created_at=v_data["created_at"],
                    status=v_data["status"],
                    metadata=v_data["metadata"]
                )

                history.versions.append(version)

            return history
        except Exception as e:
            logger.error(f"Failed to load version history for {model_id}: {e}")
            return None

    def list_models(self) -> List[str]:
        """List all models with version history."""
        try:
            return [
                f.stem.replace("_versions", "")
                for f in self.storage_path.glob("*_versions.json")
            ]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def delete_version_history(self, model_id: str) -> bool:
        """Delete version history file."""
        try:
            file_path = self._get_history_file(model_id)
            if file_path.exists():
                file_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete version history for {model_id}: {e}")
            return False


class ModelVersionManager:
    """
    Manages model versions, metadata, and provides rollback capabilities.

    This class provides:
    - Version history tracking
    - Metadata storage and retrieval
    - Rollback to previous versions
    - Version comparison and validation
    """

    def __init__(self, storage_backend: Optional[VersionStorage] = None):
        """
        Initialize the version manager.

        Args:
            storage_backend: Storage backend for version data (defaults to FileVersionStorage)
        """
        self.storage = storage_backend or FileVersionStorage()
        self.version_histories: Dict[str, VersionHistory] = {}
        self._load_all_histories()

    def _load_all_histories(self):
        """Load all version histories from storage."""
        model_ids = self.storage.list_models()
        for model_id in model_ids:
            history = self.storage.load_version_history(model_id)
            if history:
                self.version_histories[model_id] = history

    def create_version(self, model_id: str, model_config: OmniModelConfig,
                      metadata: Optional[Dict[str, Any]] = None) -> 'ModelVersion':
        """
        Create a new model version.

        Args:
            model_id: Model identifier
            model_config: Model configuration
            metadata: Optional metadata

        Returns:
            Created ModelVersion
        """
        # Generate version string (timestamp-based for simplicity)
        version = f"v{int(time.time())}"

        version_obj = ModelVersion(
            model_id=model_id,
            version=version,
            config=model_config,
            metadata=metadata or {}
        )

        # Get or create version history
        if model_id not in self.version_histories:
            self.version_histories[model_id] = VersionHistory(model_id=model_id)

        history = self.version_histories[model_id]
        history.add_version(version_obj)

        # Save to storage
        self.storage.save_version_history(history)

        logger.info(f"Created version {version} for model {model_id}")
        return version_obj

    def get_version(self, model_id: str, version: str) -> Optional['ModelVersion']:
        """
        Retrieve a specific model version.

        Args:
            model_id: Model identifier
            version: Version string

        Returns:
            ModelVersion if found, None otherwise
        """
        history = self.version_histories.get(model_id)
        if history:
            return history.get_version(version)
        return None

    def get_latest_version(self, model_id: str) -> Optional['ModelVersion']:
        """
        Get the latest version for a model.

        Args:
            model_id: Model identifier

        Returns:
            Latest ModelVersion if found, None otherwise
        """
        history = self.version_histories.get(model_id)
        if history:
            return history.get_latest_version()
        return None

    def rollback_to_version(self, model_id: str, version: str) -> bool:
        """
        Rollback to a specific version.

        Args:
            model_id: Model identifier
            version: Target version to rollback to

        Returns:
            True if rollback successful, False otherwise
        """
        history = self.version_histories.get(model_id)
        if not history:
            logger.error(f"No version history found for model {model_id}")
            return False

        target_version = history.get_version(version)
        if not target_version:
            logger.error(f"Version {version} not found for model {model_id}")
            return False

        # Mark target version as active
        for v in history.versions:
            if v.version == version:
                v.status = "active"
            elif v.status == "active":
                v.status = "inactive"

        # Save updated history
        success = self.storage.save_version_history(history)
        if success:
            logger.info(f"Rolled back model {model_id} to version {version}")
        return success

    def compare_versions(self, model_id: str, version1: str, version2: str) -> Optional[VersionComparison]:
        """
        Compare two model versions.

        Args:
            model_id: Model identifier
            version1: First version to compare
            version2: Second version to compare

        Returns:
            VersionComparison if both versions found, None otherwise
        """
        history = self.version_histories.get(model_id)
        if not history:
            return None

        v1 = history.get_version(version1)
        v2 = history.get_version(version2)

        if not v1 or not v2:
            return None

        # Perform comparison (simplified)
        differences = {}
        breaking_changes = []
        recommendations = []

        # Compare configurations
        if v1.config.model_arch != v2.config.model_arch:
            differences["model_arch"] = {
                "from": v1.config.model_arch,
                "to": v2.config.model_arch
            }
            breaking_changes.append("Model architecture changed")
            recommendations.append("Verify model compatibility")

        if v1.config.model_stage != v2.config.model_stage:
            differences["model_stage"] = {
                "from": v1.config.model_stage,
                "to": v2.config.model_stage
            }
            breaking_changes.append("Model stage changed")
            recommendations.append("Test stage-specific functionality")

        # Calculate compatibility score (simplified)
        compatibility_score = 1.0
        if breaking_changes:
            compatibility_score = 0.5  # Reduced for breaking changes

        return VersionComparison(
            version1=version1,
            version2=version2,
            differences=differences,
            compatibility_score=compatibility_score,
            breaking_changes=breaking_changes,
            recommendations=recommendations
        )

    def list_versions(self, model_id: str) -> List['ModelVersion']:
        """
        List all versions for a model.

        Args:
            model_id: Model identifier

        Returns:
            List of ModelVersion objects
        """
        history = self.version_histories.get(model_id)
        if history:
            return history.versions.copy()
        return []

    def get_version_history(self, model_id: str) -> Optional[VersionHistory]:
        """
        Get the complete version history for a model.

        Args:
            model_id: Model identifier

        Returns:
            VersionHistory if found, None otherwise
        """
        return self.version_histories.get(model_id)

    def delete_model_versions(self, model_id: str) -> bool:
        """
        Delete all version history for a model.

        Args:
            model_id: Model identifier

        Returns:
            True if successful, False otherwise
        """
        if model_id in self.version_histories:
            del self.version_histories[model_id]

        success = self.storage.delete_version_history(model_id)
        if success:
            logger.info(f"Deleted version history for model {model_id}")
        return success

    def list_models(self) -> List[str]:
        """
        List all models with version history.

        Returns:
            List of model IDs
        """
        return list(self.version_histories.keys())


# Import ModelVersion here to avoid circular imports
@dataclass
class ModelVersion:
    """Represents a model version with metadata."""
    model_id: str
    version: str
    config: OmniModelConfig
    created_at: float = field(default_factory=time.time)
    status: str = "inactive"  # active, inactive, deprecated
    metadata: Dict[str, Any] = field(default_factory=dict)
