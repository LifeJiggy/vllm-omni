"""
Dynamic Model Registry for Real-time Model Switching

This module provides a dynamic model registry that extends the static OmniModelRegistry
to support runtime registration, deregistration, and management of multiple model versions.
"""

import asyncio
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.model_executor.models.registry import OmniModelRegistry

logger = init_logger(__name__)


@dataclass
class ModelInstance:
    """Represents a loaded model instance with metadata."""

    model_id: str
    version: str
    config: OmniModelConfig
    model: Any  # The actual loaded model
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    health_status: str = "healthy"  # healthy, degraded, failed
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        """Get age of model instance in seconds."""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Get idle time of model instance in seconds."""
        return time.time() - self.last_used

    def update_last_used(self):
        """Update the last used timestamp."""
        self.last_used = time.time()


@dataclass
class ModelVersion:
    """Represents a model version with metadata."""

    model_id: str
    version: str
    config: OmniModelConfig
    created_at: float = field(default_factory=time.time)
    status: str = "inactive"  # active, inactive, deprecated
    metadata: dict[str, Any] = field(default_factory=dict)


class DynamicModelRegistry:
    """
    Dynamic model registry that supports runtime model management.

    This class extends the static OmniModelRegistry to provide:
    - Runtime registration/deregistration of models
    - Version management and tracking
    - Health monitoring of loaded models
    - Automatic cleanup of unused models
    """

    def __init__(
        self,
        base_registry: OmniModelRegistry,
        max_cached_models: int = 5,
        model_ttl_seconds: int = 3600,
        cleanup_interval: float = 300.0,
    ):
        """
        Initialize the dynamic model registry.

        Args:
            base_registry: The base OmniModelRegistry instance
            max_cached_models: Maximum number of models to keep cached
            model_ttl_seconds: Time-to-live for cached models in seconds
            cleanup_interval: Interval for cleanup operations in seconds
        """
        self.base_registry = base_registry
        self.max_cached_models = max_cached_models
        self.model_ttl_seconds = model_ttl_seconds
        self.cleanup_interval = cleanup_interval

        # Model storage
        self.active_models: dict[str, ModelInstance] = {}  # model_id -> active ModelInstance
        self.model_versions: dict[str, list[ModelVersion]] = defaultdict(list)  # model_id -> List[ModelVersion]
        self.model_cache: dict[str, ModelInstance] = {}  # version_key -> ModelInstance

        # Threading and async
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="model-registry")
        self._cleanup_task: asyncio.Task | None = None

        # Statistics
        self.stats = {
            "models_registered": 0,
            "models_deregistered": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "load_failures": 0,
            "switch_operations": 0,
        }

        logger.info(
            f"Initialized DynamicModelRegistry with max_cached_models={max_cached_models}, "
            f"model_ttl={model_ttl_seconds}s"
        )

    def start_cleanup_task(self):
        """Start the background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def stop_cleanup_task(self):
        """Stop the background cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

    async def _cleanup_loop(self):
        """Background cleanup loop for expired models."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_expired_models(self):
        """Clean up expired models from cache."""
        expired_keys = []

        with self._lock:
            for key, instance in self.model_cache.items():
                if instance.idle_seconds > self.model_ttl_seconds and instance.model_id not in self.active_models:
                    expired_keys.append(key)

            for key in expired_keys:
                instance = self.model_cache.pop(key)
                logger.info(f"Cleaned up expired model: {instance.model_id} v{instance.version}")
                # TODO: Properly unload the model from memory
                del instance

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired models")

    def register_model(
        self, model_config: OmniModelConfig, version: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """
        Register a new model version.

        Args:
            model_config: Configuration for the model
            version: Version string for the model
            metadata: Optional metadata for the model

        Returns:
            model_id: Unique identifier for the registered model
        """
        model_id = f"{model_config.model_arch}_{model_config.model_stage}"

        with self._lock:
            # Create version entry
            version_entry = ModelVersion(
                model_id=model_id, version=version, config=model_config, metadata=metadata or {}
            )

            # Add to versions list
            self.model_versions[model_id].append(version_entry)

            # Sort versions by creation time (newest first)
            self.model_versions[model_id].sort(key=lambda v: v.created_at, reverse=True)

            self.stats["models_registered"] += 1

            logger.info(f"Registered model {model_id} version {version}")

        return model_id

    def deregister_model(self, model_id: str, version: str) -> bool:
        """
        Deregister a model version.

        Args:
            model_id: Model identifier
            version: Version to deregister

        Returns:
            True if successfully deregistered, False otherwise
        """
        with self._lock:
            if model_id not in self.model_versions:
                logger.warning(f"Model {model_id} not found for deregistration")
                return False

            # Find and remove the version
            versions = self.model_versions[model_id]
            for i, v in enumerate(versions):
                if v.version == version:
                    _removed_version = versions.pop(i)

                    # If this was the active model, remove it
                    if model_id in self.active_models and self.active_models[model_id].version == version:
                        del self.active_models[model_id]

                    # Remove from cache if present
                    cache_key = f"{model_id}_{version}"
                    if cache_key in self.model_cache:
                        del self.model_cache[cache_key]

                    self.stats["models_deregistered"] += 1
                    logger.info(f"Deregistered model {model_id} version {version}")
                    return True

            logger.warning(f"Version {version} not found for model {model_id}")
            return False

    def get_active_model(self, model_id: str) -> ModelInstance | None:
        """
        Get the currently active model instance for a model ID.

        Args:
            model_id: Model identifier

        Returns:
            Active ModelInstance or None if not found
        """
        with self._lock:
            instance = self.active_models.get(model_id)
            if instance:
                instance.update_last_used()
                self.stats["cache_hits"] += 1
            else:
                self.stats["cache_misses"] += 1
            return instance

    def switch_model(self, model_id: str, target_version: str) -> bool:
        """
        Switch the active model to a target version.

        Args:
            model_id: Model identifier
            target_version: Target version to switch to

        Returns:
            True if switch successful, False otherwise
        """
        with self._lock:
            # Find the target version
            if model_id not in self.model_versions:
                logger.error(f"Model {model_id} not found")
                return False

            target_version_obj = None
            for version in self.model_versions[model_id]:
                if version.version == target_version:
                    target_version_obj = version
                    break

            if not target_version_obj:
                logger.error(f"Version {target_version} not found for model {model_id}")
                return False

            # Try to load the model if not in cache
            cache_key = f"{model_id}_{target_version}"
            if cache_key not in self.model_cache:
                try:
                    model_instance = self._load_model(target_version_obj)
                    self.model_cache[cache_key] = model_instance
                except Exception as e:
                    logger.error(f"Failed to load model {model_id} v{target_version}: {e}")
                    self.stats["load_failures"] += 1
                    return False

            # Switch active model
            self.active_models[model_id] = self.model_cache[cache_key]
            self.stats["switch_operations"] += 1

            logger.info(f"Switched model {model_id} to version {target_version}")
            return True

    def _load_model(self, version: ModelVersion) -> ModelInstance:
        """
        Load a model instance from configuration.

        Args:
            version: ModelVersion to load

        Returns:
            Loaded ModelInstance

        Raises:
            Exception: If model loading fails
        """
        # This is a placeholder - actual model loading logic would go here
        # In a real implementation, this would use the base registry and model loading utilities

        logger.info(f"Loading model {version.model_id} version {version.version}")

        # Simulate model loading (replace with actual loading logic)
        model = f"mock_model_{version.model_id}_{version.version}"  # Placeholder

        return ModelInstance(
            model_id=version.model_id,
            version=version.version,
            config=version.config,
            model=model,
            metadata={"loaded_at": time.time()},
        )

    def get_model_versions(self, model_id: str) -> list[ModelVersion]:
        """
        Get all versions for a model.

        Args:
            model_id: Model identifier

        Returns:
            List of ModelVersion objects
        """
        with self._lock:
            return self.model_versions.get(model_id, []).copy()

    def get_registry_stats(self) -> dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        with self._lock:
            return {
                **self.stats,
                "active_models": len(self.active_models),
                "cached_models": len(self.model_cache),
                "total_versions": sum(len(versions) for versions in self.model_versions.values()),
            }

    def list_models(self) -> dict[str, list[str]]:
        """
        List all registered models and their versions.

        Returns:
            Dictionary mapping model_id to list of versions
        """
        with self._lock:
            return {model_id: [v.version for v in versions] for model_id, versions in self.model_versions.items()}

    def __del__(self):
        """Cleanup on destruction."""
        self.stop_cleanup_task()
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
