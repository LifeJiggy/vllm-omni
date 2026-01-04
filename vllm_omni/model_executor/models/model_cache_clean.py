"""
Model Cache for Real-time Model Switching

This module provides intelligent caching for loaded models with LRU eviction,
memory-aware management, and pre-loading capabilities.
"""

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

from vllm_omni.model_executor.models.dynamic_registry import ModelInstance
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached model entry with metadata."""
    model_instance: ModelInstance
    cache_key: str
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0  # Estimated memory usage
    priority: int = 0  # Higher priority items are evicted last

    @property
    def idle_time(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_accessed

    def access(self):
        """Mark entry as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class CacheStats:
    """Statistics for the model cache."""
    total_entries: int = 0
    total_size_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    loads: int = 0
    max_size_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.hits + self.misses
        return self.hits / total_requests if total_requests > 0 else 0.0

    @property
    def utilization_rate(self) -> float:
        """Calculate cache utilization rate."""
        return self.total_size_bytes / self.max_size_bytes if self.max_size_bytes > 0 else 0.0


class MemoryManager:
    """Manages memory usage for cached models."""

    def __init__(self, max_memory_gb: float = 8.0):
        """
        Initialize memory manager.

        Args:
            max_memory_gb: Maximum memory to use for cached models
        """
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.current_memory_bytes = 0
        self._lock = threading.RLock()

    def estimate_model_size(self, model_instance: ModelInstance) -> int:
        """
        Estimate the memory usage of a model instance.

        Args:
            model_instance: Model instance to estimate

        Returns:
            Estimated size in bytes
        """
        # This is a simplified estimation - in practice, you'd use more sophisticated
        # memory profiling techniques
        base_size = 1024 * 1024 * 1024  # 1GB base estimate

        # Adjust based on model architecture
        if "qwen3" in model_instance.model_id.lower():
            base_size *= 1.5  # MoE models typically use more memory
        elif "thinker" in model_instance.model_stage:
            base_size *= 0.8  # Thinker models might be smaller
        elif "talker" in model_instance.model_stage:
            base_size *= 0.6  # Talker models are typically smaller

        return base_size

    def can_allocate(self, size_bytes: int) -> bool:
        """
        Check if memory can be allocated.

        Args:
            size_bytes: Size to allocate

        Returns:
            True if allocation is possible
        """
        with self._lock:
            return self.current_memory_bytes + size_bytes <= self.max_memory_bytes

    def allocate(self, size_bytes: int) -> bool:
        """
        Allocate memory.

        Args:
            size_bytes: Size to allocate

        Returns:
            True if allocation successful
        """
        with self._lock:
            if self.can_allocate(size_bytes):
                self.current_memory_bytes += size_bytes
                return True
            return False

    def deallocate(self, size_bytes: int):
        """
        Deallocate memory.

        Args:
            size_bytes: Size to deallocate
        """
        with self._lock:
            self.current_memory_bytes = max(0, self.current_memory_bytes - size_bytes)

    def get_available_memory(self) -> int:
        """Get available memory in bytes."""
        with self._lock:
            return max(0, self.max_memory_bytes - self.current_memory_bytes)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            return {
                "max_memory_bytes": self.max_memory_bytes,
                "current_memory_bytes": self.current_memory_bytes,
                "available_memory_bytes": self.get_available_memory(),
                "utilization_rate": self.current_memory_bytes / self.max_memory_bytes
            }


class ModelCache:
    """
    Intelligent cache for loaded models with LRU eviction and memory management.

    This class provides:
    - LRU eviction policy
    - Memory-aware caching
    - Pre-loading capabilities
    - Cache warming strategies
    - Priority-based eviction
    """

    def __init__(self, max_cache_size: int = 5, memory_manager: Optional[MemoryManager] = None,
                 eviction_interval: float = 60.0):
        """
        Initialize the model cache.

        Args:
            max_cache_size: Maximum number of models to cache
            memory_manager: Memory manager instance
            eviction_interval: Interval for background eviction in seconds
        """
        self.max_cache_size = max_cache_size
        self.memory_manager = memory_manager or MemoryManager()
        self.eviction_interval = eviction_interval

        # Cache storage - OrderedDict for LRU behavior
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Background tasks
        self._eviction_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="model-cache")

        # Statistics
        self.stats = CacheStats(max_size_bytes=self.memory_manager.max_memory_bytes)

        # Pre-loading queue
        self.preload_queue: List[Tuple[str, int]] = []
        self.preload_lock = threading.Lock()

        logger.info(f"Initialized ModelCache with max_size={max_cache_size}, "
                   f"max_memory={self.memory_manager.max_memory_bytes / 1024**3:.1f}GB")

    def start_background_tasks(self):
        """Start background eviction and pre-loading tasks."""
        if self._eviction_task is None or self._eviction_task.done():
            self._eviction_task = asyncio.create_task(self._eviction_loop())

    def stop_background_tasks(self):
        """Stop background tasks."""
        if self._eviction_task and not self._eviction_task.done():
            self._eviction_task.cancel()

    async def _eviction_loop(self):
        """Background eviction loop."""
        while True:
            try:
                await asyncio.sleep(self.eviction_interval)
                await self._perform_eviction()
                await self._process_preload_queue()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in eviction loop: {e}")

    async def _perform_eviction(self):
        """Perform cache eviction based on LRU and memory pressure."""
        with self._lock:
            if len(self.cache) <= self.max_cache_size:
                return

            # Sort entries by priority (higher first) then by last access (LRU)
            entries = list(self.cache.items())
            entries.sort(key=lambda x: (-x[1].priority, x[1].last_accessed))

            # Evict least recently used items until we're under limits
            evicted_count = 0
            while len(self.cache) > self.max_cache_size and entries:
                key, entry = entries.pop()  # Remove LRU item

                # Check if we can evict this item
                if entry.priority > 0:  # Don't evict high priority items
                    continue

                # Remove from cache
                del self.cache[key]

                # Free memory
                self.memory_manager.deallocate(entry.size_bytes)

                # Update stats
                self.stats.total_entries -= 1
                self.stats.total_size_bytes -= entry.size_bytes
                self.stats.evictions += 1
                evicted_count += 1

                logger.debug(f"Evicted model {entry.cache_key} from cache")

            if evicted_count > 0:
                logger.info(f"Evicted {evicted_count} models from cache")

    async def _process_preload_queue(self):
        """Process the pre-loading queue."""
        with self.preload_lock:
            if not self.preload_queue:
                return

            # Process one item at a time to avoid overwhelming the system
            cache_key, priority = self.preload_queue.pop(0)

            # Check if already in cache
            with self._lock:
                if cache_key in self.cache:
                    return

            # TODO: Implement actual pre-loading logic
            # This would involve loading the model in the background
            logger.debug(f"Pre-loading model {cache_key} with priority {priority}")

    def get_model(self, cache_key: str) -> Optional[ModelInstance]:
        """
        Retrieve a model from cache.

        Args:
            cache_key: Cache key for the model

        Returns:
            ModelInstance if found, None otherwise
        """
        with self._lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                entry.access()

                # Move to end (most recently used)
                self.cache.move_to_end(cache_key)

                self.stats.hits += 1
                return entry.model_instance
            else:
                self.stats.misses += 1
                return None

    def put_model(self, cache_key: str, model_instance: ModelInstance,
                  priority: int = 0) -> bool:
        """
        Store a model in cache.

        Args:
            cache_key: Cache key for the model
            model_instance: Model instance to cache
            priority: Cache priority (higher = less likely to be evicted)

        Returns:
            True if successfully cached, False otherwise
        """
        # Estimate memory usage
        size_bytes = self.memory_manager.estimate_model_size(model_instance)

        # Check if we can allocate memory
        if not self.memory_manager.allocate(size_bytes):
            logger.warning(f"Cannot cache model {cache_key}: insufficient memory")
            return False

        with self._lock:
            # Remove existing entry if present
            if cache_key in self.cache:
                old_entry = self.cache[cache_key]
                self.memory_manager.deallocate(old_entry.size_bytes)
                self.stats.total_size_bytes -= old_entry.size_bytes
                del self.cache[cache_key]
                self.stats.total_entries -= 1

            # Create new entry
            entry = CacheEntry(
                model_instance=model_instance,
                cache_key=cache_key,
                size_bytes=size_bytes,
                priority=priority
            )

            # Add to cache
            self.cache[cache_key] = entry
            self.cache.move_to_end(cache_key)  # Mark as recently used

            # Update stats
            self.stats.total_entries += 1
            self.stats.total_size_bytes += size_bytes
            self.stats.loads += 1

            logger.debug(f"Cached model {cache_key} (size: {size_bytes / 1024**3:.2f}GB)")
            return True

    def preload_model(self, cache_key: str, priority: int = 0):
        """
        Pre-load a model into cache.

        Args:
            cache_key: Cache key for the model
            priority: Loading priority
        """
        with self.preload_lock:
            # Add to preload queue if not already present
            for existing_key, existing_priority in self.preload_queue:
                if existing_key == cache_key:
                    # Update priority if higher
                    if priority > existing_priority:
                        self.preload_queue.remove((existing_key, existing_priority))
                        break
                    else:
                        return

            self.preload_queue.append((cache_key, priority))
            # Sort by priority (higher first)
            self.preload_queue.sort(key=lambda x: x[1], reverse=True)

            logger.debug(f"Queued model {cache_key} for pre-loading (priority: {priority})")

    def evict_model(self, cache_key: str) -> bool:
        """
        Manually evict a model from cache.

        Args:
            cache_key: Cache key to evict

        Returns:
            True if evicted, False if not found
        """
        with self._lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                del self.cache[cache_key]

                # Free memory
                self.memory_manager.deallocate(entry.size_bytes)

                # Update stats
                self.stats.total_entries -= 1
                self.stats.total_size_bytes -= entry.size_bytes
                self.stats.evictions += 1

                logger.info(f"Manually evicted model {cache_key} from cache")
                return True
            return False

    def clear_cache(self):
        """Clear all cached models."""
        with self._lock:
            evicted_count = len(self.cache)
            total_size = self.stats.total_size_bytes

            for entry in self.cache.values():
                self.memory_manager.deallocate(entry.size_bytes)

            self.cache.clear()

            # Reset stats
            self.stats.total_entries = 0
            self.stats.total_size_bytes = 0
            self.stats.evictions += evicted_count

            logger.info(f"Cleared cache: evicted {evicted_count} models "
                       f"(freed {total_size / 1024**3:.2f}GB)")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            memory_stats = self.memory_manager.get_memory_stats()

            return {
                "cache_entries": self.stats.total_entries,
                "cache_size_bytes": self.stats.total_size_bytes,
                "cache_size_gb": self.stats.total_size_bytes / 1024**3,
                "max_cache_size": self.max_cache_size,
                "hit_rate": self.stats.hit_rate,
                "utilization_rate": self.stats.utilization_rate,
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "evictions": self.stats.evictions,
                "loads": self.stats.loads,
                "memory_stats": memory_stats,
                "preload_queue_size": len(self.preload_queue)
            }

    def list_cached_models(self) -> List[Dict[str, Any]]:
        """
        List all cached models with metadata.

        Returns:
            List of cached model information
        """
        with self._lock:
            return [
                {
                    "cache_key": entry.cache_key,
                    "model_id": entry.model_instance.model_id,
                    "version": entry.model_instance.version,
                    "access_count": entry.access_count,
                    "last_accessed": entry.last_accessed,
                    "idle_time_seconds": entry.idle_time,
                    "size_bytes": entry.size_bytes,
                    "size_gb": entry.size_bytes / 1024**3,
                    "priority": entry.priority
                }
                for entry in self.cache.values()
            ]

    def __contains__(self, cache_key: str) -> bool:
        """Check if a cache key exists."""
        with self._lock:
            return cache_key in self.cache

    def __len__(self) -> int:
        """Get number of cached models."""
        with self._lock:
            return len(self.cache)

    def __del__(self):
        """Cleanup on destruction."""
        self.stop_background_tasks()
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)