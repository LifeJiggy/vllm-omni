# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Advanced Memory Management System for vLLM-Omni

This module implements intelligent memory pooling, LRU caching for model weights,
and automatic offloading strategies to optimize memory usage.
"""

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory pool management."""
    enable_memory_pools: bool = True
    pool_sizes: Dict[str, int] = field(default_factory=lambda: {
        "cpu": 8 * 1024**3,  # 8GB
        "cuda": 16 * 1024**3  # 16GB
    })
    fragmentation_threshold: float = 0.1  # 10% fragmentation triggers cleanup
    cleanup_interval: float = 60.0  # seconds


@dataclass
class CacheConfig:
    """Configuration for model weight caching."""
    max_size_gb: float = 4.0
    enable_compression: bool = False
    compression_ratio: float = 0.5
    eviction_batch_size: int = 10
    access_count_threshold: int = 3  # Minimum accesses before caching


@dataclass
class OffloadConfig:
    """Configuration for automatic offloading."""
    enable_auto_offload: bool = True
    offload_threshold: float = 0.8  # 80% memory usage triggers offload
    restore_threshold: float = 0.6  # 60% memory usage allows restore
    min_offload_interval: float = 10.0  # seconds between offloads
    usage_window: float = 300.0  # seconds to track usage patterns


@dataclass
class MonitorConfig:
    """Configuration for memory monitoring."""
    enable_monitoring: bool = True
    monitoring_interval: float = 1.0  # seconds
    history_size: int = 1000
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "warning": 0.75,
        "critical": 0.9
    })


@dataclass
class AdvancedMemoryConfig:
    """Combined configuration for advanced memory management."""
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)


@dataclass
class MemoryBlock:
    """Represents a block of allocated memory."""
    device: str
    size: int
    ptr: Any = None
    allocated_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


@dataclass
class PoolStats:
    """Statistics for a memory pool."""
    total_size: int
    used_size: int
    free_size: int
    fragmentation_ratio: float
    allocation_count: int
    deallocation_count: int


@dataclass
class MemoryStats:
    """Comprehensive memory statistics."""
    system_memory: Tuple[int, int]  # total, available
    gpu_memory: Optional[Tuple[int, int]] = None  # total, free
    pool_stats: Dict[str, PoolStats] = field(default_factory=dict)
    cache_stats: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MemoryPoolManager:
    """
    Intelligent memory pool manager for different devices.

    Manages memory pools to reduce fragmentation and optimize allocation patterns.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.pools: Dict[str, Dict[str, MemoryBlock]] = {}
        self.lock = threading.RLock()
        self.last_cleanup = time.time()

        # Initialize pools for each device
        for device, size in config.pool_sizes.items():
            self.pools[device] = {}
            logger.info(f"Initialized memory pool for {device} with {size / 1024**3:.2f}GB")

    def allocate(self, size: int, device: str) -> MemoryBlock:
        """
        Allocate memory from the appropriate pool.

        Args:
            size: Size in bytes
            device: Target device ("cpu", "cuda", etc.)

        Returns:
            MemoryBlock representing the allocation
        """
        with self.lock:
            self._check_cleanup()

            if device not in self.pools:
                self.pools[device] = {}

            # Check if we have enough space in the pool
            pool_size = self.config.pool_sizes.get(device, 0)
            used_size = sum(block.size for block in self.pools[device].values())

            if used_size + size > pool_size:
                # Try to free up space through cleanup
                self._cleanup_pool(device, size)

                # Recheck after cleanup
                used_size = sum(block.size for block in self.pools[device].values())
                if used_size + size > pool_size:
                    raise MemoryError(f"Insufficient memory in {device} pool")

            # Create memory block
            block_id = f"{device}_{len(self.pools[device])}"
            block = MemoryBlock(device=device, size=size)
            self.pools[device][block_id] = block

            logger.debug(f"Allocated {size / 1024**2:.2f}MB on {device}")
            return block

    def deallocate(self, block: MemoryBlock):
        """
        Return memory block to the pool.

        Args:
            block: MemoryBlock to deallocate
        """
        with self.lock:
            device = block.device
            if device in self.pools:
                # Find and remove the block
                for block_id, pool_block in self.pools[device].items():
                    if pool_block is block:
                        del self.pools[device][block_id]
                        logger.debug(f"Deallocated {block.size / 1024**2:.2f}MB from {device}")
                        break

    def get_pool_stats(self, device: str) -> PoolStats:
        """
        Get statistics for a specific pool.

        Args:
            device: Device to get stats for

        Returns:
            PoolStats object
        """
        with self.lock:
            if device not in self.pools:
                return PoolStats(0, 0, 0, 0.0, 0, 0)

            pool = self.pools[device]
            total_size = self.config.pool_sizes.get(device, 0)
            used_size = sum(block.size for block in pool.values())
            free_size = total_size - used_size

            # Calculate fragmentation (simplified)
            if len(pool) <= 1:
                fragmentation = 0.0
            else:
                avg_size = used_size / len(pool)
                variance = sum((block.size - avg_size) ** 2 for block in pool.values()) / len(pool)
                fragmentation = variance ** 0.5 / avg_size

            return PoolStats(
                total_size=total_size,
                used_size=used_size,
                free_size=free_size,
                fragmentation_ratio=fragmentation,
                allocation_count=len(pool),
                deallocation_count=0  # Not tracked currently
            )

    def _check_cleanup(self):
        """Check if cleanup is needed based on time interval."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.config.cleanup_interval:
            self._cleanup_all_pools()
            self.last_cleanup = current_time

    def _cleanup_pool(self, device: str, required_size: int = 0):
        """
        Clean up a specific pool to free memory.

        Args:
            device: Device pool to clean
            required_size: Minimum size to free
        """
        if device not in self.pools:
            return

        pool = self.pools[device]
        stats = self.get_pool_stats(device)

        if stats.fragmentation_ratio > self.config.fragmentation_threshold:
            # Remove oldest blocks to reduce fragmentation
            sorted_blocks = sorted(
                pool.items(),
                key=lambda x: x[1].last_accessed
            )

            freed_size = 0
            for block_id, block in sorted_blocks:
                if freed_size >= required_size:
                    break
                freed_size += block.size
                del pool[block_id]
                logger.debug(f"Cleaned up block {block_id} from {device} pool")

    def _cleanup_all_pools(self):
        """Clean up all pools."""
        for device in list(self.pools.keys()):
            self._cleanup_pool(device)


class ModelWeightCache:
    """
    LRU cache for model weights with size management.

    Caches frequently used model weights to reduce loading time and memory pressure.
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()
        self.access_counts: Dict[str, int] = {}
        self.max_size = int(config.max_size_gb * 1024**3)
        self.current_size = 0
        self.lock = threading.RLock()

    def get(self, model_key: str, layer_name: str) -> Optional[torch.Tensor]:
        """
        Retrieve weights from cache.

        Args:
            model_key: Unique identifier for the model
            layer_name: Name of the layer

        Returns:
            Cached weights tensor or None
        """
        with self.lock:
            cache_key = f"{model_key}:{layer_name}"

            if cache_key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(cache_key)
                self.access_counts[cache_key] += 1
                return self.cache[cache_key]

            return None

    def put(self, model_key: str, layer_name: str, weights: torch.Tensor):
        """
        Store weights in cache.

        Args:
            model_key: Unique identifier for the model
            layer_name: Name of the layer
            weights: Weight tensor to cache
        """
        with self.lock:
            cache_key = f"{model_key}:{layer_name}"

            # Calculate size
            size = weights.numel() * weights.element_size()

            # Check if we should cache this
            if cache_key not in self.access_counts:
                self.access_counts[cache_key] = 0

            if self.access_counts[cache_key] < self.config.access_count_threshold:
                self.access_counts[cache_key] += 1
                return  # Don't cache yet

            # Evict if necessary
            while self.current_size + size > self.max_size:
                self._evict_lru()

            # Store in cache
            if cache_key in self.cache:
                old_size = self.cache[cache_key].numel() * self.cache[cache_key].element_size()
                self.current_size -= old_size
            else:
                self.access_counts[cache_key] += 1

            self.cache[cache_key] = weights.clone()  # Clone to avoid reference issues
            self.cache.move_to_end(cache_key)
            self.current_size += size

            logger.debug(f"Cached weights for {cache_key}, cache size: {self.current_size / 1024**3:.2f}GB")

    def _evict_lru(self):
        """Evict the least recently used item."""
        if not self.cache:
            return

        # Evict in batches for efficiency
        evicted_count = 0
        evicted_size = 0

        for cache_key in list(self.cache.keys()):
            if evicted_count >= self.config.eviction_batch_size:
                break

            weights = self.cache[cache_key]
            size = weights.numel() * weights.element_size()

            del self.cache[cache_key]
            del self.access_counts[cache_key]
            self.current_size -= size
            evicted_size += size
            evicted_count += 1

        logger.debug(f"Evicted {evicted_count} items, freed {evicted_size / 1024**2:.2f}MB")

    def clear(self):
        """Clear all cached weights."""
        with self.lock:
            self.cache.clear()
            self.access_counts.clear()
            self.current_size = 0
            logger.info("Cleared model weight cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "cache_size_gb": self.current_size / 1024**3,
                "max_size_gb": self.max_size / 1024**3,
                "utilization": self.current_size / self.max_size if self.max_size > 0 else 0,
                "item_count": len(self.cache),
                "access_counts": dict(self.access_counts)
            }


class MemoryMonitor:
    """
    Real-time memory monitoring and alerting system.
    """

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.history: deque[MemoryStats] = deque(maxlen=config.history_size)
        self.lock = threading.RLock()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()

    def start_monitoring(self):
        """Start the monitoring thread."""
        if not self.config.enable_monitoring:
            return

        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Started memory monitoring")

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5.0)
            logger.info("Stopped memory monitoring")

    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory statistics.

        Returns:
            MemoryStats object with current memory information
        """
        from .memory_utils import get_system_memory_info, get_gpu_memory_info

        system_memory = get_system_memory_info()
        gpu_memory = get_gpu_memory_info() if torch.cuda.is_available() else None

        return MemoryStats(
            system_memory=system_memory,
            gpu_memory=gpu_memory
        )

    def check_memory_pressure(self) -> str:
        """
        Assess current memory pressure level.

        Returns:
            Pressure level: "low", "warning", "critical"
        """
        stats = self.get_memory_stats()

        # Check system memory
        total_sys, available_sys = stats.system_memory
        sys_usage = (total_sys - available_sys) / total_sys

        # Check GPU memory
        gpu_usage = 0.0
        if stats.gpu_memory:
            total_gpu, free_gpu = stats.gpu_memory
            gpu_usage = (total_gpu - free_gpu) / total_gpu

        max_usage = max(sys_usage, gpu_usage)

        if max_usage >= self.config.alert_thresholds["critical"]:
            return "critical"
        elif max_usage >= self.config.alert_thresholds["warning"]:
            return "warning"
        else:
            return "low"

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                stats = self.get_memory_stats()
                with self.lock:
                    self.history.append(stats)

                pressure = self.check_memory_pressure()
                if pressure != "low":
                    logger.warning(f"Memory pressure: {pressure} - {stats}")

            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")

            self.stop_monitoring.wait(self.config.monitoring_interval)

    def get_history(self, limit: int = 100) -> List[MemoryStats]:
        """Get recent memory history."""
        with self.lock:
            return list(self.history)[-limit:]


class AutomaticOffloader:
    """
    Automatic model offloading system based on memory pressure and usage patterns.
    """

    def __init__(self, memory_monitor: MemoryMonitor, config: OffloadConfig):
        self.monitor = memory_monitor
        self.config = config
        self.last_offload_time = 0
        self.usage_history: Dict[str, deque] = {}
        self.lock = threading.RLock()

    def should_offload(self, model_id: str) -> bool:
        """
        Determine if a model should be offloaded.

        Args:
            model_id: Unique identifier for the model

        Returns:
            True if offloading is recommended
        """
        if not self.config.enable_auto_offload:
            return False

        current_time = time.time()

        # Check minimum interval
        if current_time - self.last_offload_time < self.config.min_offload_interval:
            return False

        # Check memory pressure
        pressure = self.monitor.check_memory_pressure()
        if pressure != "critical":
            return False

        # Check usage patterns
        usage_score = self._calculate_usage_score(model_id)
        if usage_score > 0.3:  # Recently used
            return False

        return True

    def offload_model(self, model: torch.nn.Module, target_device: str = "cpu"):
        """
        Offload model to target device.

        Args:
            model: PyTorch model to offload
            target_device: Target device for offloading
        """
        try:
            model.to(target_device)
            self.last_offload_time = time.time()
            logger.info(f"Offloaded model to {target_device}")
        except Exception as e:
            logger.error(f"Failed to offload model: {e}")

    def restore_model(self, model: torch.nn.Module, target_device: str = "cuda"):
        """
        Restore model to target device if conditions allow.

        Args:
            model: PyTorch model to restore
            target_device: Target device for restoration
        """
        if not self.config.enable_auto_offload:
            return

        pressure = self.monitor.check_memory_pressure()
        if pressure in ["low", "warning"]:
            try:
                model.to(target_device)
                logger.info(f"Restored model to {target_device}")
            except Exception as e:
                logger.error(f"Failed to restore model: {e}")

    def _calculate_usage_score(self, model_id: str) -> float:
        """
        Calculate usage score for a model (0-1, higher means more recent usage).

        Args:
            model_id: Model identifier

        Returns:
            Usage score between 0 and 1
        """
        with self.lock:
            if model_id not in self.usage_history:
                self.usage_history[model_id] = deque(maxlen=100)

            history = self.usage_history[model_id]
            if not history:
                return 0.0

            # Simple exponential decay score
            current_time = time.time()
            total_weight = 0
            weighted_sum = 0

            for i, timestamp in enumerate(history):
                age = current_time - timestamp
                weight = 1.0 / (1.0 + age / self.config.usage_window)
                weighted_sum += weight
                total_weight += 1.0

            return weighted_sum / total_weight if total_weight > 0 else 0.0

    def record_usage(self, model_id: str):
        """
        Record model usage for pattern analysis.

        Args:
            model_id: Model identifier
        """
        with self.lock:
            if model_id not in self.usage_history:
                self.usage_history[model_id] = deque(maxlen=100)
            self.usage_history[model_id].append(time.time())


class AdvancedMemoryManager:
    """
    Main coordinator for advanced memory management features.
    """

    def __init__(self, config: AdvancedMemoryConfig):
        self.config = config

        # Initialize components
        self.pool_manager = MemoryPoolManager(config.memory)
        self.weight_cache = ModelWeightCache(config.cache)
        self.memory_monitor = MemoryMonitor(config.monitor)
        self.offloader = AutomaticOffloader(self.memory_monitor, config.offload)

        # Start monitoring
        self.memory_monitor.start_monitoring()

        logger.info("Initialized Advanced Memory Management System")

    def allocate_memory(self, size: int, device: str) -> MemoryBlock:
        """Allocate memory through the pool manager."""
        return self.pool_manager.allocate(size, device)

    def deallocate_memory(self, block: MemoryBlock):
        """Deallocate memory through the pool manager."""
        self.pool_manager.deallocate(block)

    def cache_weights(self, model_key: str, layer_name: str, weights: torch.Tensor):
        """Cache model weights."""
        self.weight_cache.put(model_key, layer_name, weights)

    def get_cached_weights(self, model_key: str, layer_name: str) -> Optional[torch.Tensor]:
        """Retrieve cached weights."""
        return self.weight_cache.get(model_key, layer_name)

    def check_offload_needed(self, model_id: str) -> bool:
        """Check if offloading is needed for a model."""
        return self.offloader.should_offload(model_id)

    def offload_model(self, model: torch.nn.Module, model_id: str, target_device: str = "cpu"):
        """Offload a model."""
        self.offloader.offload_model(model, target_device)

    def restore_model(self, model: torch.nn.Module, model_id: str, target_device: str = "cuda"):
        """Restore a model."""
        self.offloader.restore_model(model, target_device)

    def record_model_usage(self, model_id: str):
        """Record that a model was used."""
        self.offloader.record_usage(model_id)

    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory statistics."""
        stats = self.memory_monitor.get_memory_stats()
        stats.pool_stats = {
            device: self.pool_manager.get_pool_stats(device)
            for device in self.pool_manager.pools.keys()
        }
        stats.cache_stats = self.weight_cache.get_stats()
        return stats

    def cleanup(self):
        """Clean up resources."""
        self.memory_monitor.stop_monitoring()
        self.weight_cache.clear()
        logger.info("Cleaned up resources")