# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Comprehensive tests for Advanced Memory Management System
"""

import pytest
import torch
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from vllm_omni.diffusion.utils.advanced_memory_management import (
    AdvancedMemoryConfig,
    MemoryConfig,
    CacheConfig,
    OffloadConfig,
    MonitorConfig,
    MemoryPoolManager,
    ModelWeightCache,
    AutomaticOffloader,
    MemoryMonitor,
    AdvancedMemoryManager,
    MemoryBlock,
    PoolStats,
    MemoryStats,
)


class TestMemoryPoolManager:
    """Test cases for MemoryPoolManager."""

    def test_initialization(self):
        """Test pool manager initialization."""
        config = MemoryConfig(pool_sizes={"cpu": 1024**3, "cuda": 2*1024**3})
        manager = MemoryPoolManager(config)

        assert "cpu" in manager.pools
        assert "cuda" in manager.pools
        assert len(manager.pools["cpu"]) == 0
        assert len(manager.pools["cuda"]) == 0

    def test_allocate_basic(self):
        """Test basic memory allocation."""
        config = MemoryConfig(pool_sizes={"cpu": 1024**3})
        manager = MemoryPoolManager(config)

        block = manager.allocate(1024, "cpu")

        assert block.device == "cpu"
        assert block.size == 1024
        assert len(manager.pools["cpu"]) == 1

    def test_allocate_insufficient_memory(self):
        """Test allocation when pool is full."""
        config = MemoryConfig(pool_sizes={"cpu": 1024})  # 1KB pool
        manager = MemoryPoolManager(config)

        # First allocation should succeed
        manager.allocate(512, "cpu")

        # Second allocation should fail
        with pytest.raises(MemoryError):
            manager.allocate(600, "cpu")

    def test_deallocate(self):
        """Test memory deallocation."""
        config = MemoryConfig(pool_sizes={"cpu": 1024**3})
        manager = MemoryPoolManager(config)

        block = manager.allocate(1024, "cpu")
        assert len(manager.pools["cpu"]) == 1

        manager.deallocate(block)
        assert len(manager.pools["cpu"]) == 0

    def test_get_pool_stats(self):
        """Test pool statistics retrieval."""
        config = MemoryConfig(pool_sizes={"cpu": 2048})
        manager = MemoryPoolManager(config)

        # Allocate some memory
        manager.allocate(1024, "cpu")

        stats = manager.get_pool_stats("cpu")

        assert stats.total_size == 2048
        assert stats.used_size == 1024
        assert stats.free_size == 1024
        assert stats.allocation_count == 1

    def test_pool_cleanup(self):
        """Test automatic pool cleanup."""
        config = MemoryConfig(
            pool_sizes={"cpu": 2048},
            fragmentation_threshold=0.0,  # Force cleanup
            cleanup_interval=0.1
        )
        manager = MemoryPoolManager(config)

        # Allocate multiple blocks
        blocks = []
        for i in range(3):
            block = manager.allocate(512, "cpu")
            blocks.append(block)

        # Force cleanup check
        time.sleep(0.2)
        manager.allocate(1, "cpu")  # Trigger cleanup

        # Should have cleaned up some blocks
        assert len(manager.pools["cpu"]) <= 3


class TestModelWeightCache:
    """Test cases for ModelWeightCache."""

    def test_initialization(self):
        """Test cache initialization."""
        config = CacheConfig(max_size_gb=1.0)
        cache = ModelWeightCache(config)

        assert cache.max_size == 1024**3
        assert cache.current_size == 0
        assert len(cache.cache) == 0

    def test_put_and_get(self):
        """Test basic cache put and get operations."""
        config = CacheConfig(max_size_gb=1.0, access_count_threshold=1)
        cache = ModelWeightCache(config)

        # Create test tensor
        weights = torch.randn(10, 10)

        # Put in cache
        cache.put("model1", "layer1", weights)

        # Get from cache
        cached_weights = cache.get("model1", "layer1")

        assert cached_weights is not None
        assert torch.equal(cached_weights, weights)

    def test_cache_miss(self):
        """Test cache miss behavior."""
        config = CacheConfig(max_size_gb=1.0)
        cache = ModelWeightCache(config)

        result = cache.get("nonexistent", "layer1")
        assert result is None

    def test_access_count_threshold(self):
        """Test access count threshold before caching."""
        config = CacheConfig(max_size_gb=1.0, access_count_threshold=3)
        cache = ModelWeightCache(config)

        weights = torch.randn(10, 10)

        # First two accesses shouldn't cache
        cache.put("model1", "layer1", weights)
        assert len(cache.cache) == 0

        cache.put("model1", "layer1", weights)
        assert len(cache.cache) == 0

        # Third access should cache
        cache.put("model1", "layer1", weights)
        assert len(cache.cache) == 1

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        config = CacheConfig(max_size_gb=0.001, eviction_batch_size=1)  # Very small cache
        cache = ModelWeightCache(config)

        weights1 = torch.randn(100, 100)  # Large tensor
        weights2 = torch.randn(100, 100)  # Large tensor

        # Add first item
        cache.put("model1", "layer1", weights1)
        assert len(cache.cache) == 1

        # Add second item (should evict first)
        cache.put("model2", "layer2", weights2)
        assert len(cache.cache) == 1
        assert "model1:layer1" not in cache.cache
        assert "model2:layer2" in cache.cache

    def test_get_stats(self):
        """Test cache statistics."""
        config = CacheConfig(max_size_gb=1.0)
        cache = ModelWeightCache(config)

        weights = torch.randn(10, 10)
        cache.put("model1", "layer1", weights)

        stats = cache.get_stats()

        assert stats["item_count"] == 1
        assert stats["cache_size_gb"] > 0
        assert stats["utilization"] > 0

    def test_clear_cache(self):
        """Test cache clearing."""
        config = CacheConfig(max_size_gb=1.0)
        cache = ModelWeightCache(config)

        weights = torch.randn(10, 10)
        cache.put("model1", "layer1", weights)
        assert len(cache.cache) == 1

        cache.clear()
        assert len(cache.cache) == 0
        assert cache.current_size == 0


class TestMemoryMonitor:
    """Test cases for MemoryMonitor."""

    @patch('vllm_omni.diffusion.utils.advanced_memory_management.get_system_memory_info')
    @patch('vllm_omni.diffusion.utils.advanced_memory_management.get_gpu_memory_info')
    def test_get_memory_stats(self, mock_gpu, mock_sys):
        """Test memory statistics retrieval."""
        mock_sys.return_value = (16*1024**3, 8*1024**3)  # 16GB total, 8GB available
        mock_gpu.return_value = (8*1024**3, 6*1024**3)   # 8GB total, 6GB free

        config = MonitorConfig()
        monitor = MemoryMonitor(config)

        stats = monitor.get_memory_stats()

        assert stats.system_memory == (16*1024**3, 8*1024**3)
        assert stats.gpu_memory == (8*1024**3, 6*1024**3)

    def test_check_memory_pressure(self):
        """Test memory pressure assessment."""
        config = MonitorConfig()
        monitor = MemoryMonitor(config)

        # Mock memory stats
        with patch.object(monitor, 'get_memory_stats') as mock_stats:
            # Low pressure
            mock_stats.return_value = MemoryStats(
                system_memory=(16*1024**3, 12*1024**3),  # 25% usage
                gpu_memory=(8*1024**3, 6*1024**3)        # 25% usage
            )
            assert monitor.check_memory_pressure() == "low"

            # Warning pressure
            mock_stats.return_value = MemoryStats(
                system_memory=(16*1024**3, 4*1024**3),   # 75% usage
                gpu_memory=(8*1024**3, 6*1024**3)        # 25% usage
            )
            assert monitor.check_memory_pressure() == "warning"

            # Critical pressure
            mock_stats.return_value = MemoryStats(
                system_memory=(16*1024**3, 2*1024**3),   # 87.5% usage
                gpu_memory=(8*1024**3, 6*1024**3)        # 25% usage
            )
            assert monitor.check_memory_pressure() == "critical"

    def test_monitoring_thread(self):
        """Test monitoring thread lifecycle."""
        config = MonitorConfig(enable_monitoring=True, monitoring_interval=0.1)
        monitor = MemoryMonitor(config)

        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring_thread is not None
        assert monitor.monitoring_thread.is_alive()

        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.monitoring_thread.is_alive()

    def test_history_management(self):
        """Test memory history collection."""
        config = MonitorConfig(history_size=5)
        monitor = MemoryMonitor(config)

        # Add some history
        for i in range(10):
            stats = MemoryStats(system_memory=(16*1024**3, 8*1024**3))
            monitor.history.append(stats)

        # Should only keep last 5
        assert len(monitor.history) == 5

        # Test history retrieval
        history = monitor.get_history(limit=3)
        assert len(history) == 3


class TestAutomaticOffloader:
    """Test cases for AutomaticOffloader."""

    def test_should_offload_disabled(self):
        """Test offloading when disabled."""
        monitor = Mock()
        config = OffloadConfig(enable_auto_offload=False)
        offloader = AutomaticOffloader(monitor, config)

        assert not offloader.should_offload("model1")

    def test_should_offload_memory_pressure(self):
        """Test offloading based on memory pressure."""
        monitor = Mock()
        config = OffloadConfig(enable_auto_offload=True)
        offloader = AutomaticOffloader(monitor, config)

        # Low pressure - should not offload
        monitor.check_memory_pressure.return_value = "low"
        assert not offloader.should_offload("model1")

        # Critical pressure - should offload
        monitor.check_memory_pressure.return_value = "critical"
        assert offloader.should_offload("model1")

    def test_should_offload_usage_patterns(self):
        """Test offloading based on usage patterns."""
        monitor = Mock()
        monitor.check_memory_pressure.return_value = "critical"
        config = OffloadConfig(enable_auto_offload=True, usage_window=1.0)
        offloader = AutomaticOffloader(monitor, config)

        # Record recent usage
        offloader.record_usage("model1")
        time.sleep(0.1)  # Small delay

        # Should not offload recently used model
        assert not offloader.should_offload("model1")

    def test_offload_model(self):
        """Test model offloading."""
        monitor = Mock()
        config = OffloadConfig()
        offloader = AutomaticOffloader(monitor, config)

        model = Mock()
        offloader.offload_model(model, "cpu")

        model.to.assert_called_once_with("cpu")
        assert offloader.last_offload_time > 0

    def test_restore_model(self):
        """Test model restoration."""
        monitor = Mock()
        monitor.check_memory_pressure.return_value = "low"
        config = OffloadConfig(enable_auto_offload=True)
        offloader = AutomaticOffloader(monitor, config)

        model = Mock()
        offloader.restore_model(model, "cuda")

        model.to.assert_called_once_with("cuda")

    def test_usage_score_calculation(self):
        """Test usage score calculation."""
        monitor = Mock()
        config = OffloadConfig(usage_window=10.0)
        offloader = AutomaticOffloader(monitor, config)

        # No usage history
        score = offloader._calculate_usage_score("model1")
        assert score == 0.0

        # Add usage history
        offloader.record_usage("model1")
        time.sleep(0.1)
        offloader.record_usage("model1")

        score = offloader._calculate_usage_score("model1")
        assert score > 0.0


class TestAdvancedMemoryManager:
    """Test cases for AdvancedMemoryManager."""

    def test_initialization(self):
        """Test manager initialization."""
        config = AdvancedMemoryConfig()
        manager = AdvancedMemoryManager(config)

        assert manager.pool_manager is not None
        assert manager.weight_cache is not None
        assert manager.memory_monitor is not None
        assert manager.offloader is not None

    def test_memory_operations(self):
        """Test memory allocation/deallocation through manager."""
        config = AdvancedMemoryConfig()
        manager = AdvancedMemoryManager(config)

        # Allocate memory
        block = manager.allocate_memory(1024, "cpu")
        assert block.device == "cpu"
        assert block.size == 1024

        # Deallocate memory
        manager.deallocate_memory(block)

    def test_weight_caching(self):
        """Test weight caching through manager."""
        config = AdvancedMemoryConfig()
        manager = AdvancedMemoryManager(config)

        weights = torch.randn(10, 10)

        # Cache weights
        manager.cache_weights("model1", "layer1", weights)

        # Retrieve weights
        cached = manager.get_cached_weights("model1", "layer1")
        assert cached is not None
        assert torch.equal(cached, weights)

    def test_offloading_operations(self):
        """Test offloading operations through manager."""
        config = AdvancedMemoryConfig()
        manager = AdvancedMemoryManager(config)

        model = Mock()

        # Test offloading
        manager.offload_model(model, "model1", "cpu")
        model.to.assert_called_once_with("cpu")

        # Test restoration
        with patch.object(manager.memory_monitor, 'check_memory_pressure', return_value="low"):
            manager.restore_model(model, "model1", "cuda")
            assert model.to.call_count == 2

    def test_usage_tracking(self):
        """Test model usage tracking."""
        config = AdvancedMemoryConfig()
        manager = AdvancedMemoryManager(config)

        manager.record_model_usage("model1")

        # Should be recorded in offloader
        assert "model1" in manager.offloader.usage_history

    def test_memory_stats_aggregation(self):
        """Test memory statistics aggregation."""
        config = AdvancedMemoryConfig()
        manager = AdvancedMemoryManager(config)

        stats = manager.get_memory_stats()

        assert stats.system_memory is not None
        assert "pool_stats" in stats.__dict__
        assert "cache_stats" in stats.__dict__

    def test_cleanup(self):
        """Test resource cleanup."""
        config = AdvancedMemoryConfig()
        manager = AdvancedMemoryManager(config)

        # Add some data
        weights = torch.randn(10, 10)
        manager.cache_weights("model1", "layer1", weights)

        assert len(manager.weight_cache.cache) == 1

        # Cleanup
        manager.cleanup()

        assert len(manager.weight_cache.cache) == 0


class TestIntegration:
    """Integration tests for the complete system."""

    def test_full_memory_workflow(self):
        """Test complete memory management workflow."""
        config = AdvancedMemoryConfig()
        manager = AdvancedMemoryManager(config)

        # Allocate memory
        block = manager.allocate_memory(1024, "cpu")

        # Cache some weights
        weights = torch.randn(5, 5)
        manager.cache_weights("test_model", "test_layer", weights)

        # Record usage
        manager.record_model_usage("test_model")

        # Get stats
        stats = manager.get_memory_stats()
        assert stats is not None

        # Cleanup
        manager.deallocate_memory(block)
        manager.cleanup()

    def test_concurrent_access(self):
        """Test concurrent access to memory manager."""
        config = AdvancedMemoryConfig()
        manager = AdvancedMemoryManager(config)

        results = []

        def worker(worker_id):
            # Allocate memory
            block = manager.allocate_memory(512, "cpu")
            time.sleep(0.01)

            # Cache weights
            weights = torch.randn(3, 3)
            manager.cache_weights(f"model_{worker_id}", "layer", weights)

            # Record usage
            manager.record_model_usage(f"model_{worker_id}")

            results.append(f"worker_{worker_id}_done")

            # Cleanup
            manager.deallocate_memory(block)

        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        assert len(results) == 5
        assert len(manager.weight_cache.cache) == 5

    def test_memory_pressure_response(self):
        """Test system response to memory pressure."""
        config = AdvancedMemoryConfig()
        manager = AdvancedMemoryManager(config)

        model = Mock()

        # Simulate memory pressure
        with patch.object(manager.memory_monitor, 'check_memory_pressure', return_value="critical"):
            should_offload = manager.check_offload_needed("test_model")
            assert should_offload

            manager.offload_model(model, "test_model", "cpu")
            model.to.assert_called_once_with("cpu")

        # Simulate low pressure
        with patch.object(manager.memory_monitor, 'check_memory_pressure', return_value="low"):
            manager.restore_model(model, "test_model", "cuda")
            assert model.to.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__])