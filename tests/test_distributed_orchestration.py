# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from vllm_omni.distributed.health_monitor import HealthMonitor
from vllm_omni.distributed.load_balancer import LoadBalancer
from vllm_omni.distributed.orchestrator import DistributedScheduler


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
class TestDistributedOrchestration:
    """Test distributed inference orchestration."""

    def test_load_balancer_round_robin(self):
        """Test round-robin load balancing."""
        balancer = LoadBalancer.remote(num_nodes=3, strategy="round_robin")

        prompts = ["prompt1", "prompt2", "prompt3", "prompt4"]
        batches = ray.get(balancer.distribute_requests.remote(prompts=prompts, healthy_nodes=[0, 1, 2]))

        assert len(batches) <= 3  # Should distribute across nodes
        total_distributed = sum(len(batch["prompts"]) for batch in batches.values())
        assert total_distributed == 4

    def test_load_balancer_least_loaded(self):
        """Test least-loaded load balancing."""
        balancer = LoadBalancer.remote(num_nodes=2, strategy="least_loaded")

        # Simulate load on node 0
        ray.get(balancer.update_node_load.remote(0, 5))

        prompts = ["prompt1", "prompt2"]
        batches = ray.get(balancer.distribute_requests.remote(prompts=prompts, healthy_nodes=[0, 1]))

        # Should prefer node 1 (less loaded)
        assert 1 in batches
        assert len(batches[1]["prompts"]) >= len(batches.get(0, {"prompts": []})["prompts"])

    def test_health_monitor_node_failure(self):
        """Test health monitor failure detection."""
        monitor = HealthMonitor.remote(max_failures=2)

        # Mock node actors
        mock_nodes = [Mock(), Mock()]
        ray.get(monitor.register_nodes.remote(mock_nodes))

        # Simulate failures
        ray.get(monitor._handle_node_failure.remote(0))
        ray.get(monitor._handle_node_failure.remote(0))

        healthy = ray.get(monitor.get_healthy_nodes.remote())
        assert 0 not in healthy

    def test_distributed_scheduler_basic(self):
        """Test basic distributed scheduler functionality."""
        scheduler = DistributedScheduler.remote(model="test-model", num_nodes=2)

        stats = ray.get(scheduler.get_stats.remote())
        assert stats["num_nodes"] == 2
        assert "healthy_nodes" in stats
        assert "load_distribution" in stats

    def test_failure_recovery(self):
        """Test failure recovery mechanisms."""
        scheduler = DistributedScheduler.remote(model="test-model", num_nodes=2)

        # Simulate node failure and recovery
        ray.get(scheduler._handle_node_failure.remote(0))

        stats = ray.get(scheduler.get_stats.remote())
        assert 0 in stats["unhealthy_nodes"] or 0 not in stats["healthy_nodes"]


if __name__ == "__main__":
    pytest.main([__file__])
