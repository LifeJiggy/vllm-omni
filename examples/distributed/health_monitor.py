# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import logging
import time

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


@ray.remote
class HealthMonitor:
    """Monitors node health and handles failure recovery."""

    def __init__(self, health_check_interval: float = 5.0, max_failures: int = 3):
        self.health_check_interval = health_check_interval
        self.max_failures = max_failures

        # Node tracking
        self.nodes = []  # List of node actors
        self.node_health = {}  # node_id -> health status
        self.failure_counts = {}  # node_id -> failure count
        self.last_health_check = {}  # node_id -> timestamp

        # Start health monitoring
        self.monitoring_task = None
        self._start_monitoring()

        logger.info("Initialized health monitor")

    def register_nodes(self, nodes: list):
        """Register nodes for health monitoring."""
        self.nodes = nodes
        for i, node in enumerate(nodes):
            self.node_health[i] = True
            self.failure_counts[i] = 0
            self.last_health_check[i] = time.time()

        logger.info(f"Registered {len(nodes)} nodes for health monitoring")

    def _start_monitoring(self):
        """Start background health monitoring."""
        if RAY_AVAILABLE:
            self.monitoring_task = asyncio.create_task(self._monitor_health())

    async def _monitor_health(self):
        """Background task to monitor node health."""
        while True:
            try:
                await self._check_all_nodes()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _check_all_nodes(self):
        """Check health of all registered nodes."""
        current_time = time.time()

        for node_id, node_actor in enumerate(self.nodes):
            try:
                # Ping node for health check
                health_info = await node_actor.get_health.remote()
                if health_info:
                    self.node_health[node_id] = True
                    self.failure_counts[node_id] = 0
                    self.last_health_check[node_id] = current_time
                    logger.debug(f"Node {node_id} is healthy")
                else:
                    self._handle_node_failure(node_id)

            except Exception as e:
                logger.warning(f"Health check failed for node {node_id}: {e}")
                self._handle_node_failure(node_id)

    def _handle_node_failure(self, node_id: int):
        """Handle node failure detection."""
        self.failure_counts[node_id] += 1

        if self.failure_counts[node_id] >= self.max_failures:
            self.node_health[node_id] = False
            logger.error(f"Node {node_id} marked as unhealthy after {self.failure_counts[node_id]} failures")
            # TODO: Trigger recovery mechanisms
        else:
            logger.warning(f"Node {node_id} health check failed ({self.failure_counts[node_id]}/{self.max_failures})")

    def get_healthy_nodes(self) -> list[int]:
        """Get list of healthy node IDs."""
        return [node_id for node_id, healthy in self.node_health.items() if healthy]

    def get_unhealthy_nodes(self) -> list[int]:
        """Get list of unhealthy node IDs."""
        return [node_id for node_id, healthy in self.node_health.items() if not healthy]

    def get_health_stats(self) -> dict:
        """Get comprehensive health statistics."""
        return {
            "total_nodes": len(self.nodes),
            "healthy_nodes": len(self.get_healthy_nodes()),
            "unhealthy_nodes": len(self.get_unhealthy_nodes()),
            "node_health": dict(self.node_health),
            "failure_counts": dict(self.failure_counts),
            "last_checks": dict(self.last_health_check),
        }

    def reset_node_health(self, node_id: int):
        """Reset health status for a node (after recovery)."""
        if node_id in self.node_health:
            self.node_health[node_id] = True
            self.failure_counts[node_id] = 0
            self.last_health_check[node_id] = time.time()
            logger.info(f"Reset health status for node {node_id}")

    async def wait_for_recovery(self, node_id: int, timeout: float = 30.0) -> bool:
        """Wait for a node to recover."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.node_health.get(node_id, False):
                return True
            await asyncio.sleep(1.0)
        return False