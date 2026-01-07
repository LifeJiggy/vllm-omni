# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections import defaultdict
from typing import Any

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


@ray.remote
class LoadBalancer:
    """Distributes requests across available nodes using load balancing strategies."""

    def __init__(self, num_nodes: int, strategy: str = "round_robin"):
        self.num_nodes = num_nodes
        self.strategy = strategy

        # Load tracking
        self.node_loads = defaultdict(int)  # node_id -> current load
        self.node_capacities = {i: 100 for i in range(num_nodes)}  # node_id -> capacity
        self.round_robin_index = 0

        # Request history for analytics
        self.request_history = defaultdict(list)

        logger.info(f"Initialized load balancer with {num_nodes} nodes, strategy: {strategy}")

    def distribute_requests(
        self, prompts: Any, sampling_params_list: list | None = None, healthy_nodes: list[int] | None = None
    ) -> dict[int, dict[str, Any]]:
        """Distribute requests across healthy nodes."""
        if healthy_nodes is None:
            healthy_nodes = list(range(self.num_nodes))

        if not healthy_nodes:
            return {}

        # Normalize inputs
        if not isinstance(prompts, list):
            prompts = [prompts]
        if sampling_params_list and not isinstance(sampling_params_list, list):
            sampling_params_list = [sampling_params_list]

        # Distribute based on strategy
        if self.strategy == "round_robin":
            return self._distribute_round_robin(prompts, sampling_params_list, healthy_nodes)
        elif self.strategy == "least_loaded":
            return self._distribute_least_loaded(prompts, sampling_params_list, healthy_nodes)
        else:
            return self._distribute_round_robin(prompts, sampling_params_list, healthy_nodes)

    def _distribute_round_robin(
        self, prompts: list, sampling_params_list: list | None, healthy_nodes: list[int]
    ) -> dict[int, dict[str, Any]]:
        """Round-robin distribution."""
        batches = defaultdict(lambda: {"prompts": [], "sampling_params_list": []})

        for i, prompt in enumerate(prompts):
            node_id = healthy_nodes[self.round_robin_index % len(healthy_nodes)]
            batches[node_id]["prompts"].append(prompt)

            if sampling_params_list and i < len(sampling_params_list):
                batches[node_id]["sampling_params_list"].append(sampling_params_list[i])

            self.round_robin_index += 1
            self.node_loads[node_id] += 1

        return dict(batches)

    def _distribute_least_loaded(
        self, prompts: list, sampling_params_list: list | None, healthy_nodes: list[int]
    ) -> dict[int, dict[str, Any]]:
        """Distribute to least loaded nodes."""
        batches = defaultdict(lambda: {"prompts": [], "sampling_params_list": []})

        for i, prompt in enumerate(prompts):
            # Find least loaded healthy node
            node_id = min(healthy_nodes, key=lambda x: self.node_loads[x])

            batches[node_id]["prompts"].append(prompt)

            if sampling_params_list and i < len(sampling_params_list):
                batches[node_id]["sampling_params_list"].append(sampling_params_list[i])

            self.node_loads[node_id] += 1

        return dict(batches)

    def update_node_load(self, node_id: int, load_delta: int):
        """Update load for a specific node."""
        self.node_loads[node_id] += load_delta

    def get_load_stats(self) -> dict[str, Any]:
        """Get current load statistics."""
        return {
            "node_loads": dict(self.node_loads),
            "node_capacities": dict(self.node_capacities),
            "strategy": self.strategy,
            "total_requests": sum(self.node_loads.values()),
        }

    def reset_loads(self):
        """Reset all node loads (for testing or periodic reset)."""
        self.node_loads.clear()
        logger.info("Reset all node loads")
