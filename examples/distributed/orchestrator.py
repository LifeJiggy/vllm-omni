# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import logging
import time
from typing import Any

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False

from vllm_omni.outputs import OmniRequestOutput
from .health_monitor import HealthMonitor
from .load_balancer import LoadBalancer

logger = logging.getLogger(__name__)


class DistributedOrchestrator:
    """Coordinates inference across multiple nodes with load balancing and failure recovery."""

    def __init__(self, model: str, num_nodes: int = 2, ray_address: str | None = None, **kwargs):
        if not RAY_AVAILABLE:
            raise ImportError("Ray is required for distributed orchestration")

        self.model = model
        self.num_nodes = num_nodes
        self.ray_address = ray_address
        self.kwargs = kwargs

        # Initialize Ray cluster
        if not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True)

        # Create distributed scheduler
        self.scheduler = DistributedScheduler.remote(model=model, num_nodes=num_nodes, **kwargs)

        logger.info(f"Initialized distributed orchestrator with {num_nodes} nodes")

    async def generate(
        self, prompts: Any, sampling_params_list: list | None = None, **kwargs
    ) -> list[OmniRequestOutput]:
        """Generate outputs using distributed orchestration."""
        return await self.scheduler.generate.remote(
            prompts=prompts, sampling_params_list=sampling_params_list, **kwargs
        )

    def close(self):
        """Clean up distributed resources."""
        if hasattr(self, "scheduler"):
            ray.kill(self.scheduler)
        if ray.is_initialized():
            ray.shutdown()


@ray.remote
class DistributedScheduler:
    """Ray actor managing distributed inference across multiple nodes."""

    def __init__(self, model: str, num_nodes: int, **kwargs):
        self.model = model
        self.num_nodes = num_nodes
        self.kwargs = kwargs

        # Initialize components
        self.load_balancer = LoadBalancer.remote(num_nodes)
        self.health_monitor = HealthMonitor.remote()

        # Create node actors
        self.node_actors = []
        for i in range(num_nodes):
            node = NodeWorker.remote(node_id=i, model=model, **kwargs)
            self.node_actors.append(node)

        # Register nodes with health monitor
        self.health_monitor.register_nodes.remote(self.node_actors)

        logger.info(f"Created distributed scheduler with {num_nodes} nodes")

    async def generate(self, prompts, sampling_params_list=None, **kwargs):
        """Distribute generation requests across healthy nodes with failure recovery."""
        max_retries = kwargs.get("max_retries", 2)

        for attempt in range(max_retries + 1):
            try:
                # Get healthy nodes
                healthy_nodes = await self.health_monitor.get_healthy_nodes.remote()

                if not healthy_nodes:
                    if attempt < max_retries:
                        logger.warning(f"No healthy nodes available, attempt {attempt + 1}/{max_retries + 1}")
                        await asyncio.sleep(2**attempt)  # Exponential backoff
                        continue
                    raise RuntimeError("No healthy nodes available after retries")

                # Distribute requests using load balancer
                request_batches = await self.load_balancer.distribute_requests.remote(
                    prompts=prompts, sampling_params_list=sampling_params_list, healthy_nodes=healthy_nodes
                )

                # Execute on nodes with failure handling
                results = []
                failed_batches = []

                for node_id, batch in request_batches.items():
                    if batch:
                        try:
                            node_actor = self.node_actors[node_id]
                            batch_result = await node_actor.process_batch.remote(batch)
                            results.extend(batch_result)
                        except Exception as e:
                            logger.error(f"Node {node_id} failed to process batch: {e}")
                            failed_batches.append((node_id, batch))
                            # Mark node as potentially unhealthy
                            await self._handle_node_failure(node_id)

                # Retry failed batches on remaining healthy nodes
                if failed_batches:
                    await self._retry_failed_batches(failed_batches, results)

                return results

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(2**attempt)
                else:
                    raise e

        return []

    async def _handle_node_failure(self, node_id: int):
        """Handle node failure by marking as unhealthy."""
        # The health monitor will handle the actual marking
        # Here we could trigger recovery mechanisms
        logger.warning(f"Handling failure for node {node_id}")

    async def _retry_failed_batches(self, failed_batches, results):
        """Retry failed batches on remaining healthy nodes."""
        if not failed_batches:
            return

        healthy_nodes = await self.health_monitor.get_healthy_nodes.remote()
        if not healthy_nodes:
            logger.error("No healthy nodes available for retry")
            return

        logger.info(f"Retrying {len(failed_batches)} failed batches on {len(healthy_nodes)} healthy nodes")

        # Redistribute failed batches
        all_failed_prompts = []
        all_failed_params = []

        for _, batch in failed_batches:
            all_failed_prompts.extend(batch.get("prompts", []))
            all_failed_params.extend(batch.get("sampling_params_list", []))

        if all_failed_prompts:
            retry_batches = await self.load_balancer.distribute_requests.remote(
                prompts=all_failed_prompts, sampling_params_list=all_failed_params, healthy_nodes=healthy_nodes
            )

            # Execute retry batches
            for node_id, batch in retry_batches.items():
                if batch:
                    try:
                        node_actor = self.node_actors[node_id]
                        batch_result = await node_actor.process_batch.remote(batch)
                        results.extend(batch_result)
                        logger.info(f"Successfully retried batch on node {node_id}")
                    except Exception as e:
                        logger.error(f"Retry also failed on node {node_id}: {e}")

    def get_stats(self):
        """Get cluster statistics."""
        return {
            "num_nodes": self.num_nodes,
            "healthy_nodes": ray.get(self.health_monitor.get_healthy_nodes.remote()),
            "unhealthy_nodes": ray.get(self.health_monitor.get_unhealthy_nodes.remote()),
            "load_distribution": ray.get(self.load_balancer.get_load_stats.remote()),
            "health_stats": ray.get(self.health_monitor.get_health_stats.remote()),
        }


@ray.remote
class NodeWorker:
    """Individual node worker running Omni inference."""

    def __init__(self, node_id: int, model: str, **kwargs):
        self.node_id = node_id
        self.model = model
        self.kwargs = kwargs

        # Initialize Omni instance for this node
        from vllm_omni.entrypoints.omni import Omni

        self.omni = Omni(model=model, **kwargs)

        logger.info(f"Node {node_id} initialized with model {model}")

    def process_batch(self, batch):
        """Process a batch of requests."""
        prompts = batch.get("prompts", [])
        sampling_params_list = batch.get("sampling_params_list", [])

        results = []
        for output in self.omni.generate(prompts=prompts, sampling_params_list=sampling_params_list):
            results.append(output)

        return results

    def get_health(self):
        """Check node health."""
        return {"node_id": self.node_id, "status": "healthy", "timestamp": time.time()}