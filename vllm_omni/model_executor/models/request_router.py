"""
Request Router for Model Switching Integration

This module provides request routing functionality that integrates with the
vLLM-Omni request processing pipeline to route requests to appropriate model
versions during switching operations.
"""

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

from vllm_omni.model_executor.models.switching_orchestrator import SwitchingOrchestrator

logger = init_logger(__name__)


@dataclass
class RoutingDecision:
    """Represents a routing decision for a request."""

    model_id: str
    target_version: str
    request_id: str
    is_switching: bool = False
    switching_operation_id: str | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RequestRouter:
    """
    Routes requests to appropriate model versions based on active switching operations.

    This class integrates with the Omni entry points to provide seamless request routing
    during model switching operations.
    """

    def __init__(self, switching_orchestrator: SwitchingOrchestrator):
        """
        Initialize the request router.

        Args:
            switching_orchestrator: The switching orchestrator instance
        """
        self.switching_orchestrator = switching_orchestrator
        self._routing_cache: dict[str, RoutingDecision] = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        self._cache_timestamps: dict[str, float] = {}

        logger.info("Initialized RequestRouter")

    def route_request(
        self, model_id: str, request_data: dict[str, Any], request_id: str | None = None
    ) -> RoutingDecision:
        """
        Route a request to the appropriate model version.

        Args:
            model_id: Model identifier
            request_data: Request data (prompts, parameters, etc.)
            request_id: Optional request ID for consistent routing

        Returns:
            RoutingDecision with target version and metadata
        """
        # Generate request ID if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Check cache first
        cache_key = f"{model_id}:{request_id}"
        if self._is_cache_valid(cache_key):
            cached_decision = self._routing_cache.get(cache_key)
            if cached_decision:
                return cached_decision

        # Get routing decision from orchestrator
        target_version, switching_state = self.switching_orchestrator.get_routing_decision(model_id, request_id)

        # Create routing decision
        decision = RoutingDecision(
            model_id=model_id,
            target_version=target_version,
            request_id=request_id,
            is_switching=switching_state is not None,
            switching_operation_id=switching_state.operation.operation_id if switching_state else None,
            metadata={
                "original_request_id": request_id,
                "routing_timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop() else None,
            },
        )

        # Add switching metadata if applicable
        if switching_state:
            decision.metadata.update(
                {
                    "switch_strategy": switching_state.operation.strategy_type.value,
                    "switch_progress": switching_state.operation.progress,
                    "from_version": switching_state.source_version,
                    "to_version": switching_state.target_version,
                }
            )

        # Cache the decision
        self._cache_decision(cache_key, decision)

        if switching_state:
            logger.debug(
                f"Routed request {request_id} for {model_id} to version {target_version} "
                f"(switching active: {switching_state.operation.operation_id})"
            )
        else:
            logger.debug(f"Routed request {request_id} for {model_id} to version {target_version}")

        return decision

    def route_batch_request(
        self, model_id: str, requests: list[dict[str, Any]], batch_id: str | None = None
    ) -> list[RoutingDecision]:
        """
        Route a batch of requests.

        Args:
            model_id: Model identifier
            requests: List of request data
            batch_id: Optional batch ID for consistent routing

        Returns:
            List of RoutingDecision objects
        """
        if batch_id is None:
            batch_id = str(uuid.uuid4())

        decisions = []
        for i, request_data in enumerate(requests):
            # Use batch_id + index for consistent routing within batch
            request_id = f"{batch_id}_{i}"
            decision = self.route_request(model_id, request_data, request_id)
            decisions.append(decision)

        return decisions

    async def check_switch_completions(self):
        """
        Check for completed switches and clean them up.
        This should be called periodically to maintain clean state.
        """
        # Get all models that might have active switches
        active_models = list(self.switching_orchestrator.active_switches.keys())

        for model_id in active_models:
            completed = await self.switching_orchestrator.check_switch_completion(model_id)
            if completed:
                # Clear cache entries for this model
                self._clear_model_cache(model_id)
                logger.info(f"Cleared routing cache for completed switch on {model_id}")

    def get_routing_stats(self) -> dict[str, Any]:
        """
        Get routing statistics.

        Returns:
            Dictionary with routing statistics
        """
        return {
            "cache_size": len(self._routing_cache),
            "active_switches": len(self.switching_orchestrator.get_active_switches()),
            "cache_ttl": self._cache_ttl,
        }

    def clear_cache(self, model_id: str | None = None):
        """
        Clear the routing cache.

        Args:
            model_id: Optional model ID to clear cache for specific model
        """
        if model_id:
            self._clear_model_cache(model_id)
        else:
            self._routing_cache.clear()
            self._cache_timestamps.clear()
            logger.info("Cleared all routing cache")

    def _clear_model_cache(self, model_id: str):
        """Clear cache entries for a specific model."""
        keys_to_remove = [k for k in self._routing_cache.keys() if k.startswith(f"{model_id}:")]
        for key in keys_to_remove:
            del self._routing_cache[key]
            del self._cache_timestamps[key]

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cache entry is still valid."""
        if cache_key not in self._cache_timestamps:
            return False

        import time

        current_time = time.time()
        cache_time = self._cache_timestamps[cache_key]

        return (current_time - cache_time) < self._cache_ttl

    def _cache_decision(self, cache_key: str, decision: RoutingDecision):
        """Cache a routing decision."""
        import time

        self._routing_cache[cache_key] = decision
        self._cache_timestamps[cache_key] = time.time()

        # Periodic cleanup of expired entries
        if len(self._routing_cache) > 1000:  # Arbitrary threshold
            self._cleanup_expired_cache()

    def _cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        import time

        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items() if (current_time - timestamp) >= self._cache_ttl
        ]

        for key in expired_keys:
            self._routing_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired routing cache entries")


# Global router instance - would be initialized by the main application
_global_router: RequestRouter | None = None


def get_request_router() -> RequestRouter | None:
    """Get the global request router instance."""
    return _global_router


def set_request_router(router: RequestRouter):
    """Set the global request router instance."""
    global _global_router
    _global_router = router


def route_request(model_id: str, request_data: dict[str, Any], request_id: str | None = None) -> RoutingDecision:
    """
    Convenience function to route a request using the global router.

    Args:
        model_id: Model identifier
        request_data: Request data
        request_id: Optional request ID

    Returns:
        RoutingDecision

    Raises:
        RuntimeError: If no global router is set
    """
    router = get_request_router()
    if not router:
        raise RuntimeError("No global request router configured")

    return router.route_request(model_id, request_data, request_id)
