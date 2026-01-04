"""
Switching-enabled Omni entrypoint that integrates model switching into the request processing pipeline.

This module provides a wrapper around the Omni class that adds model switching capabilities,
enabling seamless traffic routing during model switches.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Generator
from vllm.inputs import PromptType
from vllm.logger import init_logger

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.model_executor.models.switching_orchestrator import SwitchingOrchestrator
from vllm_omni.model_executor.models.request_router import RequestRouter, RoutingDecision
from vllm_omni.model_executor.models.model_switcher import ModelSwitcher
from vllm_omni.model_executor.models.dynamic_registry import DynamicModelRegistry
from vllm_omni.model_executor.models.model_cache import ModelCache
from vllm_omni.model_executor.models.transition_manager import TransitionManager
from vllm_omni.model_executor.models.switching_strategies import SwitchingStrategyType

logger = init_logger(__name__)


class OmniWithSwitching:
    """
    Omni entrypoint with integrated model switching capabilities.

    This class wraps the standard Omni class and adds switching functionality,
    allowing requests to be routed to appropriate model versions during switches.
    """

    def __init__(self,
                 model: str,
                 enable_switching: bool = True,
                 switching_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize Omni with switching capabilities.

        Args:
            model: Model identifier
            enable_switching: Whether to enable switching functionality
            switching_config: Configuration for switching components
            **kwargs: Additional arguments passed to Omni
        """
        self.model = model
        self.enable_switching = enable_switching

        # Initialize the underlying Omni instance
        self.omni = Omni(model, **kwargs)

        if enable_switching:
            self._initialize_switching_components(switching_config or {})
            logger.info(f"Initialized OmniWithSwitching for model {model} with switching enabled")
        else:
            self.switching_orchestrator = None
            self.request_router = None
            logger.info(f"Initialized OmniWithSwitching for model {model} with switching disabled")

    def _initialize_switching_components(self, config: Dict[str, Any]):
        """Initialize switching components."""
        # Create switching infrastructure
        # Note: In a real implementation, these would be shared across multiple Omni instances
        registry = DynamicModelRegistry()
        cache = ModelCache(max_size=config.get("cache_size", 10))
        transition_manager = TransitionManager()

        model_switcher = ModelSwitcher(
            registry=registry,
            cache=cache,
            transition_manager=transition_manager,
            max_concurrent_switches=config.get("max_concurrent_switches", 3)
        )

        self.switching_orchestrator = SwitchingOrchestrator(model_switcher)
        self.request_router = RequestRouter(self.switching_orchestrator)

        # Set as global router for convenience
        from vllm_omni.model_executor.models.request_router import set_request_router
        set_request_router(self.request_router)

    def generate(self, *args, **kwargs) -> Generator[OmniRequestOutput, None, None]:
        """
        Generate outputs with switching-aware request routing.

        If switching is enabled, requests will be routed to appropriate model versions
        based on active switching operations.
        """
        if not self.enable_switching:
            # Use standard Omni behavior
            yield from self.omni.generate(*args, **kwargs)
            return

        # Extract prompts and parameters
        prompts = args[0] if args else kwargs.get("prompts")
        sampling_params_list = args[1] if len(args) > 1 else kwargs.get("sampling_params_list")

        if prompts is None:
            if kwargs.get("prompt") is not None:
                prompts = kwargs.get("prompt")
            else:
                raise ValueError("prompts is required for generation")

        # Route requests based on switching state
        if isinstance(prompts, (list, tuple)):
            # Batch request routing
            routing_decisions = self.request_router.route_batch_request(
                self.model, [{"prompt": p} for p in prompts]
            )

            # Group by target version
            version_groups = {}
            for i, decision in enumerate(routing_decisions):
                version = decision.target_version
                if version not in version_groups:
                    version_groups[version] = []
                version_groups[version].append((i, prompts[i], decision))

            # Process each version group
            for version, requests in version_groups.items():
                group_prompts = [req[1] for req in requests]
                group_decisions = [req[2] for req in requests]

                # Generate for this version
                outputs = list(self.omni.generate(
                    prompts=group_prompts,
                    sampling_params_list=sampling_params_list,
                    **{k: v for k, v in kwargs.items() if k not in ["prompts", "prompt"]}
                ))

                # Yield outputs with routing metadata
                for i, output in enumerate(outputs):
                    original_idx, _, decision = requests[i]
                    # Add routing metadata to output
                    if hasattr(output, 'metadata'):
                        output.metadata.update({
                            "routing_decision": {
                                "target_version": decision.target_version,
                                "is_switching": decision.is_switching,
                                "switching_operation_id": decision.switching_operation_id,
                            }
                        })
                    yield output

        else:
            # Single request routing
            decision = self.request_router.route_request(self.model, {"prompt": prompts})

            # Generate with routing
            outputs = self.omni.generate(
                prompts=prompts,
                sampling_params_list=sampling_params_list,
                **{k: v for k, v in kwargs.items() if k not in ["prompts", "prompt"]}
            )

            # Add routing metadata to outputs
            for output in outputs:
                if hasattr(output, 'metadata'):
                    output.metadata.update({
                        "routing_decision": {
                            "target_version": decision.target_version,
                            "is_switching": decision.is_switching,
                            "switching_operation_id": decision.switching_operation_id,
                        }
                    })
                yield output

    async def start_switch(self,
                          target_version: str,
                          strategy_type: SwitchingStrategyType = SwitchingStrategyType.IMMEDIATE,
                          strategy_config: Optional[Dict[str, Any]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a model switching operation.

        Args:
            target_version: Target version to switch to
            strategy_type: Switching strategy to use
            strategy_config: Strategy-specific configuration
            metadata: Additional metadata

        Returns:
            Operation ID for tracking the switch
        """
        if not self.enable_switching or not self.switching_orchestrator:
            raise RuntimeError("Switching is not enabled for this Omni instance")

        return await self.switching_orchestrator.start_switch(
            model_id=self.model,
            target_version=target_version,
            strategy_type=strategy_type,
            strategy_config=strategy_config,
            metadata=metadata
        )

    def get_switching_status(self) -> Dict[str, Any]:
        """
        Get current switching status.

        Returns:
            Dictionary with switching status information
        """
        if not self.enable_switching or not self.switching_orchestrator:
            return {"enabled": False}

        active_switches = self.switching_orchestrator.get_active_switches()
        stats = self.switching_orchestrator.get_switch_statistics()

        return {
            "enabled": True,
            "active_switches": active_switches,
            "statistics": stats,
            "routing_stats": self.request_router.get_routing_stats() if self.request_router else {}
        }

    async def abort_switch(self, operation_id: str) -> bool:
        """
        Abort an active switching operation.

        Args:
            operation_id: Operation ID to abort

        Returns:
            True if aborted successfully
        """
        if not self.enable_switching or not self.switching_orchestrator:
            return False

        return await self.switching_orchestrator.abort_switch(self.model)

    def clear_routing_cache(self, model_id: Optional[str] = None):
        """
        Clear the routing cache.

        Args:
            model_id: Optional model ID to clear cache for
        """
        if self.request_router:
            self.request_router.clear_cache(model_id)

    # Delegate other methods to the underlying Omni instance
    def __getattr__(self, name):
        """Delegate attribute access to the underlying Omni instance."""
        return getattr(self.omni, name)


# Convenience function for creating switching-enabled Omni instances
def create_omni_with_switching(model: str,
                               enable_switching: bool = True,
                               **kwargs) -> OmniWithSwitching:
    """
    Create an Omni instance with switching capabilities.

    Args:
        model: Model identifier
        enable_switching: Whether to enable switching functionality
        **kwargs: Additional arguments passed to OmniWithSwitching

    Returns:
        OmniWithSwitching instance
    """
    return OmniWithSwitching(
        model=model,
        enable_switching=enable_switching,
        **kwargs
    )