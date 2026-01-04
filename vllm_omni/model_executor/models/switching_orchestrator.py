"""
Switching Orchestrator for Real-time Model Switching Integration

This module provides the orchestrator that integrates model switching functionality
into the vLLM-Omni request processing pipeline, enabling seamless traffic routing
during model switches.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

from vllm_omni.model_executor.models.model_switcher import ModelSwitcher
from vllm_omni.model_executor.models.switching_strategies import SwitchingStrategyType, SwitchOperation

logger = init_logger(__name__)


@dataclass
class SwitchingState:
    """Tracks the state of an active model switch."""

    model_id: str
    operation: SwitchOperation
    strategy_instance: Any  # The actual strategy instance
    start_time: float = field(default_factory=time.time)

    @property
    def is_active(self) -> bool:
        """Check if the switch is still active."""
        return self.operation.status in ["active", "pending"]

    @property
    def target_version(self) -> str:
        """Get the target version for this switch."""
        return self.operation.to_version

    @property
    def source_version(self) -> str:
        """Get the source version for this switch."""
        return self.operation.from_version


class SwitchingOrchestrator:
    """
    Orchestrator for integrating model switching into the request processing pipeline.

    This class manages active switching operations and routes requests to appropriate
    model versions based on the current switching state and strategy.
    """

    def __init__(self, model_switcher: ModelSwitcher):
        """
        Initialize the switching orchestrator.

        Args:
            model_switcher: The underlying model switcher instance
        """
        self.model_switcher = model_switcher
        self.active_switches: dict[str, SwitchingState] = {}
        self._lock = asyncio.Lock()

        logger.info("Initialized SwitchingOrchestrator")

    async def start_switch(
        self,
        model_id: str,
        target_version: str,
        strategy_type: SwitchingStrategyType = SwitchingStrategyType.IMMEDIATE,
        strategy_config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Start a model switching operation.

        Args:
            model_id: Model identifier
            target_version: Target version to switch to
            strategy_type: Switching strategy to use
            strategy_config: Strategy-specific configuration
            metadata: Additional metadata

        Returns:
            Operation ID for tracking the switch
        """
        async with self._lock:
            # Check if there's already an active switch for this model
            if model_id in self.active_switches:
                existing_switch = self.active_switches[model_id]
                if existing_switch.is_active:
                    logger.warning(f"Switch already active for {model_id}, aborting new switch request")
                    return existing_switch.operation.operation_id

            # Start the switch using the model switcher
            result = await self.model_switcher.switch_model(
                model_id=model_id,
                target_version=target_version,
                strategy_type=strategy_type,
                strategy_config=strategy_config,
                metadata=metadata,
            )

            if result.success:
                # Create switching state for tracking
                operation = SwitchOperation(
                    operation_id=result.operation_id,
                    model_id=model_id,
                    from_version=result.from_version,
                    to_version=result.to_version,
                    strategy_type=strategy_type,
                    strategy_config=strategy_config or {},
                    metadata=metadata or {},
                )
                operation.start()

                # Import here to avoid circular imports
                from vllm_omni.model_executor.models.switching_strategies import create_strategy

                strategy_instance = create_strategy(strategy_type, strategy_config or {})

                switching_state = SwitchingState(
                    model_id=model_id, operation=operation, strategy_instance=strategy_instance
                )

                self.active_switches[model_id] = switching_state
                logger.info(f"Started switch operation {result.operation_id} for {model_id}")
            else:
                logger.error(f"Failed to start switch for {model_id}: {result.error_message}")

            return result.operation_id

    def get_routing_decision(self, model_id: str, request_id: str) -> tuple[str, SwitchingState | None]:
        """
        Determine which model version should handle a request.

        Args:
            model_id: Model identifier
            request_id: Unique request identifier for consistent routing

        Returns:
            Tuple of (target_version, switching_state) where switching_state is None if no active switch
        """
        switching_state = self.active_switches.get(model_id)

        if not switching_state or not switching_state.is_active:
            # No active switch, use default routing
            return self._get_default_version(model_id), None

        # Use the strategy to determine routing
        try:
            target_version = switching_state.strategy_instance.get_traffic_distribution(
                switching_state.operation, request_id
            )
            return target_version, switching_state
        except Exception as e:
            logger.error(f"Error getting traffic distribution for {model_id}: {e}")
            # Fallback to source version on error
            return switching_state.source_version, switching_state

    def _get_default_version(self, model_id: str) -> str:
        """
        Get the default version for a model when no switch is active.

        Args:
            model_id: Model identifier

        Returns:
            Default version to use
        """
        # Try to get from registry
        try:
            active_model = self.model_switcher.registry.get_active_model(model_id)
            if active_model:
                return active_model.version
        except Exception as e:
            logger.warning(f"Could not get active version for {model_id}: {e}")

        # Fallback - this should be improved based on actual registry implementation
        return "default"

    async def check_switch_completion(self, model_id: str) -> bool:
        """
        Check if an active switch for a model has completed.

        Args:
            model_id: Model identifier

        Returns:
            True if switch completed and was cleaned up
        """
        async with self._lock:
            switching_state = self.active_switches.get(model_id)
            if not switching_state:
                return False

            # Check if the operation is completed
            if switching_state.operation.status == "completed":
                # Clean up the completed switch
                del self.active_switches[model_id]
                logger.info(f"Cleaned up completed switch for {model_id}")
                return True
            elif switching_state.operation.status == "failed":
                # Clean up failed switches too
                del self.active_switches[model_id]
                logger.warning(f"Cleaned up failed switch for {model_id}")
                return True

            return False

    def get_active_switches(self) -> list[dict[str, Any]]:
        """
        Get information about all active switches.

        Returns:
            List of active switch information
        """
        active_info = []
        for model_id, state in self.active_switches.items():
            if state.is_active:
                active_info.append(
                    {
                        "model_id": model_id,
                        "operation_id": state.operation.operation_id,
                        "from_version": state.source_version,
                        "to_version": state.target_version,
                        "strategy": state.operation.strategy_type.value,
                        "progress": state.operation.progress,
                        "start_time": state.start_time,
                        "status": state.operation.status,
                    }
                )
        return active_info

    async def abort_switch(self, model_id: str) -> bool:
        """
        Abort an active switch for a model.

        Args:
            model_id: Model identifier

        Returns:
            True if aborted successfully
        """
        async with self._lock:
            switching_state = self.active_switches.get(model_id)
            if not switching_state or not switching_state.is_active:
                return False

            # Abort the operation
            success = self.model_switcher.abort_switch(switching_state.operation.operation_id)
            if success:
                del self.active_switches[model_id]
                logger.info(f"Aborted switch for {model_id}")

            return success

    def get_switch_statistics(self) -> dict[str, Any]:
        """
        Get statistics about switching operations.

        Returns:
            Dictionary with switching statistics
        """
        stats = {
            "active_switches": len([s for s in self.active_switches.values() if s.is_active]),
            "total_switches": len(self.active_switches),
            "switches_by_strategy": {},
        }

        for state in self.active_switches.values():
            strategy = state.operation.strategy_type.value
            if strategy not in stats["switches_by_strategy"]:
                stats["switches_by_strategy"][strategy] = 0
            stats["switches_by_strategy"][strategy] += 1

        return stats
