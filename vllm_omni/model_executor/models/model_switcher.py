"""
Model Switcher Orchestrator for Real-time Model Switching

This module provides the main orchestrator for model switching operations,
coordinating between the registry, cache, transition manager, and switching strategies.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.model_executor.models.dynamic_registry import DynamicModelRegistry, ModelInstance
from vllm_omni.model_executor.models.model_cache import ModelCache
from vllm_omni.model_executor.models.transition_manager import TransitionManager
from vllm_omni.model_executor.models.switching_strategies import (
    SwitchingStrategy, SwitchOperation, SwitchingStrategyType, create_strategy
)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class SwitchResult:
    """Result of a model switch operation."""
    operation_id: str
    success: bool
    model_id: str
    from_version: str
    to_version: str
    strategy_type: SwitchingStrategyType
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelSwitcher:
    """
    Main orchestrator for model switching operations.

    This class coordinates between all components to provide seamless model switching
    with different strategies and comprehensive monitoring.
    """

    def __init__(self,
                 registry: DynamicModelRegistry,
                 cache: ModelCache,
                 transition_manager: TransitionManager,
                 max_concurrent_switches: int = 3):
        """
        Initialize the model switcher.

        Args:
            registry: Dynamic model registry
            cache: Model cache instance
            transition_manager: Transition manager
            max_concurrent_switches: Maximum concurrent switch operations
        """
        self.registry = registry
        self.cache = cache
        self.transition_manager = transition_manager
        self.max_concurrent_switches = max_concurrent_switches

        # Switch operation tracking
        self.active_operations: Dict[str, SwitchOperation] = {}
        self.completed_operations: List[SwitchOperation] = []
        self.max_completed_history = 100

        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_switches)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_switches, thread_name_prefix="switcher")

        # Statistics
        self.stats = {
            "total_switches": 0,
            "successful_switches": 0,
            "failed_switches": 0,
            "average_switch_duration": 0.0,
            "switches_by_strategy": {}
        }

        logger.info(f"Initialized ModelSwitcher with max_concurrent_switches={max_concurrent_switches}")

    async def switch_model(self,
                          model_id: str,
                          target_version: str,
                          strategy_type: SwitchingStrategyType = SwitchingStrategyType.IMMEDIATE,
                          strategy_config: Optional[Dict[str, Any]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> SwitchResult:
        """
        Initiate a model switch operation.

        Args:
            model_id: Model identifier
            target_version: Target version to switch to
            strategy_type: Switching strategy to use
            strategy_config: Strategy-specific configuration
            metadata: Additional metadata for the operation

        Returns:
            SwitchResult with operation details
        """
        operation_id = str(uuid.uuid4())
        strategy_config = strategy_config or {}
        metadata = metadata or {}

        # Get current active version
        current_instance = self.registry.get_active_model(model_id)
        if not current_instance:
            error_msg = f"No active model found for {model_id}"
            logger.error(error_msg)
            return SwitchResult(
                operation_id=operation_id,
                success=False,
                model_id=model_id,
                from_version="unknown",
                to_version=target_version,
                strategy_type=strategy_type,
                error_message=error_msg
            )

        from_version = current_instance.version

        # Create switch operation
        operation = SwitchOperation(
            operation_id=operation_id,
            model_id=model_id,
            from_version=from_version,
            to_version=target_version,
            strategy_type=strategy_type,
            strategy_config=strategy_config,
            metadata=metadata
        )

        # Check if target version exists
        if not self.registry.get_model_versions(model_id):
            error_msg = f"Model {model_id} not found in registry"
            operation.fail(error_msg)
            return self._create_result(operation, error_msg)

        versions = [v.version for v in self.registry.get_model_versions(model_id)]
        if target_version not in versions:
            error_msg = f"Target version {target_version} not found for model {model_id}. Available: {versions}"
            operation.fail(error_msg)
            return self._create_result(operation, error_msg)

        # Check concurrency limit
        if len(self.active_operations) >= self.max_concurrent_switches:
            error_msg = f"Maximum concurrent switches ({self.max_concurrent_switches}) reached"
            operation.fail(error_msg)
            return self._create_result(operation, error_msg)

        # Add to active operations
        self.active_operations[operation_id] = operation

        try:
            # Execute switch in background
            async with self._semaphore:
                result = await self._execute_switch(operation)

        except Exception as e:
            error_msg = f"Exception during switch: {e}"
            operation.fail(error_msg)
            logger.error(error_msg)
            result = self._create_result(operation, error_msg)

        finally:
            # Clean up active operation
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]

            # Add to completed history
            self.completed_operations.append(operation)
            if len(self.completed_operations) > self.max_completed_history:
                self.completed_operations.pop(0)

        # Update statistics
        self._update_stats(operation)

        return result

    async def _execute_switch(self, operation: SwitchOperation) -> SwitchResult:
        """
        Execute the switch operation.

        Args:
            operation: Switch operation to execute

        Returns:
            SwitchResult
        """
        try:
            # Create strategy instance
            strategy = create_strategy(operation.strategy_type, operation.strategy_config)

            # Pre-load target model if not in cache
            cache_key = f"{operation.model_id}_{operation.target_version}"
            if cache_key not in self.cache:
                logger.info(f"Pre-loading target model {cache_key}")
                self.cache.preload_model(cache_key, priority=10)

                # Give it a moment to load
                await asyncio.sleep(0.1)

            # Execute the switching strategy
            logger.info(f"Executing {operation.strategy_type.value} switch for {operation.model_id}: "
                       f"{operation.from_version} -> {operation.to_version}")

            success = await strategy.execute_switch(self, operation)

            if success:
                return self._create_result(operation)
            else:
                error_msg = f"Switch strategy failed for {operation.model_id}"
                return self._create_result(operation, error_msg)

        except Exception as e:
            error_msg = f"Exception during switch execution: {e}"
            logger.error(error_msg)
            return self._create_result(operation, error_msg)

    def _perform_immediate_switch(self, model_id: str, target_version: str) -> bool:
        """
        Perform an immediate switch to target version.

        Args:
            model_id: Model identifier
            target_version: Target version

        Returns:
            True if successful
        """
        try:
            # Get current active model
            current_instance = self.registry.get_active_model(model_id)
            if not current_instance:
                logger.error(f"No active model found for {model_id}")
                return False

            # Get target model from cache or registry
            cache_key = f"{model_id}_{target_version}"
            target_instance = self.cache.get_model(cache_key)

            if not target_instance:
                # Try to switch directly (loads if needed)
                success = self.registry.switch_model(model_id, target_version)
                if not success:
                    logger.error(f"Failed to switch {model_id} to {target_version}")
                    return False
                target_instance = self.registry.get_active_model(model_id)

            # Begin transition for seamless handover
            if target_instance:
                transition_id = self.transition_manager.begin_transition(
                    model_id, current_instance, target_instance
                )

                # For immediate switch, complete transition immediately
                # In practice, you'd wait for in-flight requests to complete
                self.transition_manager.complete_transition(transition_id)

                logger.info(f"Immediate switch completed for {model_id} to {target_version}")
                return True
            else:
                logger.error(f"Failed to get target instance for {model_id} v{target_version}")
                return False

        except Exception as e:
            logger.error(f"Exception during immediate switch: {e}")
            return False

    def _create_result(self, operation: SwitchOperation, error_message: Optional[str] = None) -> SwitchResult:
        """
        Create a SwitchResult from an operation.

        Args:
            operation: Completed operation
            error_message: Optional error message

        Returns:
            SwitchResult
        """
        return SwitchResult(
            operation_id=operation.operation_id,
            success=operation.status == "completed",
            model_id=operation.model_id,
            from_version=operation.from_version,
            to_version=operation.to_version,
            strategy_type=operation.strategy_type,
            duration_seconds=operation.duration,
            error_message=error_message,
            metadata=operation.metadata
        )

    def _update_stats(self, operation: SwitchOperation):
        """Update switcher statistics."""
        self.stats["total_switches"] += 1

        if operation.status == "completed":
            self.stats["successful_switches"] += 1
        else:
            self.stats["failed_switches"] += 1

        # Update average duration
        if operation.duration:
            total_duration = self.stats["average_switch_duration"] * (self.stats["total_switches"] - 1)
            total_duration += operation.duration
            self.stats["average_switch_duration"] = total_duration / self.stats["total_switches"]

        # Update strategy counts
        strategy_key = operation.strategy_type.value
        if strategy_key not in self.stats["switches_by_strategy"]:
            self.stats["switches_by_strategy"][strategy_key] = 0
        self.stats["switches_by_strategy"][strategy_key] += 1

    def get_switch_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a switch operation.

        Args:
            operation_id: Operation identifier

        Returns:
            Operation status dictionary or None if not found
        """
        # Check active operations
        operation = self.active_operations.get(operation_id)
        if operation:
            return {
                "operation_id": operation.operation_id,
                "model_id": operation.model_id,
                "from_version": operation.from_version,
                "to_version": operation.to_version,
                "strategy_type": operation.strategy_type.value,
                "status": operation.status,
                "progress": operation.progress,
                "created_at": operation.created_at,
                "started_at": operation.started_at,
                "metadata": operation.metadata
            }

        # Check completed operations
        for op in self.completed_operations:
            if op.operation_id == operation_id:
                return {
                    "operation_id": op.operation_id,
                    "model_id": op.model_id,
                    "from_version": op.from_version,
                    "to_version": op.to_version,
                    "strategy_type": op.strategy_type.value,
                    "status": op.status,
                    "progress": op.progress,
                    "created_at": op.created_at,
                    "started_at": op.started_at,
                    "completed_at": op.completed_at,
                    "duration": op.duration,
                    "metadata": op.metadata
                }

        return None

    def list_active_switches(self) -> List[Dict[str, Any]]:
        """
        List all active switch operations.

        Returns:
            List of active operation status dictionaries
        """
        return [self.get_switch_status(op_id) for op_id in self.active_operations.keys()]

    def abort_switch(self, operation_id: str) -> bool:
        """
        Abort an active switch operation.

        Args:
            operation_id: Operation identifier

        Returns:
            True if aborted, False if not found or already completed
        """
        operation = self.active_operations.get(operation_id)
        if not operation:
            logger.warning(f"Cannot abort operation {operation_id}: not found or already completed")
            return False

        operation.fail("Operation aborted by user")
        del self.active_operations[operation_id]
        self.completed_operations.append(operation)

        logger.info(f"Aborted switch operation {operation_id}")
        return True

    def get_switcher_stats(self) -> Dict[str, Any]:
        """
        Get switcher statistics.

        Returns:
            Dictionary with switcher statistics
        """
        success_rate = 0.0
        if self.stats["total_switches"] > 0:
            success_rate = self.stats["successful_switches"] / self.stats["total_switches"]

        return {
            **self.stats,
            "active_operations": len(self.active_operations),
            "success_rate": success_rate,
            "completed_operations_history": len(self.completed_operations)
        }

    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """
        Get information about available switching strategies.

        Returns:
            List of strategy information dictionaries
        """
        strategies = []
        for strategy_type in SwitchingStrategyType:
            try:
                strategy = create_strategy(strategy_type, {})
                strategies.append({
                    "type": strategy_type.value,
                    "name": strategy_type.value.replace("_", " ").title(),
                    "description": self._get_strategy_description(strategy_type),
                    "config_schema": strategy.get_strategy_config_schema()
                })
            except Exception as e:
                logger.warning(f"Failed to create strategy {strategy_type}: {e}")

        return strategies

    def _get_strategy_description(self, strategy_type: SwitchingStrategyType) -> str:
        """Get human-readable description for a strategy type."""
        descriptions = {
            SwitchingStrategyType.IMMEDIATE: "Instantly switch all traffic to the new model version",
            SwitchingStrategyType.GRADUAL: "Gradually increase traffic to the new version over time",
            SwitchingStrategyType.AB_TEST: "Route traffic between versions for A/B testing",
            SwitchingStrategyType.CANARY: "Start with small traffic percentage, gradually increase based on success metrics"
        }
        return descriptions.get(strategy_type, "Unknown strategy")

    def validate_switch_request(self, model_id: str, target_version: str) -> Dict[str, Any]:
        """
        Validate a switch request before execution.

        Args:
            model_id: Model identifier
            target_version: Target version

        Returns:
            Validation result dictionary
        """
        result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }

        # Check if model exists
        versions = self.registry.get_model_versions(model_id)
        if not versions:
            result["valid"] = False
            result["errors"].append(f"Model {model_id} not found")
            return result

        # Check if target version exists
        version_strings = [v.version for v in versions]
        if target_version not in version_strings:
            result["valid"] = False
            result["errors"].append(f"Version {target_version} not found for model {model_id}")
            return result

        # Check if already active
        active_instance = self.registry.get_active_model(model_id)
        if active_instance and active_instance.version == target_version:
            result["warnings"].append(f"Version {target_version} is already active for {model_id}")

        # Check cache status
        cache_key = f"{model_id}_{target_version}"
        if cache_key not in self.cache:
            result["warnings"].append(f"Target version {target_version} not in cache, will be loaded during switch")

        # Check for active switches
        active_switches = [op for op in self.active_operations.values() if op.model_id == model_id]
        if active_switches:
            result["warnings"].append(f"There are {len(active_switches)} active switch operations for {model_id}")

        return result
