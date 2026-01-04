"""
Switching Strategies for Real-time Model Switching

This module provides different strategies for model switching including immediate,
gradual rollout, A/B testing, and canary deployments.
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from vllm.logger import init_logger

logger = init_logger(__name__)


class SwitchingStrategyType(Enum):
    """Types of switching strategies."""
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    AB_TEST = "ab_test"
    CANARY = "canary"


@dataclass
class SwitchOperation:
    """Represents a model switch operation."""
    operation_id: str
    model_id: str
    from_version: str
    to_version: str
    strategy_type: SwitchingStrategyType
    strategy_config: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"  # pending, active, completed, failed
    progress: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Get operation duration if completed."""
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        return None

    def start(self):
        """Mark operation as started."""
        self.started_at = time.time()
        self.status = "active"

    def complete(self):
        """Mark operation as completed."""
        self.completed_at = time.time()
        self.status = "completed"
        self.progress = 1.0

    def fail(self, reason: str):
        """Mark operation as failed."""
        self.completed_at = time.time()
        self.status = "failed"
        self.metadata["failure_reason"] = reason

    def update_progress(self, progress: float):
        """Update operation progress."""
        self.progress = max(0.0, min(1.0, progress))


class SwitchingStrategy(ABC):
    """
    Abstract base class for switching strategies.

    Strategies define how traffic is gradually shifted from one model version
    to another during a switch operation.
    """

    @abstractmethod
    async def execute_switch(self, switcher: 'ModelSwitcher', operation: SwitchOperation) -> bool:
        """
        Execute the switching strategy.

        Args:
            switcher: Model switcher instance
            operation: Switch operation to execute

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def get_traffic_distribution(self, operation: SwitchOperation, request_id: str) -> str:
        """
        Determine which model version should handle a request.

        Args:
            operation: Switch operation
            request_id: Request identifier for consistent routing

        Returns:
            Model version to route request to
        """
        pass

    @abstractmethod
    def get_strategy_config_schema(self) -> Dict[str, Any]:
        """
        Get configuration schema for this strategy.

        Returns:
            JSON schema for strategy configuration
        """
        pass


class ImmediateSwitch(SwitchingStrategy):
    """
    Immediate switching strategy.

    Instantly switches all traffic to the new model version.
    This is the simplest and fastest strategy but provides no gradual rollout.
    """

    async def execute_switch(self, switcher: 'ModelSwitcher', operation: SwitchOperation) -> bool:
        """
        Execute immediate switch.

        Args:
            switcher: Model switcher instance
            operation: Switch operation

        Returns:
            True if successful
        """
        try:
            operation.start()

            # Immediately switch to new version
            success = switcher._perform_immediate_switch(
                operation.model_id,
                operation.to_version
            )

            if success:
                operation.complete()
                logger.info(f"Immediate switch completed for {operation.model_id}: "
                           f"{operation.from_version} -> {operation.to_version}")
            else:
                operation.fail("Immediate switch failed")
                logger.error(f"Immediate switch failed for {operation.model_id}")

            return success

        except Exception as e:
            operation.fail(f"Exception during immediate switch: {e}")
            logger.error(f"Exception during immediate switch: {e}")
            return False

    def get_traffic_distribution(self, operation: SwitchOperation, request_id: str) -> str:
        """
        Route all traffic to new version after switch is complete.

        Args:
            operation: Switch operation
            request_id: Request identifier

        Returns:
            Target version
        """
        if operation.status == "completed":
            return operation.to_version
        else:
            return operation.from_version

    def get_strategy_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema."""
        return {
            "type": "object",
            "properties": {},
            "description": "Immediate switching - no configuration needed"
        }


class GradualRollout(SwitchingStrategy):
    """
    Gradual rollout strategy.

    Gradually increases traffic to the new model version over time
    according to a specified schedule.
    """

    async def execute_switch(self, switcher: 'ModelSwitcher', operation: SwitchOperation) -> bool:
        """
        Execute gradual rollout.

        Args:
            switcher: Model switcher instance
            operation: Switch operation

        Returns:
            True if successful
        """
        try:
            operation.start()

            config = operation.strategy_config
            duration_minutes = config.get("duration_minutes", 10)
            steps = config.get("steps", 10)
            step_duration = (duration_minutes * 60) / steps

            logger.info(f"Starting gradual rollout for {operation.model_id} over {duration_minutes} minutes")

            for step in range(steps + 1):
                progress = step / steps
                operation.update_progress(progress)

                # Update traffic distribution (this would be used by the routing logic)
                operation.metadata["current_traffic_percentage"] = progress * 100

                if step < steps:
                    await asyncio.sleep(step_duration)

            # Complete the switch
            success = switcher._perform_immediate_switch(
                operation.model_id,
                operation.to_version
            )

            if success:
                operation.complete()
                logger.info(f"Gradual rollout completed for {operation.model_id}")
            else:
                operation.fail("Gradual rollout failed at completion")
                logger.error(f"Gradual rollout failed for {operation.model_id}")

            return success

        except Exception as e:
            operation.fail(f"Exception during gradual rollout: {e}")
            logger.error(f"Exception during gradual rollout: {e}")
            return False

    def get_traffic_distribution(self, operation: SwitchOperation, request_id: str) -> str:
        """
        Route traffic based on current rollout progress.

        Args:
            operation: Switch operation
            request_id: Request identifier

        Returns:
            Model version to route to
        """
        if operation.status == "completed":
            return operation.to_version

        # Use progress to determine traffic distribution
        traffic_percentage = operation.metadata.get("current_traffic_percentage", 0)

        # Simple hash-based routing for consistency
        # In production, you'd want more sophisticated routing
        request_hash = hash(request_id) % 100
        if request_hash < traffic_percentage:
            return operation.to_version
        else:
            return operation.from_version

    def get_strategy_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema."""
        return {
            "type": "object",
            "properties": {
                "duration_minutes": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 1440,  # 24 hours
                    "default": 10,
                    "description": "Total duration of rollout in minutes"
                },
                "steps": {
                    "type": "integer",
                    "minimum": 2,
                    "maximum": 100,
                    "default": 10,
                    "description": "Number of rollout steps"
                }
            },
            "required": []
        }


class ABTestSwitch(SwitchingStrategy):
    """
    A/B testing strategy.

    Routes traffic between two model versions based on experiment configuration,
    allowing comparison of model performance.
    """

    async def execute_switch(self, switcher: 'ModelSwitcher', operation: SwitchOperation) -> bool:
        """
        Execute A/B test setup.

        Args:
            switcher: Model switcher instance
            operation: Switch operation

        Returns:
            True if successful
        """
        try:
            operation.start()

            config = operation.strategy_config
            test_duration_hours = config.get("test_duration_hours", 24)
            traffic_percentage = config.get("traffic_percentage", 50)

            operation.metadata.update({
                "test_duration_hours": test_duration_hours,
                "traffic_percentage": traffic_percentage,
                "start_time": time.time()
            })

            logger.info(f"Starting A/B test for {operation.model_id}: "
                       f"{traffic_percentage}% traffic to {operation.to_version}")

            # Wait for test duration
            await asyncio.sleep(test_duration_hours * 3600)

            # Test completed - could implement automatic winner selection here
            operation.complete()
            logger.info(f"A/B test completed for {operation.model_id}")

            return True

        except Exception as e:
            operation.fail(f"Exception during A/B test: {e}")
            logger.error(f"Exception during A/B test: {e}")
            return False

    def get_traffic_distribution(self, operation: SwitchOperation, request_id: str) -> str:
        """
        Route traffic based on A/B test configuration.

        Args:
            operation: Switch operation
            request_id: Request identifier

        Returns:
            Model version to route to
        """
        if operation.status != "active":
            return operation.from_version

        config = operation.strategy_config
        traffic_percentage = config.get("traffic_percentage", 50)

        # Use consistent hashing for A/B routing
        # This ensures the same user/request always gets the same variant
        request_hash = hash(request_id) % 100
        if request_hash < traffic_percentage:
            return operation.to_version
        else:
            return operation.from_version

    def get_strategy_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema."""
        return {
            "type": "object",
            "properties": {
                "traffic_percentage": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "default": 50,
                    "description": "Percentage of traffic to route to new version"
                },
                "test_duration_hours": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 168,  # 1 week
                    "default": 24,
                    "description": "Duration of A/B test in hours"
                },
                "target_metric": {
                    "type": "string",
                    "enum": ["latency", "throughput", "accuracy", "error_rate"],
                    "default": "latency",
                    "description": "Metric to optimize for"
                }
            },
            "required": []
        }


class CanarySwitch(SwitchingStrategy):
    """
    Canary deployment strategy.

    Routes a small percentage of traffic to the new version initially,
    then gradually increases based on success metrics.
    """

    async def execute_switch(self, switcher: 'ModelSwitcher', operation: SwitchOperation) -> bool:
        """
        Execute canary deployment.

        Args:
            switcher: Model switcher instance
            operation: Switch operation

        Returns:
            True if successful
        """
        try:
            operation.start()

            config = operation.strategy_config
            initial_percentage = config.get("initial_percentage", 5)
            max_percentage = config.get("max_percentage", 100)
            step_percentage = config.get("step_percentage", 10)
            evaluation_period_minutes = config.get("evaluation_period_minutes", 15)

            current_percentage = initial_percentage
            operation.metadata["current_traffic_percentage"] = current_percentage

            logger.info(f"Starting canary deployment for {operation.model_id} "
                       f"with initial {initial_percentage}% traffic")

            while current_percentage < max_percentage:
                operation.update_progress(current_percentage / max_percentage)
                operation.metadata["current_traffic_percentage"] = current_percentage

                # Wait for evaluation period
                await asyncio.sleep(evaluation_period_minutes * 60)

                # Evaluate success metrics (simplified - in practice you'd check actual metrics)
                success = self._evaluate_canary_success(operation)

                if not success:
                    logger.warning(f"Canary evaluation failed at {current_percentage}%, rolling back")
                    operation.fail("Canary evaluation failed")
                    return False

                # Increase traffic
                current_percentage = min(current_percentage + step_percentage, max_percentage)

            # Complete the rollout
            success = switcher._perform_immediate_switch(
                operation.model_id,
                operation.to_version
            )

            if success:
                operation.complete()
                logger.info(f"Canary deployment completed for {operation.model_id}")
            else:
                operation.fail("Canary deployment failed at completion")

            return success

        except Exception as e:
            operation.fail(f"Exception during canary deployment: {e}")
            logger.error(f"Exception during canary deployment: {e}")
            return False

    def _evaluate_canary_success(self, operation: SwitchOperation) -> bool:
        """
        Evaluate if canary deployment is successful.

        Args:
            operation: Switch operation

        Returns:
            True if successful
        """
        # Simplified evaluation - in practice, you'd check actual metrics
        # like error rates, latency, etc.
        config = operation.strategy_config
        max_error_rate = config.get("max_error_rate_threshold", 0.05)

        # Mock evaluation - assume success for demo
        return random.random() > 0.1  # 90% success rate

    def get_traffic_distribution(self, operation: SwitchOperation, request_id: str) -> str:
        """
        Route traffic based on canary deployment progress.

        Args:
            operation: Switch operation
            request_id: Request identifier

        Returns:
            Model version to route to
        """
        if operation.status == "completed":
            return operation.to_version
        elif operation.status != "active":
            return operation.from_version

        traffic_percentage = operation.metadata.get("current_traffic_percentage", 0)

        # Use consistent hashing for canary routing
        request_hash = hash(request_id) % 100
        if request_hash < traffic_percentage:
            return operation.to_version
        else:
            return operation.from_version

    def get_strategy_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema."""
        return {
            "type": "object",
            "properties": {
                "initial_percentage": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 5,
                    "description": "Initial percentage of traffic for canary"
                },
                "max_percentage": {
                    "type": "number",
                    "minimum": 10,
                    "maximum": 100,
                    "default": 100,
                    "description": "Maximum percentage to rollout"
                },
                "step_percentage": {
                    "type": "number",
                    "minimum": 5,
                    "maximum": 50,
                    "default": 10,
                    "description": "Percentage increase per step"
                },
                "evaluation_period_minutes": {
                    "type": "number",
                    "minimum": 5,
                    "maximum": 1440,  # 24 hours
                    "default": 15,
                    "description": "Minutes to evaluate each canary step"
                },
                "max_error_rate_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.05,
                    "description": "Maximum error rate threshold for success"
                }
            },
            "required": []
        }


# Strategy registry
STRATEGY_CLASSES = {
    SwitchingStrategyType.IMMEDIATE: ImmediateSwitch,
    SwitchingStrategyType.GRADUAL: GradualRollout,
    SwitchingStrategyType.AB_TEST: ABTestSwitch,
    SwitchingStrategyType.CANARY: CanarySwitch,
}


def create_strategy(strategy_type: SwitchingStrategyType, config: Dict[str, Any]) -> SwitchingStrategy:
    """
    Create a switching strategy instance.

    Args:
        strategy_type: Type of strategy to create
        config: Strategy configuration

    Returns:
        Switching strategy instance
    """
    strategy_class = STRATEGY_CLASSES.get(strategy_type)
    if not strategy_class:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    # Validate configuration against schema
    strategy = strategy_class()
    schema = strategy.get_strategy_config_schema()

    # Basic validation (in production, use a proper JSON schema validator)
    for key, value in config.items():
        if key in schema.get("properties", {}):
            prop_schema = schema["properties"][key]
            if "minimum" in prop_schema and value < prop_schema["minimum"]:
                raise ValueError(f"{key} must be >= {prop_schema['minimum']}")
            if "maximum" in prop_schema and value > prop_schema["maximum"]:
                raise ValueError(f"{key} must be <= {prop_schema['maximum']}")

    return strategy</content>
