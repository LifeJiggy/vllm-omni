#!/usr/bin/env python3
"""
Example: Integrating Real-time Model Switching with vLLM-Omni Engine

This example demonstrates how to integrate the model switching system
with the vLLM-Omni inference engine for seamless model transitions.
"""

import asyncio
import logging
import time
from typing import Any

# vLLM-Omni imports
from vllm_omni.config.model import OmniModelConfig
from vllm_omni.model_executor.models import (
    DynamicModelRegistry,
    HealthMonitor,
    ModelCache,
    ModelSwitcher,
    TransitionManager,
)
from vllm_omni.model_executor.models.config import ModelSwitchingConfig
from vllm_omni.model_executor.models.switching_strategies import SwitchingStrategyType


# Mock vLLM engine for demonstration
class MockVLLEngine:
    """Mock vLLM engine for demonstration purposes."""

    def __init__(self):
        self.models = {}  # model_id -> model_instance
        self.request_count = 0

    def register_model(self, model_id: str, model_instance):
        """Register a model with the engine."""
        self.models[model_id] = model_instance
        print(f"Registered model: {model_id}")

    def process_request(self, request):
        """Process an inference request."""
        self.request_count += 1

        # Simulate processing
        time.sleep(0.01)  # 10ms processing time

        return {
            "request_id": request.get("id", self.request_count),
            "result": f"Processed by {request.get('model_id', 'unknown')}",
            "latency_ms": 10,
        }


class SwitchingEnabledEngine(MockVLLEngine):
    """
    vLLM-Omni engine with real-time model switching capabilities.

    This example shows how to integrate the switching system with
    the inference engine's request processing pipeline.
    """

    def __init__(self, switching_config: ModelSwitchingConfig | None = None):
        super().__init__()

        # Initialize switching components
        self.config = switching_config or ModelSwitchingConfig()

        # Create base registry (mock for this example)
        base_registry = type(
            "MockRegistry", (), {"is_text_generation_model": lambda self, arch, config: (True, True)}
        )()

        # Initialize switching system
        self.registry = DynamicModelRegistry(
            base_registry=base_registry,
            max_cached_models=self.config.max_cached_models,
            model_ttl_seconds=self.config.model_ttl_seconds,
        )

        self.cache = ModelCache(
            max_cache_size=self.config.max_cache_size,
            memory_manager=None,  # Simplified for demo
        )

        self.transition_manager = TransitionManager()
        self.health_monitor = HealthMonitor()

        self.switcher = ModelSwitcher(
            registry=self.registry,
            cache=self.cache,
            transition_manager=self.transition_manager,
            max_concurrent_switches=self.config.max_concurrent_switches,
        )

        # Start background tasks
        self._start_background_tasks()

        print("Switching-enabled engine initialized")

    def _start_background_tasks(self):
        """Start background tasks for the switching system."""
        # Note: In a real implementation, these would be properly managed
        # with the asyncio event loop
        pass

    def register_switching_model(
        self, model_config: OmniModelConfig, version: str, model_instance=None, metadata: dict[str, Any] | None = None
    ):
        """
        Register a model version for switching.

        Args:
            model_config: Model configuration
            version: Version string
            model_instance: Actual model instance (mock for demo)
            metadata: Optional metadata
        """
        # Register with switching system
        model_id = self.registry.register_model(model_config, version, metadata)

        # Register with engine
        if model_instance:
            self.register_model(model_id, model_instance)

        # Set as active if first version
        if len(self.registry.model_versions[model_id]) == 1:
            self.registry.switch_model(model_id, version)

        print(f"Registered switching model: {model_id} v{version}")
        return model_id

    def switch_model(
        self,
        model_id: str,
        target_version: str,
        strategy: SwitchingStrategyType = SwitchingStrategyType.IMMEDIATE,
        strategy_config: dict[str, Any] | None = None,
    ):
        """
        Initiate a model switch operation.

        Args:
            model_id: Model identifier
            target_version: Target version
            strategy: Switching strategy
            strategy_config: Strategy configuration
        """

        async def _switch():
            result = await self.switcher.switch_model(
                model_id=model_id,
                target_version=target_version,
                strategy_type=strategy,
                strategy_config=strategy_config,
            )
            return result

        # In a real implementation, this would be properly awaited
        # For demo purposes, we'll simulate the switch
        print(f"Initiating {strategy.value} switch: {model_id} -> v{target_version}")

        # Mock successful switch
        success = self.registry.switch_model(model_id, target_version)
        return {
            "operation_id": f"switch_{int(time.time())}",
            "success": success,
            "model_id": model_id,
            "from_version": "previous",
            "to_version": target_version,
            "strategy": strategy.value,
        }

    def process_request(self, request):
        """
        Process a request with switching-aware routing.

        Args:
            request: Inference request
        """
        model_id = request.get("model_id", "default_model")
        request_id = request.get("id", f"req_{self.request_count + 1}")

        # Check for active transitions
        assigned_model = self.transition_manager.handle_request_routing(model_id, request_id)

        if assigned_model:
            # Route to specific model instance during transition
            print(
                f"Routing request {request_id} to transition model: {assigned_model.model_id} v{assigned_model.version}"
            )
            result = super().process_request(request)
            result["assigned_model"] = f"{assigned_model.model_id} v{assigned_model.version}"
        else:
            # Normal routing
            result = super().process_request(request)

        # Record metrics for health monitoring
        latency_ms = result.get("latency_ms", 10)
        error = result.get("error", False)
        model_key = f"{model_id}_{self._get_active_version(model_id)}"
        self.health_monitor.record_request(model_key, latency_ms, error)

        return result

    def _get_active_version(self, model_id: str) -> str:
        """Get the active version for a model."""
        active_instance = self.registry.get_active_model(model_id)
        return active_instance.version if active_instance else "unknown"

    def get_switching_stats(self):
        """Get switching system statistics."""
        return {
            "registry": self.registry.get_registry_stats(),
            "cache": self.cache.get_cache_stats(),
            "switcher": self.switcher.get_switcher_stats(),
            "health": self.health_monitor.get_monitor_stats(),
            "transitions": self.transition_manager.get_transition_stats(),
        }


async def demo_model_switching():
    """Demonstrate the model switching integration."""

    print("🚀 Starting vLLM-Omni Model Switching Demo")
    print("=" * 50)

    # Initialize engine with switching
    engine = SwitchingEnabledEngine()

    # Register model versions
    print("\n📝 Registering model versions...")

    config_v1 = OmniModelConfig(
        model="Qwen/Qwen2.5-Omni-7B", model_arch="Qwen2_5OmniForConditionalGeneration", model_stage="thinker"
    )

    config_v2 = OmniModelConfig(
        model="Qwen/Qwen2.5-Omni-7B-v2", model_arch="Qwen2_5OmniForConditionalGeneration", model_stage="thinker"
    )

    # Register versions
    model_id = engine.register_switching_model(config_v1, "v1.0", metadata={"performance": "baseline"})
    engine.register_switching_model(config_v2, "v2.0", metadata={"performance": "improved"})

    print(f"✅ Registered model: {model_id}")

    # Process some requests
    print("\n🔄 Processing requests with v1.0...")
    for i in range(5):
        request = {"id": f"req_{i + 1}", "model_id": model_id, "prompt": f"Test request {i + 1}"}
        result = engine.process_request(request)
        print(f"  Request {i + 1}: {result['result']}")

    # Initiate model switch
    print("\n🔄 Switching to v2.0 with gradual rollout...")
    switch_result = engine.switch_model(
        model_id=model_id,
        target_version="v2.0",
        strategy=SwitchingStrategyType.GRADUAL,
        strategy_config={"duration_minutes": 1, "steps": 3},
    )

    print(f"Switch initiated: {switch_result['success']}")

    # Process requests during transition
    print("\n🔄 Processing requests during transition...")
    for i in range(5, 10):
        request = {"id": f"req_{i + 1}", "model_id": model_id, "prompt": f"Transition request {i + 1}"}
        result = engine.process_request(request)
        assigned = result.get("assigned_model", "normal routing")
        print(f"  Request {i + 1}: {result['result']} (routed: {assigned})")

    # Check final stats
    print("\n📊 Final Statistics:")
    stats = engine.get_switching_stats()
    print(f"  Models registered: {stats['registry']['total_versions']}")
    print(f"  Cache entries: {stats['cache']['cache_entries']}")
    print(f"  Total requests: {stats['health']['total_requests']}")
    print(f"  Active transitions: {stats['transitions']['active_transitions']}")

    print("\n✅ Demo completed successfully!")


async def demo_api_server():
    """Demonstrate the REST API server."""

    print("\n🌐 Starting Model Switching API Server Demo")
    print("=" * 50)

    # Initialize components
    base_registry = type("MockRegistry", (), {"is_text_generation_model": lambda self, arch, config: (True, True)})()

    registry = DynamicModelRegistry(base_registry)
    cache = ModelCache(max_cache_size=3)
    transition_manager = TransitionManager()
    health_monitor = HealthMonitor()

    switcher = ModelSwitcher(registry, cache, transition_manager)

    # Create API
    from vllm_omni.model_executor.models.api import ModelSwitchingAPI

    _api = ModelSwitchingAPI(registry, switcher, health_monitor)

    print("API endpoints available:")
    print("  POST /models - Register new models")
    print("  GET  /models - List registered models")
    print("  POST /switch - Initiate model switches")
    print("  GET  /health/models - Monitor model health")
    print("  GET  /alerts - View active alerts")
    print("  GET  /stats - Get system statistics")

    # Note: In a real scenario, you would call:
    # await api.start_server(host="0.0.0.0", port=8001)

    print("API server demo complete (server not started in demo)")


def main():
    """Main demo function."""
    logging.basicConfig(level=logging.INFO)

    print("🎯 vLLM-Omni Real-time Model Switching Demo")
    print("This demo shows how to integrate model switching with the inference engine.")

    # Run demos
    asyncio.run(demo_model_switching())
    asyncio.run(demo_api_server())

    print("\n🎉 All demos completed!")
    print("\nNext steps:")
    print("1. Integrate with your actual vLLM engine")
    print("2. Configure health monitoring thresholds")
    print("3. Set up alerting for production")
    print("4. Deploy the REST API for management")


if __name__ == "__main__":
    main()
