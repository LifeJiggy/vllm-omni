"""
Unit tests for Real-time Model Switching components.

This module contains comprehensive tests for all model switching components.
"""

from unittest.mock import Mock, patch

import pytest

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.model_executor.models.config import ModelSwitchingConfig
from vllm_omni.model_executor.models.dynamic_registry import DynamicModelRegistry, ModelInstance
from vllm_omni.model_executor.models.health_monitor import HealthMonitor, HealthStatus
from vllm_omni.model_executor.models.model_cache import MemoryManager, ModelCache
from vllm_omni.model_executor.models.model_switcher import ModelSwitcher
from vllm_omni.model_executor.models.switching_strategies import (
    GradualRollout,
    ImmediateSwitch,
    SwitchingStrategyType,
    create_strategy,
)
from vllm_omni.model_executor.models.transition_manager import TransitionManager
from vllm_omni.model_executor.models.version_manager import FileVersionStorage, ModelVersionManager


class TestDynamicModelRegistry:
    """Test cases for DynamicModelRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a test registry."""
        base_registry = Mock()
        base_registry.is_text_generation_model.return_value = (True, True)
        return DynamicModelRegistry(base_registry, max_cached_models=2)

    def test_register_model(self, registry):
        """Test model registration."""
        config = OmniModelConfig(
            model="test-model", model_arch="Qwen2_5OmniForConditionalGeneration", model_stage="thinker"
        )

        model_id = registry.register_model(config, "v1.0")
        assert model_id == "Qwen2_5OmniForConditionalGeneration_thinker"
        assert len(registry.model_versions[model_id]) == 1

    def test_switch_model(self, registry):
        """Test model switching."""
        config = OmniModelConfig(
            model="test-model", model_arch="Qwen2_5OmniForConditionalGeneration", model_stage="thinker"
        )

        model_id = registry.register_model(config, "v1.0")
        registry.register_model(config, "v2.0")

        # Mock the _load_model method
        with patch.object(registry, "_load_model") as mock_load:
            mock_load.return_value = ModelInstance(model_id=model_id, version="v2.0", config=config, model="mock_model")

            success = registry.switch_model(model_id, "v2.0")
            assert success
            assert registry.get_active_model(model_id).version == "v2.0"

    def test_deregister_model(self, registry):
        """Test model deregistration."""
        config = OmniModelConfig(
            model="test-model", model_arch="Qwen2_5OmniForConditionalGeneration", model_stage="thinker"
        )

        model_id = registry.register_model(config, "v1.0")
        success = registry.deregister_model(model_id, "v1.0")
        assert success
        assert len(registry.model_versions[model_id]) == 0


class TestModelVersionManager:
    """Test cases for ModelVersionManager."""

    @pytest.fixture
    def version_manager(self, tmp_path):
        """Create a test version manager."""
        storage = FileVersionStorage(str(tmp_path / "versions"))
        return ModelVersionManager(storage)

    def test_create_version(self, version_manager):
        """Test version creation."""
        config = OmniModelConfig(
            model="test-model", model_arch="Qwen2_5OmniForConditionalGeneration", model_stage="thinker"
        )

        version = version_manager.create_version("test_model", config, {"author": "test"})
        assert version.model_id == "test_model"
        assert version.version.startswith("v")
        assert version.metadata["author"] == "test"

    def test_rollback_to_version(self, version_manager):
        """Test version rollback."""
        config = OmniModelConfig(
            model="test-model", model_arch="Qwen2_5OmniForConditionalGeneration", model_stage="thinker"
        )

        v1 = version_manager.create_version("test_model", config)
        _v2 = version_manager.create_version("test_model", config)

        success = version_manager.rollback_to_version("test_model", v1.version)
        assert success

        latest = version_manager.get_latest_version("test_model")
        assert latest.status == "active"


class TestModelCache:
    """Test cases for ModelCache."""

    @pytest.fixture
    def cache(self):
        """Create a test cache."""
        memory_manager = MemoryManager(max_memory_gb=1.0)
        return ModelCache(max_cache_size=2, memory_manager=memory_manager)

    def test_put_and_get_model(self, cache):
        """Test basic cache operations."""
        config = OmniModelConfig(
            model="test-model", model_arch="Qwen2_5OmniForConditionalGeneration", model_stage="thinker"
        )

        instance = ModelInstance(model_id="test_model", version="v1.0", config=config, model="mock_model")

        success = cache.put_model("test_key", instance)
        assert success

        retrieved = cache.get_model("test_key")
        assert retrieved == instance

    def test_cache_eviction(self, cache):
        """Test LRU eviction."""
        config = OmniModelConfig(
            model="test-model", model_arch="Qwen2_5OmniForConditionalGeneration", model_stage="thinker"
        )

        # Add 3 items to cache with max size 2
        for i in range(3):
            instance = ModelInstance(
                model_id=f"test_model_{i}", version=f"v{i}", config=config, model=f"mock_model_{i}"
            )
            cache.put_model(f"key_{i}", instance)

        # Cache should only have 2 items
        assert len(cache) == 2
        assert "key_0" not in cache  # First item should be evicted


class TestTransitionManager:
    """Test cases for TransitionManager."""

    @pytest.fixture
    def transition_manager(self):
        """Create a test transition manager."""
        return TransitionManager()

    def test_begin_transition(self, transition_manager):
        """Test transition initiation."""
        config = OmniModelConfig(
            model="test-model", model_arch="Qwen2_5OmniForConditionalGeneration", model_stage="thinker"
        )

        old_model = ModelInstance("test_model", "v1.0", config, "old_model")
        new_model = ModelInstance("test_model", "v2.0", config, "new_model")

        transition_id = transition_manager.begin_transition("test_model", old_model, new_model)
        assert transition_id
        assert transition_manager.is_transition_active("test_model")

    def test_complete_transition(self, transition_manager):
        """Test transition completion."""
        config = OmniModelConfig(
            model="test-model", model_arch="Qwen2_5OmniForConditionalGeneration", model_stage="thinker"
        )

        old_model = ModelInstance("test_model", "v1.0", config, "old_model")
        new_model = ModelInstance("test_model", "v2.0", config, "new_model")

        transition_id = transition_manager.begin_transition("test_model", old_model, new_model)
        success = transition_manager.complete_transition(transition_id)
        assert success
        assert not transition_manager.is_transition_active("test_model")


class TestSwitchingStrategies:
    """Test cases for switching strategies."""

    def test_create_strategy(self):
        """Test strategy creation."""
        strategy = create_strategy(SwitchingStrategyType.IMMEDIATE, {})
        assert isinstance(strategy, ImmediateSwitch)

        strategy = create_strategy(SwitchingStrategyType.GRADUAL, {"duration_minutes": 5})
        assert isinstance(strategy, GradualRollout)

    def test_immediate_strategy(self):
        """Test immediate switching strategy."""
        strategy = ImmediateSwitch()

        # Mock switcher
        switcher = Mock()
        switcher._perform_immediate_switch.return_value = True

        operation = Mock()
        operation.model_id = "test_model"
        operation.from_version = "v1.0"
        operation.to_version = "v2.0"

        # This would normally be async, but we'll test the logic
        assert strategy.get_traffic_distribution(operation, "req1") == "v2.0"

    def test_gradual_strategy(self):
        """Test gradual rollout strategy."""
        strategy = GradualRollout()

        operation = Mock()
        operation.status = "active"
        operation.metadata = {"current_traffic_percentage": 30}
        operation.from_version = "v1.0"
        operation.to_version = "v2.0"

        # Mock request routing
        with patch("random.random", return_value=0.2):  # 20% - should route to old
            version = strategy.get_traffic_distribution(operation, "req1")
            assert version == "v1.0"

        with patch("random.random", return_value=0.4):  # 40% - should route to new
            version = strategy.get_traffic_distribution(operation, "req2")
            assert version == "v2.0"


class TestHealthMonitor:
    """Test cases for HealthMonitor."""

    @pytest.fixture
    def health_monitor(self):
        """Create a test health monitor."""
        return HealthMonitor()

    def test_record_request(self, health_monitor):
        """Test request recording."""
        health_monitor.record_request("test_model_v1.0", 100.0, error=False)
        health_monitor.record_request("test_model_v1.0", 200.0, error=True)

        metrics = health_monitor.get_model_metrics("test_model_v1.0")
        assert metrics["current_metrics"]["request_count"] == 2
        assert metrics["current_metrics"]["error_count"] == 1
        assert metrics["current_metrics"]["error_rate"] == 0.5

    def test_health_status_calculation(self, health_monitor):
        """Test health status calculation."""
        # Record some healthy requests
        for _ in range(10):
            health_monitor.record_request("test_model_v1.0", 100.0, error=False)

        status = health_monitor.get_health_status("test_model_v1.0")
        assert status == HealthStatus.HEALTHY

        # Record many errors
        for _ in range(15):
            health_monitor.record_request("test_model_v1.0", 100.0, error=True)

        status = health_monitor.get_health_status("test_model_v1.0")
        assert status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]


class TestModelSwitcher:
    """Test cases for ModelSwitcher."""

    @pytest.fixture
    def switcher_components(self):
        """Create test switcher components."""
        registry = DynamicModelRegistry(Mock())
        cache = ModelCache(max_cache_size=2)
        transition_manager = TransitionManager()
        return registry, cache, transition_manager

    @pytest.fixture
    def switcher(self, switcher_components):
        """Create a test switcher."""
        registry, cache, transition_manager = switcher_components
        return ModelSwitcher(registry, cache, transition_manager)

    def test_validate_switch_request(self, switcher):
        """Test switch request validation."""
        config = OmniModelConfig(
            model="test-model", model_arch="Qwen2_5OmniForConditionalGeneration", model_stage="thinker"
        )

        # Register a model
        model_id = switcher.registry.register_model(config, "v1.0")

        # Valid request
        result = switcher.validate_switch_request(model_id, "v1.0")
        assert result["valid"]

        # Invalid version
        result = switcher.validate_switch_request(model_id, "v2.0")
        assert not result["valid"]
        assert "not found" in result["errors"][0]


class TestModelSwitchingConfig:
    """Test cases for configuration."""

    def test_config_validation(self):
        """Test configuration validation."""
        config = ModelSwitchingConfig()

        # Valid config should not raise
        assert config.error_rate_warning_threshold < config.error_rate_critical_threshold

        # Invalid config should raise
        with pytest.raises(ValueError):
            ModelSwitchingConfig(error_rate_warning_threshold=0.8, error_rate_critical_threshold=0.5)

    def test_env_config(self):
        """Test environment-based configuration."""
        with patch.dict(
            "os.environ", {"VLLM_MODEL_SWITCHING_MAX_CACHED_MODELS": "10", "VLLM_MODEL_SWITCHING_ENABLE_CACHE": "false"}
        ):
            config = ModelSwitchingConfig.from_env()
            assert config.max_cached_models == 10
            assert not config.enable_model_cache


if __name__ == "__main__":
    pytest.main([__file__])
