"""
Test for switching integration with vLLM-Omni request processing pipeline.

This test verifies that the switching orchestrator and request router
properly integrate with the Omni entry points.
"""

from unittest.mock import Mock, patch

import pytest

from vllm_omni.model_executor.models.dynamic_registry import DynamicModelRegistry
from vllm_omni.model_executor.models.model_cache import ModelCache
from vllm_omni.model_executor.models.model_switcher import ModelSwitcher
from vllm_omni.model_executor.models.request_router import RequestRouter, RoutingDecision
from vllm_omni.model_executor.models.switching_orchestrator import SwitchingOrchestrator
from vllm_omni.model_executor.models.switching_strategies import SwitchingStrategyType


class TestSwitchingIntegration:
    """Test switching integration components."""

    @pytest.fixture
    def mock_model_switcher(self):
        """Create a mock model switcher."""
        switcher = Mock(spec=ModelSwitcher)
        switcher.registry = Mock(spec=DynamicModelRegistry)
        switcher.cache = Mock(spec=ModelCache)
        return switcher

    @pytest.fixture
    def switching_orchestrator(self, mock_model_switcher):
        """Create a switching orchestrator."""
        return SwitchingOrchestrator(mock_model_switcher)

    @pytest.fixture
    def request_router(self, switching_orchestrator):
        """Create a request router."""
        return RequestRouter(switching_orchestrator)

    def test_routing_decision_creation(self, request_router):
        """Test that routing decisions are created correctly."""
        decision = request_router.route_request("test_model", {"prompt": "test"})

        assert isinstance(decision, RoutingDecision)
        assert decision.model_id == "test_model"
        assert decision.target_version == "default"  # No active switch
        assert not decision.is_switching
        assert decision.switching_operation_id is None

    def test_batch_routing(self, request_router):
        """Test batch request routing."""
        requests = [{"prompt": "test1"}, {"prompt": "test2"}, {"prompt": "test3"}]

        decisions = request_router.route_batch_request("test_model", requests)

        assert len(decisions) == 3
        for i, decision in enumerate(decisions):
            assert isinstance(decision, RoutingDecision)
            assert decision.model_id == "test_model"
            assert decision.request_id == f"batch_{i}"
            assert decision.target_version == "default"

    @pytest.mark.asyncio
    async def test_switch_lifecycle(self, switching_orchestrator, mock_model_switcher):
        """Test the complete switch lifecycle."""
        # Mock successful switch
        mock_result = Mock()
        mock_result.success = True
        mock_result.operation_id = "test_op_123"
        mock_result.from_version = "v1.0"
        mock_result.to_version = "v2.0"
        mock_result.strategy_type = SwitchingStrategyType.IMMEDIATE

        mock_model_switcher.switch_model = Mock(return_value=mock_result)

        # Start switch
        operation_id = await switching_orchestrator.start_switch(
            model_id="test_model", target_version="v2.0", strategy_type=SwitchingStrategyType.IMMEDIATE
        )

        assert operation_id == "test_op_123"
        assert "test_model" in switching_orchestrator.active_switches

        # Check routing after switch
        request_router = RequestRouter(switching_orchestrator)
        decision = request_router.route_request("test_model", {"prompt": "test"})

        # Should route to new version (this depends on strategy implementation)
        assert isinstance(decision, RoutingDecision)

    def test_routing_cache(self, request_router):
        """Test routing cache functionality."""
        # First request
        decision1 = request_router.route_request("test_model", {"prompt": "test"}, "req1")
        decision2 = request_router.route_request("test_model", {"prompt": "test"}, "req1")

        # Should be the same object from cache
        assert decision1 is decision2

        # Clear cache
        request_router.clear_cache()
        decision3 = request_router.route_request("test_model", {"prompt": "test"}, "req1")

        # Should be different object after cache clear
        assert decision1 is not decision3

    def test_routing_statistics(self, request_router):
        """Test routing statistics."""
        # Make some requests
        request_router.route_request("test_model", {"prompt": "test1"})
        request_router.route_request("test_model", {"prompt": "test2"})

        stats = request_router.get_routing_stats()

        assert "cache_size" in stats
        assert "active_switches" in stats
        assert stats["cache_size"] >= 2  # At least 2 cached decisions

    @pytest.mark.asyncio
    async def test_switch_abort(self, switching_orchestrator, mock_model_switcher):
        """Test switch abortion."""
        # Mock successful switch
        mock_result = Mock()
        mock_result.success = True
        mock_result.operation_id = "test_op_123"
        mock_model_switcher.switch_model = Mock(return_value=mock_result)
        mock_model_switcher.abort_switch = Mock(return_value=True)

        # Start switch
        await switching_orchestrator.start_switch("test_model", "v2.0")

        # Abort switch
        success = await switching_orchestrator.abort_switch("test_model")

        assert success
        assert "test_model" not in switching_orchestrator.active_switches
        mock_model_switcher.abort_switch.assert_called_once()


class TestOmniSwitchingIntegration:
    """Test integration with Omni entry points."""

    @pytest.fixture
    def mock_omni(self):
        """Create a mock Omni instance."""
        omni = Mock()
        omni.generate = Mock(return_value=iter([]))
        return omni

    @patch("vllm_omni.entrypoints.omni_switching.Omni")
    def test_omni_with_switching_creation(self, mock_omni_class, mock_omni):
        """Test creation of OmniWithSwitching."""
        mock_omni_class.return_value = mock_omni

        from vllm_omni.entrypoints.omni_switching import OmniWithSwitching

        omni_switching = OmniWithSwitching("test_model", enable_switching=True)

        assert omni_switching.model == "test_model"
        assert omni_switching.enable_switching
        assert omni_switching.omni == mock_omni
        assert omni_switching.switching_orchestrator is not None
        assert omni_switching.request_router is not None

    @patch("vllm_omni.entrypoints.omni_switching.Omni")
    def test_omni_without_switching(self, mock_omni_class, mock_omni):
        """Test creation of Omni without switching."""
        mock_omni_class.return_value = mock_omni

        from vllm_omni.entrypoints.omni_switching import OmniWithSwitching

        omni_switching = OmniWithSwitching("test_model", enable_switching=False)

        assert omni_switching.enable_switching is False
        assert omni_switching.switching_orchestrator is None
        assert omni_switching.request_router is None

    @patch("vllm_omni.entrypoints.omni_switching.Omni")
    def test_generate_with_switching(self, mock_omni_class, mock_omni):
        """Test generation with switching enabled."""
        # Mock Omni output
        mock_output = Mock()
        mock_output.metadata = {}
        mock_omni.generate.return_value = [mock_output]
        mock_omni_class.return_value = mock_omni

        from vllm_omni.entrypoints.omni_switching import OmniWithSwitching

        omni_switching = OmniWithSwitching("test_model", enable_switching=True)

        # Generate
        outputs = list(omni_switching.generate(prompts=["test prompt"]))

        assert len(outputs) == 1
        assert "routing_decision" in outputs[0].metadata

    def test_create_omni_with_switching_function(self):
        """Test the convenience function."""
        from vllm_omni.entrypoints.omni_switching import create_omni_with_switching

        with patch("vllm_omni.entrypoints.omni_switching.OmniWithSwitching") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            result = create_omni_with_switching("test_model", enable_switching=True)

            assert result == mock_instance
            mock_class.assert_called_once_with(model="test_model", enable_switching=True)


if __name__ == "__main__":
    pytest.main([__file__])
