#!/usr/bin/env python3
"""
Simple test script for DiT batching functionality.

This script tests the batching logic without requiring full vLLM installation.
"""

import sys
import os

# Add the vllm_omni package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all batching-related imports work."""
    print("Testing imports...")

    try:
        from vllm_omni.diffusion.config.batching import DiTBatchingConfig
        print("PASS: DiTBatchingConfig import successful")

        from vllm_omni.diffusion.scheduler.dit_batching_scheduler import DiTBatchingScheduler, BatchingConfig
        print("PASS: DiTBatchingScheduler import successful")

        from vllm_omni.diffusion.scheduler.compatibility import CompatibilityGroupManager, RequestCompatibilityChecker
        print("PASS: Compatibility classes import successful")

        from vllm_omni.diffusion.worker.dynamic_batch_handler import DynamicBatchHandler
        print("PASS: DynamicBatchHandler import successful")

        return True
    except ImportError as e:
        print(f"FAIL: Import failed: {e}")
        return False

def test_batching_config():
    """Test DiTBatchingConfig functionality."""
    print("\nTesting DiTBatchingConfig...")

    try:
        # Test default config
        config = DiTBatchingConfig()
        assert config.enable_batching == False
        print("PASS: Default config works")

        # Test batching enabled config
        config = DiTBatchingConfig(
            enable_batching=True,
            max_batch_size=4,
            max_wait_time_ms=100,
            max_memory_mb=8192
        )
        assert config.enable_batching == True
        assert config.max_batch_size == 4
        print("PASS: Batching config works")

        return True
    except Exception as e:
        print(f"FAIL: Batching config test failed: {e}")
        return False

def test_compatibility_checker():
    """Test compatibility checking logic."""
    print("\nTesting compatibility checker...")

    try:
        from vllm_omni.diffusion.request import OmniDiffusionRequest

        checker = RequestCompatibilityChecker()

        # Create test requests
        req1 = OmniDiffusionRequest(
            prompt="test prompt 1",
            height=1024,
            width=1024,
            num_inference_steps=50,
            guidance_scale=7.5
        )

        req2 = OmniDiffusionRequest(
            prompt="test prompt 2",
            height=1024,
            width=1024,
            num_inference_steps=45,  # Within 20% tolerance
            guidance_scale=7.0
        )

        req3 = OmniDiffusionRequest(
            prompt="test prompt 3",
            height=512,
            width=512,  # Different resolution
            num_inference_steps=50,
            guidance_scale=7.5
        )

        # Test compatibility
        assert checker.is_compatible(req1, req2), "Similar requests should be compatible"
        assert not checker.is_compatible(req1, req3), "Different resolutions should not be compatible"

        print("PASS: Compatibility checking works")
        return True
    except Exception as e:
        print(f"FAIL: Compatibility test failed: {e}")
        return False

def test_scheduler():
    """Test the batching scheduler."""
    print("\nTesting DiTBatchingScheduler...")

    try:
        from vllm_omni.diffusion.config.batching import DiTBatchingConfig
        from vllm_omni.diffusion.request import OmniDiffusionRequest

        # Create config
        config = DiTBatchingConfig(
            enable_batching=True,
            max_batch_size=4,
            max_wait_time_ms=100
        )

        # Convert to scheduler config
        scheduler_config = BatchingConfig(
            max_batch_size=config.max_batch_size,
            min_batch_size=config.min_batch_size,
            max_wait_time_ms=config.max_wait_time_ms,
            max_memory_mb=config.max_memory_mb,
        )

        scheduler = DiTBatchingScheduler(scheduler_config)

        # Create test requests
        requests = []
        for i in range(3):
            req = OmniDiffusionRequest(
                prompt=f"test prompt {i}",
                request_id=f"test-{i}",
                height=1024,
                width=1024,
                num_inference_steps=50
            )
            requests.append(req)

        # Add requests
        for req in requests:
            import asyncio
            asyncio.run(scheduler.add_request(req))

        # Get batch
        batch = asyncio.run(scheduler.get_next_batch())
        assert batch is not None, "Should get a batch"
        assert len(batch) > 0, "Batch should not be empty"

        print("PASS: Scheduler works")
        return True
    except Exception as e:
        print(f"FAIL: Scheduler test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing DiT Batching Implementation")
    print("=" * 50)

    tests = [
        test_imports,
        test_batching_config,
        test_compatibility_checker,
        test_scheduler,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: All tests passed! DiT batching implementation is working.")
        return 0
    else:
        print("FAILED: Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())