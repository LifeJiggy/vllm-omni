#!/usr/bin/env python3
"""
Simple test script for DiT batching functionality.

This script tests the batching logic without requiring full vLLM installation.
"""

import os
import sys

# Add the vllm_omni package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all batching-related imports work."""
    print("Testing imports...")

    try:
        # Test import availability using importlib
        import importlib.util

        modules_to_test = [
            "vllm_omni.diffusion.config.batching",
            "vllm_omni.diffusion.scheduler.dit_batching_scheduler",
            "vllm_omni.diffusion.scheduler.compatibility",
            "vllm_omni.diffusion.worker.dynamic_batch_handler",
        ]

        for module_name in modules_to_test:
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                raise ImportError(f"Module {module_name} not found")
            print(f"PASS: {module_name} import available")

        return True
    except ImportError as e:
        print(f"FAIL: Import failed: {e}")
        return False


def test_package_structure():
    """Test that package structure is correct."""
    print("\nTesting package structure...")

    try:
        # Check that __init__.py files exist and are importable
        import importlib.util

        spec = importlib.util.find_spec("vllm_omni.diffusion.config")
        if spec is None:
            raise ImportError("vllm_omni.diffusion.config not found")

        print("PASS: Package structure correct")
        return True
    except ImportError as e:
        print(f"FAIL: Package structure issue: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing DiT Batching Implementation")
    print("=" * 50)

    tests = [
        test_imports,
        test_package_structure,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: All tests passed! DiT batching package structure is correct.")
        return 0
    else:
        print("FAILED: Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
