# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import torch
from typing import Optional, Tuple
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available, system memory monitoring disabled")


def get_system_memory_info() -> Tuple[int, int]:
    """Get system memory information in bytes.

    Returns:
        Tuple of (total_memory, available_memory) in bytes
    """
    if not HAS_PSUTIL:
        # Fallback: estimate based on common system specs
        logger.warning("Using fallback memory estimation (psutil not available)")
        return 16 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024  # 16GB total, 8GB available

    mem = psutil.virtual_memory()
    return mem.total, mem.available


def get_gpu_memory_info(device: Optional[torch.device] = None) -> Tuple[int, int]:
    """Get GPU memory information in bytes.

    Args:
        device: The device to check. If None, uses current device.

    Returns:
        Tuple of (total_memory, free_memory) in bytes
    """
    if not torch.cuda.is_available():
        return 0, 0

    if device is None:
        device = torch.cuda.current_device()

    try:
        # Get memory info for the specific device
        mem_info = torch.cuda.mem_get_info(device)
        return mem_info[1], mem_info[0]  # total, free
    except Exception:
        return 0, 0


def estimate_model_memory_usage(model_name_or_path: str, model_type: str = "diffusion") -> int:
    """Estimate memory usage for loading a model.

    Args:
        model_name_or_path: Model name or path
        model_type: Type of model ("diffusion", "llm", etc.)

    Returns:
        Estimated memory usage in bytes
    """
    # This is a rough estimation - in practice, you'd need more sophisticated analysis
    # For now, use some heuristics based on known model sizes

    if "Qwen" in model_name_or_path:
        if "Image-Edit" in model_name_or_path:
            # Qwen Image Edit models are quite large
            return 32 * 1024 * 1024 * 1024  # 32GB estimate
        elif "Omni" in model_name_or_path:
            return 16 * 1024 * 1024 * 1024  # 16GB estimate

    # Default estimate
    return 8 * 1024 * 1024 * 1024  # 8GB


def select_optimal_device(model_name_or_path: str, model_type: str = "diffusion") -> str:
    """Select the optimal device for loading a model based on available memory.

    Args:
        model_name_or_path: Model name or path
        model_type: Type of model

    Returns:
        Device string ("cpu", "cuda", "cuda:0", etc.)
    """
    estimated_usage = estimate_model_memory_usage(model_name_or_path, model_type)

    # Check GPU memory first
    if torch.cuda.is_available():
        total_gpu, free_gpu = get_gpu_memory_info()
        if free_gpu > estimated_usage * 1.2:  # 20% buffer
            logger.info(f"Selected GPU device (free: {free_gpu/1024**3:.2f}GB, "
                       f"estimated usage: {estimated_usage/1024**3:.2f}GB)")
            return "cuda"

    # Check system memory
    total_sys, available_sys = get_system_memory_info()
    if available_sys > estimated_usage * 1.2:  # 20% buffer
        logger.info(f"Selected CPU device (available: {available_sys/1024**3:.2f}GB, "
                   f"estimated usage: {estimated_usage/1024**3:.2f}GB)")
        return "cpu"

    # Fallback to CPU even if memory is tight
    logger.warning(f"Memory may be insufficient. Using CPU anyway "
                  f"(available: {available_sys/1024**3:.2f}GB, "
                  f"estimated usage: {estimated_usage/1024**3:.2f}GB)")
    return "cpu"


def log_memory_usage(stage: str = ""):
    """Log current memory usage for diagnostics.

    Args:
        stage: Optional stage description for logging
    """
    prefix = f"[{stage}] " if stage else ""

    # System memory
    total_sys, available_sys = get_system_memory_info()
    used_sys = total_sys - available_sys
    logger.info(f"{prefix}System memory: {used_sys/1024**3:.2f}GB used / "
               f"{total_sys/1024**3:.2f}GB total "
               f"({available_sys/1024**3:.2f}GB available)")

    # GPU memory
    if torch.cuda.is_available():
        total_gpu, free_gpu = get_gpu_memory_info()
        used_gpu = total_gpu - free_gpu
        logger.info(f"{prefix}GPU memory: {used_gpu/1024**3:.2f}GB used / "
                   f"{total_gpu/1024**3:.2f}GB total "
                   f"({free_gpu/1024**3:.2f}GB available)")


def check_memory_thresholds(min_available_gb: float = 4.0) -> bool:
    """Check if system has sufficient memory available.

    Args:
        min_available_gb: Minimum available memory in GB

    Returns:
        True if sufficient memory is available
    """
    min_available_bytes = min_available_gb * 1024 * 1024 * 1024

    # Check system memory
    _, available_sys = get_system_memory_info()
    if available_sys < min_available_bytes:
        logger.warning(f"Low system memory: {available_sys/1024**3:.2f}GB available "
                      f"(minimum required: {min_available_gb}GB)")
        return False

    # Check GPU memory if available
    if torch.cuda.is_available():
        _, free_gpu = get_gpu_memory_info()
        if free_gpu < min_available_bytes:
            logger.warning(f"Low GPU memory: {free_gpu/1024**3:.2f}GB free "
                          f"(minimum required: {min_available_gb}GB)")
            return False

    return True