# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import multiprocessing as mp
import os
import time
import uuid
import weakref
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pprint import pformat
from typing import Any

from omegaconf import OmegaConf
from tqdm.auto import tqdm
from vllm.inputs import PromptType
from vllm.logger import init_logger

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors import (
    get_stage_connector_config,
    initialize_orchestrator_connectors,
)
from vllm_omni.distributed.omni_connectors.adapter import try_send_via_connector
from vllm_omni.distributed.ray_utils.utils import (
    create_placement_group,
    get_ray_queue_class,
    try_close_ray,
)
from vllm_omni.entrypoints.log_utils import OrchestratorMetrics
from vllm_omni.entrypoints.omni_stage import OmniStage
from vllm_omni.entrypoints.stage_utils import SHUTDOWN_TASK
from vllm_omni.entrypoints.stage_utils import maybe_load_from_ipc as _load
from vllm_omni.entrypoints.utils import (
    get_final_stage_id_for_e2e,
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)
from vllm_omni.outputs import OmniRequestOutput

# Switching integration imports
from vllm_omni.model_executor.models.switching_orchestrator import SwitchingOrchestrator
from vllm_omni.model_executor.models.request_router import RequestRouter, RoutingDecision
from vllm_omni.model_executor.models.model_switcher import ModelSwitcher
from vllm_omni.model_executor.models.dynamic_registry import DynamicModelRegistry
from vllm_omni.model_executor.models.model_cache import ModelCache
from vllm_omni.model_executor.models.transition_manager import TransitionManager

logger = init_logger(__name__)