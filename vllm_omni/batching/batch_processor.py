# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from typing import Any, Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

from .request_queue import QueuedRequest
from vllm.logger import init_logger

logger = init_logger(__name__)


class BatchProcessor:
    """Processes batches of requests using the underlying engine.

    Handles the execution of batched requests and result distribution.
    """

    def __init__(
        self,
        engine: Any,
        max_concurrent_batches: int = 4,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        self.engine = engine
        self.max_concurrent_batches = max_concurrent_batches
        self.executor = executor or ThreadPoolExecutor(max_workers=max_concurrent_batches)
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)

    async def process_batch(
        self,
        batch: List[QueuedRequest],
        result_callback: Optional[Callable[[str, Any], None]] = None,
    ) -> None:
        """Process a batch of requests.

        Args:
            batch: List of requests to process
            result_callback: Optional callback for individual request results
        """
        if not batch:
            return

        async with self._semaphore:
            try:
                # Extract request data
                request_ids = [req.request_id for req in batch]
                prompts = [req.data.get("prompt") for req in batch]
                sampling_params_list = [req.data.get("sampling_params", {}) for req in batch]

                logger.info(f"Processing batch of {len(batch)} requests: {request_ids}")

                # For now, process requests sequentially
                # TODO: Implement true batching in AsyncOmni
                results = []
                for i, req in enumerate(batch):
                    try:
                        # Process individual request
                        result = await self._process_single_request(
                            req.request_id,
                            req.data,
                        )
                        results.append(result)

                        # Call callback if provided
                        if result_callback:
                            result_callback(req.request_id, result)

                    except Exception as e:
                        logger.error(f"Failed to process request {req.request_id}: {e}")
                        if result_callback:
                            result_callback(req.request_id, {"error": str(e)})

                logger.info(f"Completed batch processing for {len(results)} requests")

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Fail all requests in batch
                for req in batch:
                    if result_callback:
                        result_callback(req.request_id, {"error": str(e)})

    async def _process_single_request(self, request_id: str, request_data: Dict[str, Any]) -> Any:
        """Process a single request using the engine."""
        # Extract request parameters
        prompt = request_data.get("prompt")
        sampling_params = request_data.get("sampling_params", {})
        output_modalities = request_data.get("output_modalities")

        # Call engine generate method
        results = []
        async for result in self.engine.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=[sampling_params],
            output_modalities=output_modalities,
        ):
            results.append(result)

        # Return the final result
        return results[-1] if results else None

    async def shutdown(self) -> None:
        """Shutdown the batch processor."""
        if self.executor:
            self.executor.shutdown(wait=True)

    def get_active_batch_count(self) -> int:
        """Get the number of currently active batches."""
        # This is approximate since semaphore doesn't expose current count
        return self.max_concurrent_batches - self._semaphore._value