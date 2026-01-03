# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable

from .request_queue import RequestQueue, QueuedRequest, RequestPriority
from .modality_batcher import ModalityBatcher
from .batch_processor import BatchProcessor
from vllm.logger import init_logger

logger = init_logger(__name__)


class MultiModalBatchingScheduler:
    """Main scheduler for multi-modal request batching.

    Coordinates request queuing, batching, and processing with support
    for different modalities and priority levels.
    """

    def __init__(
        self,
        engine: Any,
        max_batch_size: int = 32,
        batch_timeout: float = 0.1,
        max_queue_size: int = 1000,
        max_concurrent_batches: int = 4,
    ):
        self.request_queue = RequestQueue(max_size=max_queue_size)
        self.modality_batcher = ModalityBatcher(
            max_batch_size=max_batch_size,
            batch_timeout=batch_timeout,
        )
        self.batch_processor = BatchProcessor(
            engine=engine,
            max_concurrent_batches=max_concurrent_batches,
        )

        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._result_callbacks: Dict[str, Callable] = {}

    async def start(self) -> None:
        """Start the batching scheduler."""
        if self._running:
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Multi-modal batching scheduler started")

    async def stop(self) -> None:
        """Stop the batching scheduler."""
        if not self._running:
            return

        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        await self.batch_processor.shutdown()
        logger.info("Multi-modal batching scheduler stopped")

    async def submit_request(
        self,
        request_id: str,
        prompt: Any,
        sampling_params: Optional[Dict[str, Any]] = None,
        modality: str = "text",
        priority: RequestPriority = RequestPriority.NORMAL,
        output_modalities: Optional[List[str]] = None,
        callback: Optional[Callable[[str, Any], None]] = None,
    ) -> None:
        """Submit a request for batching and processing.

        Args:
            request_id: Unique request identifier
            prompt: The input prompt/data
            sampling_params: Sampling parameters for generation
            modality: Request modality (text, image, video, etc.)
            priority: Request priority level
            output_modalities: Expected output modalities
            callback: Optional callback for result notification
        """
        request_data = {
            "prompt": prompt,
            "sampling_params": sampling_params or {},
            "output_modalities": output_modalities,
        }

        if callback:
            self._result_callbacks[request_id] = callback

        await self.request_queue.put(
            request_id=request_id,
            data=request_data,
            modality=modality,
            priority=priority,
        )

        logger.debug(f"Submitted request {request_id} with modality {modality}")

    async def _scheduler_loop(self) -> None:
        """Main scheduling loop that manages batching and processing."""
        try:
            while self._running:
                # Collect pending requests
                pending_requests = []
                batch_start_time = time.time()

                # Get requests until we have enough for a batch or timeout
                while self._running:
                    try:
                        # Try to get a request with timeout
                        request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=self.modality_batcher.batch_timeout
                        )
                        pending_requests.append(request)

                        # Check if we should flush the current batch
                        if self.modality_batcher.should_flush_batch(
                            pending_requests, time.time()
                        ):
                            break

                    except asyncio.TimeoutError:
                        # Timeout reached, process what we have
                        if pending_requests:
                            break
                        # No requests, continue waiting
                        continue

                if not pending_requests:
                    continue

                # Create batches from pending requests
                batches = self.modality_batcher.create_batches(pending_requests)

                # Process each batch
                for batch in batches:
                    if not batch:
                        continue

                    asyncio.create_task(
                        self._process_batch_async(batch)
                    )

        except Exception as e:
            logger.error(f"Scheduler loop error: {e}")
            raise

    async def _process_batch_async(self, batch: List[QueuedRequest]) -> None:
        """Process a batch asynchronously."""
        try:
            await self.batch_processor.process_batch(
                batch=batch,
                result_callback=self._handle_result,
            )
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Handle errors for all requests in batch
            for req in batch:
                self._handle_result(req.request_id, {"error": str(e)})

    def _handle_result(self, request_id: str, result: Any) -> None:
        """Handle result for a completed request."""
        callback = self._result_callbacks.pop(request_id, None)
        if callback:
            try:
                callback(request_id, result)
            except Exception as e:
                logger.error(f"Error in result callback for {request_id}: {e}")

        logger.debug(f"Handled result for request {request_id}")

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the current queue state."""
        return {
            "queue_size": self.request_queue.qsize(),
            "pending_modalities": list(self.request_queue.get_pending_modalities()),
            "active_batches": self.batch_processor.get_active_batch_count(),
        }

    async def clear_queue(self) -> None:
        """Clear all pending requests from the queue."""
        await self.request_queue.clear()
        self._result_callbacks.clear()
        logger.info("Cleared request queue and callbacks")