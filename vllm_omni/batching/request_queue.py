# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class RequestPriority(Enum):
    """Priority levels for requests."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass(order=True)
class QueuedRequest:
    """A request in the queue with priority and metadata."""

    priority: int
    timestamp: float
    request_id: str
    data: dict[str, Any] = field(compare=False)
    modality: str = field(compare=False, default="text")
    future: asyncio.Future | None = field(compare=False, default=None)

    def __post_init__(self):
        # For heapq, we want higher priority (larger number) to come first
        # So we negate the priority for the comparison
        self.priority = -self.priority


class RequestQueue:
    """Priority queue for managing incoming requests.

    Supports different priority levels and modality-based organization.
    """

    def __init__(self, max_size: int = 1000):
        self._queue: list[QueuedRequest] = []
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._not_full = asyncio.Condition(self._lock)

    async def put(
        self,
        request_id: str,
        data: dict[str, Any],
        modality: str = "text",
        priority: RequestPriority = RequestPriority.NORMAL,
    ) -> None:
        """Add a request to the queue.

        Args:
            request_id: Unique identifier for the request
            data: Request data (prompt, params, etc.)
            modality: Request modality (text, image, video, etc.)
            priority: Request priority level
        """
        async with self._lock:
            while len(self._queue) >= self._max_size:
                await self._not_full.wait()

            request = QueuedRequest(
                priority=priority.value,
                timestamp=time.time(),
                request_id=request_id,
                data=data,
                modality=modality,
            )

            heapq.heappush(self._queue, request)
            self._not_empty.notify()

            logger.debug(f"Queued request {request_id} with priority {priority.name}")

    async def get(self) -> QueuedRequest:
        """Get the highest priority request from the queue."""
        async with self._lock:
            while not self._queue:
                await self._not_empty.wait()

            request = heapq.heappop(self._queue)
            self._not_full.notify()

            logger.debug(f"Dequeued request {request.request_id}")
            return request

    async def peek(self) -> QueuedRequest | None:
        """Peek at the highest priority request without removing it."""
        async with self._lock:
            if not self._queue:
                return None
            return self._queue[0]

    def qsize(self) -> int:
        """Get the current queue size."""
        return len(self._queue)

    def empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self._queue) == 0

    async def clear(self) -> None:
        """Clear all requests from the queue."""
        async with self._lock:
            self._queue.clear()
            self._not_full.notify_all()

    def get_requests_by_modality(self, modality: str) -> list[QueuedRequest]:
        """Get all requests of a specific modality (for batching)."""
        return [req for req in self._queue if req.modality == modality]

    def get_pending_modalities(self) -> set[str]:
        """Get set of all modalities currently in the queue."""
        return {req.modality for req in self._queue}
