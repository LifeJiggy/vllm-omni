# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Dict, List, Optional, Set
from collections import defaultdict

from .request_queue import QueuedRequest
from vllm.logger import init_logger

logger = init_logger(__name__)


class MclodalityBatcher:
    """Groups requests by modality for efficient batching.

    Supports smart scheduling that considers modality compatibility
    and resource requirements.
    """

    def __init__(self, max_batch_size: int = 32, batch_timeout: float = 0.1):
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout

        # Modality compatibility matrix
        # True means modalities can be batched together
        self._compatibility_matrix: Dict[str, Set[str]] = {
            "text": {"text", "image", "video"},
            "image": {"text", "image"},
            "video": {"text", "video"},
            "audio": {"text", "audio"},
        }

    def can_batch_together(self, modality1: str, modality2: str) -> bool:
        """Check if two modalities can be batched together."""
        return modality2 in self._compatibility_matrix.get(modality1, set())

    def create_batches(self, requests: List[QueuedRequest]) -> List[List[QueuedRequest]]:
        """Create batches from a list of requests.

        Groups requests by compatible modalities and respects batch size limits.

        Args:
            requests: List of requests to batch

        Returns:
            List of batches, where each batch is a list of compatible requests
        """
        if not requests:
            return []

        # Group requests by modality
        modality_groups: Dict[str, List[QueuedRequest]] = defaultdict(list)
        for req in requests:
            modality_groups[req.modality].append(req)

        batches = []

        # First, try to create homogeneous batches (same modality)
        for modality, reqs in modality_groups.items():
            if len(reqs) <= self.max_batch_size:
                batches.append(reqs)
            else:
                # Split large groups into smaller batches
                for i in range(0, len(reqs), self.max_batch_size):
                    batch = reqs[i:i + self.max_batch_size]
                    batches.append(batch)

        # Then, try to merge compatible heterogeneous batches
        merged_batches = self._merge_compatible_batches(batches)

        return merged_batches

    def _merge_compatible_batches(self, batches: List[List[QueuedRequest]]) -> List[List[QueuedRequest]]:
        """Merge compatible batches to optimize resource usage."""
        if len(batches) <= 1:
            return batches

        merged = []
        used = [False] * len(batches)

        for i, batch1 in enumerate(batches):
            if used[i]:
                continue

            current_batch = batch1[:]
            used[i] = True

            # Try to merge with other compatible batches
            for j, batch2 in enumerate(batches):
                if used[j] or len(current_batch) + len(batch2) > self.max_batch_size:
                    continue

                # Check if all modalities in batch2 are compatible with current_batch
                if self._batches_compatible(current_batch, batch2):
                    current_batch.extend(batch2)
                    used[j] = True

            merged.append(current_batch)

        return merged

    def _batches_compatible(self, batch1: List[QueuedRequest], batch2: List[QueuedRequest]) -> bool:
        """Check if two batches have compatible modalities."""
        batch1_modalities = {req.modality for req in batch1}
        batch2_modalities = {req.modality for req in batch2}

        # Check compatibility between all modality pairs
        for mod1 in batch1_modalities:
            for mod2 in batch2_modalities:
                if not self.can_batch_together(mod1, mod2):
                    return False

        return True

    def get_optimal_batch_size(self, modality: str) -> int:
        """Get optimal batch size for a specific modality."""
        # Different modalities may have different optimal batch sizes
        modality_batch_sizes = {
            "text": self.max_batch_size,
            "image": min(self.max_batch_size, 8),  # Images are more memory intensive
            "video": min(self.max_batch_size, 4),  # Videos are very memory intensive
            "audio": min(self.max_batch_size, 16),  # Audio is moderately intensive
        }

        return modality_batch_sizes.get(modality, self.max_batch_size)

    def should_flush_batch(self, batch: List[QueuedRequest], current_time: float) -> bool:
        """Determine if a batch should be flushed for processing.

        Args:
            batch: Current batch of requests
            current_time: Current timestamp

        Returns:
            True if batch should be processed now
        """
        if not batch:
            return False

        # Flush if batch is full
        if len(batch) >= self.max_batch_size:
            return True

        # Flush if timeout exceeded
        oldest_request_time = min(req.timestamp for req in batch)
        if current_time - oldest_request_time >= self.batch_timeout:
            return True

        return False