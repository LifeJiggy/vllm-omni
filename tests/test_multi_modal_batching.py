# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from vllm_omni.batching.request_queue import RequestQueue, RequestPriority
from vllm_omni.batching.modality_batcher import ModalityBatcher
from vllm_omni.batching.batch_processor import BatchProcessor
from vllm_omni.batching.scheduler import MultiModalBatchingScheduler


class TestRequestQueue:
    @pytest.mark.asyncio
    async def test_queue_priority_ordering(self):
        """Test that requests are dequeued in priority order."""
        queue = RequestQueue(max_size=10)

        # Add requests with different priorities
        await queue.put("low_req", {"prompt": "low"}, priority=RequestPriority.LOW)
        await queue.put("high_req", {"prompt": "high"}, priority=RequestPriority.HIGH)
        await queue.put("normal_req", {"prompt": "normal"}, priority=RequestPriority.NORMAL)

        # Should get high priority first
        req1 = await queue.get()
        assert req1.request_id == "high_req"

        # Then normal
        req2 = await queue.get()
        assert req2.request_id == "normal_req"

        # Then low
        req3 = await queue.get()
        assert req3.request_id == "low_req"

    @pytest.mark.asyncio
    async def test_queue_modality_tracking(self):
        """Test modality tracking in queue."""
        queue = RequestQueue(max_size=10)

        await queue.put("text_req", {"prompt": "text"}, modality="text")
        await queue.put("image_req", {"prompt": "image"}, modality="image")

        modalities = queue.get_pending_modalities()
        assert "text" in modalities
        assert "image" in modalities

        text_reqs = queue.get_requests_by_modality("text")
        assert len(text_reqs) == 1
        assert text_reqs[0].request_id == "text_req"


class TestModalityBatcher:
    def test_create_batches_homogeneous(self):
        """Test batching of homogeneous modalities."""
        batcher = ModalityBatcher(max_batch_size=3)

        requests = [
            MagicMock(modality="text", request_id="req1"),
            MagicMock(modality="text", request_id="req2"),
            MagicMock(modality="text", request_id="req3"),
            MagicMock(modality="text", request_id="req4"),
        ]

        batches = batcher.create_batches(requests)

        # Should create batches of size 3 and 1
        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 1

    def test_create_batches_heterogeneous(self):
        """Test batching of mixed modalities."""
        batcher = ModalityBatcher(max_batch_size=4)

        requests = [
            MagicMock(modality="text", request_id="text1"),
            MagicMock(modality="image", request_id="image1"),
            MagicMock(modality="text", request_id="text2"),
            MagicMock(modality="video", request_id="video1"),
        ]

        batches = batcher.create_batches(requests)

        # Should create separate batches for incompatible modalities
        assert len(batches) >= 1

        # Check that incompatible modalities are not batched together
        for batch in batches:
            modalities = {req.modality for req in batch}
            if len(modalities) > 1:
                # If multiple modalities in batch, they must be compatible
                for mod1 in modalities:
                    for mod2 in modalities:
                        assert batcher.can_batch_together(mod1, mod2)


class TestBatchProcessor:
    @pytest.mark.asyncio
    async def test_process_batch(self):
        """Test batch processing."""
        mock_engine = AsyncMock()
        mock_engine.generate.return_value = [
            MagicMock(request_output="result1"),
            MagicMock(request_output="result2"),
        ]

        processor = BatchProcessor(mock_engine, max_concurrent_batches=1)

        batch = [
            MagicMock(request_id="req1", data={"prompt": "test1"}),
            MagicMock(request_id="req2", data={"prompt": "test2"}),
        ]

        results_received = []

        def result_callback(req_id, result):
            results_received.append((req_id, result))

        await processor.process_batch(batch, result_callback)

        # Should have called engine.generate for each request
        assert mock_engine.generate.call_count == 2

        # Should have received results
        assert len(results_received) == 2


class TestMultiModalBatchingScheduler:
    @pytest.mark.asyncio
    async def test_submit_and_process_request(self):
        """Test end-to-end request submission and processing."""
        mock_engine = AsyncMock()
        mock_engine.generate.return_value = [MagicMock(request_output="result")]

        scheduler = MultiModalBatchingScheduler(
            engine=mock_engine,
            max_batch_size=2,
            batch_timeout=0.01,  # Short timeout for testing
        )

        await scheduler.start()

        # Submit a request
        result_future = asyncio.Future()

        def result_callback(req_id, result):
            result_future.set_result((req_id, result))

        await scheduler.submit_request(
            request_id="test_req",
            prompt="test prompt",
            modality="text",
            callback=result_callback,
        )

        # Wait for result with timeout
        try:
            req_id, result = await asyncio.wait_for(result_future, timeout=1.0)
            assert req_id == "test_req"
            assert result.request_output == "result"
        except asyncio.TimeoutError:
            pytest.fail("Request processing timed out")

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_scheduler_stats(self):
        """Test scheduler statistics."""
        mock_engine = AsyncMock()
        scheduler = MultiModalBatchingScheduler(engine=mock_engine)

        stats = scheduler.get_queue_stats()
        assert "queue_size" in stats
        assert "pending_modalities" in stats
        assert "active_batches" in stats