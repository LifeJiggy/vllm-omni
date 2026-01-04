"""
Transition Manager for Real-time Model Switching

This module provides seamless transitions during model switching with zero downtime,
ensuring in-flight requests complete with old models while new requests use switched models.
"""

import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vllm.logger import init_logger

from vllm_omni.model_executor.models.dynamic_registry import ModelInstance

logger = init_logger(__name__)


class TransitionState(Enum):
    """States for a model transition."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class TransitionRequest:
    """Represents a request during transition."""

    request_id: str
    model_id: str
    from_version: str
    to_version: str
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    assigned_model: ModelInstance | None = None

    @property
    def duration(self) -> float | None:
        """Get request duration if completed."""
        if self.completed_at:
            return self.completed_at - self.created_at
        return None

    def complete(self, assigned_model: ModelInstance):
        """Mark request as completed."""
        self.completed_at = time.time()
        self.assigned_model = assigned_model


@dataclass
class TransitionOperation:
    """Represents a model transition operation."""

    transition_id: str
    model_id: str
    from_version: str
    to_version: str
    state: TransitionState = TransitionState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    active_requests: set[str] = field(default_factory=set)
    completed_requests: list[TransitionRequest] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        """Get transition duration if completed."""
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        return None

    @property
    def total_requests(self) -> int:
        """Get total number of requests in this transition."""
        return len(self.active_requests) + len(self.completed_requests)

    def start(self):
        """Mark transition as started."""
        self.state = TransitionState.ACTIVE
        self.started_at = time.time()

    def complete(self):
        """Mark transition as completed."""
        self.state = TransitionState.COMPLETED
        self.completed_at = time.time()

    def fail(self, reason: str):
        """Mark transition as failed."""
        self.state = TransitionState.FAILED
        self.completed_at = time.time()
        self.metadata["failure_reason"] = reason

    def rollback(self):
        """Mark transition as rolled back."""
        self.state = TransitionState.ROLLED_BACK
        self.completed_at = time.time()


class RequestTracker:
    """Tracks requests during model transitions."""

    def __init__(self, max_history_size: int = 10000):
        """
        Initialize request tracker.

        Args:
            max_history_size: Maximum number of completed requests to keep in history
        """
        self.active_requests: dict[str, TransitionRequest] = {}
        self.completed_requests: list[TransitionRequest] = []
        self.max_history_size = max_history_size
        self._lock = threading.RLock()

    def track_request(self, request: TransitionRequest):
        """
        Start tracking a request.

        Args:
            request: Request to track
        """
        with self._lock:
            self.active_requests[request.request_id] = request

    def complete_request(self, request_id: str, assigned_model: ModelInstance):
        """
        Mark a request as completed.

        Args:
            request_id: Request ID
            assigned_model: Model that handled the request
        """
        with self._lock:
            if request_id in self.active_requests:
                request = self.active_requests.pop(request_id)
                request.complete(assigned_model)

                # Add to completed history
                self.completed_requests.append(request)

                # Maintain history size limit
                if len(self.completed_requests) > self.max_history_size:
                    self.completed_requests.pop(0)

    def get_active_requests(self, model_id: str | None = None) -> list[TransitionRequest]:
        """
        Get active requests, optionally filtered by model.

        Args:
            model_id: Optional model ID filter

        Returns:
            List of active requests
        """
        with self._lock:
            requests = list(self.active_requests.values())
            if model_id:
                requests = [r for r in requests if r.model_id == model_id]
            return requests

    def get_request_stats(self) -> dict[str, Any]:
        """Get request tracking statistics."""
        with self._lock:
            return {
                "active_requests": len(self.active_requests),
                "completed_requests": len(self.completed_requests),
                "max_history_size": self.max_history_size,
            }


class TransitionManager:
    """
    Manages seamless transitions during model switching with zero downtime.

    This class provides:
    - In-flight request tracking and completion
    - Graceful request routing during transitions
    - Traffic redirection control
    - Failure recovery and rollback
    """

    def __init__(
        self,
        request_tracker: RequestTracker | None = None,
        max_transition_time: float = 300.0,  # 5 minutes
        request_timeout: float = 60.0,
    ):  # 1 minute
        """
        Initialize transition manager.

        Args:
            request_tracker: Request tracker instance
            max_transition_time: Maximum time for transitions in seconds
            request_timeout: Timeout for individual requests in seconds
        """
        self.request_tracker = request_tracker or RequestTracker()
        self.max_transition_time = max_transition_time
        self.request_timeout = request_timeout

        # Transition storage
        self.active_transitions: dict[str, TransitionOperation] = {}
        self.completed_transitions: list[TransitionOperation] = []
        self.max_completed_history = 100

        # Model routing during transitions
        self.transition_routing: dict[
            str, tuple[ModelInstance, ModelInstance]
        ] = {}  # model_id -> (old_model, new_model)

        # Threading
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="transition-mgr")

        # Background cleanup
        self._cleanup_task: asyncio.Task | None = None

        logger.info(f"Initialized TransitionManager with max_transition_time={max_transition_time}s")

    def start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def stop_cleanup_task(self):
        """Stop background cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

    async def _cleanup_loop(self):
        """Background cleanup loop for expired transitions."""
        while True:
            try:
                await asyncio.sleep(60.0)  # Check every minute
                await self._cleanup_expired_transitions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_expired_transitions(self):
        """Clean up expired transitions."""
        current_time = time.time()
        expired_transitions = []

        with self._lock:
            for transition_id, transition in self.active_transitions.items():
                if (
                    transition.state == TransitionState.ACTIVE
                    and current_time - transition.created_at > self.max_transition_time
                ):
                    expired_transitions.append(transition_id)

            for transition_id in expired_transitions:
                transition = self.active_transitions.pop(transition_id)
                transition.fail("Transition timed out")
                self.completed_transitions.append(transition)

                # Maintain history size
                if len(self.completed_transitions) > self.max_completed_history:
                    self.completed_transitions.pop(0)

                logger.warning(f"Transition {transition_id} timed out and was cleaned up")

        if expired_transitions:
            logger.info(f"Cleaned up {len(expired_transitions)} expired transitions")

    def begin_transition(self, model_id: str, from_model: ModelInstance, to_model: ModelInstance) -> str:
        """
        Begin a model transition.

        Args:
            model_id: Model identifier
            from_model: Current model instance
            to_model: Target model instance

        Returns:
            Transition ID
        """
        transition_id = str(uuid.uuid4())

        transition = TransitionOperation(
            transition_id=transition_id, model_id=model_id, from_version=from_model.version, to_version=to_model.version
        )

        with self._lock:
            self.active_transitions[transition_id] = transition
            self.transition_routing[model_id] = (from_model, to_model)

        transition.start()
        logger.info(
            f"Began transition {transition_id} for model {model_id}: {from_model.version} -> {to_model.version}"
        )

        return transition_id

    def handle_request_routing(self, model_id: str, request_id: str) -> ModelInstance:
        """
        Route a request to the appropriate model during transition.

        Args:
            model_id: Model identifier
            request_id: Unique request identifier

        Returns:
            Model instance to handle the request
        """
        with self._lock:
            # Check if there's an active transition for this model
            if model_id in self.transition_routing:
                old_model, new_model = self.transition_routing[model_id]

                # Create transition request for tracking
                transition_request = TransitionRequest(
                    request_id=request_id,
                    model_id=model_id,
                    from_version=old_model.version,
                    to_version=new_model.version,
                )

                # Track the request
                self.request_tracker.track_request(transition_request)

                # Find the active transition and add this request
                for transition in self.active_transitions.values():
                    if transition.model_id == model_id and transition.state == TransitionState.ACTIVE:
                        transition.active_requests.add(request_id)
                        break

                # For now, route all requests to the old model during transition
                # In a more sophisticated implementation, you could implement
                # gradual traffic shifting based on transition strategy
                return old_model
            else:
                # No active transition, return None to indicate normal routing
                return None

    def complete_request(self, request_id: str, assigned_model: ModelInstance):
        """
        Mark a request as completed.

        Args:
            request_id: Request identifier
            assigned_model: Model that handled the request
        """
        self.request_tracker.complete_request(request_id, assigned_model)

        # Update transition tracking
        with self._lock:
            for transition in self.active_transitions.values():
                if request_id in transition.active_requests:
                    transition.active_requests.remove(request_id)
                    transition.completed_requests.append(self.request_tracker.completed_requests[-1])
                    break

    def complete_transition(self, transition_id: str) -> bool:
        """
        Complete a transition.

        Args:
            transition_id: Transition identifier

        Returns:
            True if successfully completed
        """
        with self._lock:
            if transition_id not in self.active_transitions:
                logger.error(f"Transition {transition_id} not found")
                return False

            transition = self.active_transitions.pop(transition_id)

            # Check if all requests are completed
            if transition.active_requests:
                logger.warning(
                    f"Completing transition {transition_id} with {len(transition.active_requests)} active requests"
                )

            # Remove routing rule
            if transition.model_id in self.transition_routing:
                del self.transition_routing[transition.model_id]

            transition.complete()
            self.completed_transitions.append(transition)

            # Maintain history size
            if len(self.completed_transitions) > self.max_completed_history:
                self.completed_transitions.pop(0)

            logger.info(f"Completed transition {transition_id} for model {transition.model_id}")
            return True

    def rollback_transition(self, transition_id: str) -> bool:
        """
        Rollback a transition.

        Args:
            transition_id: Transition identifier

        Returns:
            True if successfully rolled back
        """
        with self._lock:
            if transition_id not in self.active_transitions:
                logger.error(f"Transition {transition_id} not found for rollback")
                return False

            transition = self.active_transitions.pop(transition_id)

            # Remove routing rule (rollback to old model)
            if transition.model_id in self.transition_routing:
                del self.transition_routing[transition.model_id]

            transition.rollback()
            self.completed_transitions.append(transition)

            # Maintain history size
            if len(self.completed_transitions) > self.max_completed_history:
                self.completed_transitions.pop(0)

            logger.info(f"Rolled back transition {transition_id} for model {transition.model_id}")
            return True

    def get_transition_status(self, transition_id: str) -> dict[str, Any] | None:
        """
        Get status of a transition.

        Args:
            transition_id: Transition identifier

        Returns:
            Transition status dictionary or None if not found
        """
        with self._lock:
            transition = self.active_transitions.get(transition_id) or next(
                (t for t in self.completed_transitions if t.transition_id == transition_id), None
            )

            if not transition:
                return None

            return {
                "transition_id": transition.transition_id,
                "model_id": transition.model_id,
                "from_version": transition.from_version,
                "to_version": transition.to_version,
                "state": transition.state.value,
                "created_at": transition.created_at,
                "started_at": transition.started_at,
                "completed_at": transition.completed_at,
                "duration": transition.duration,
                "active_requests": len(transition.active_requests),
                "completed_requests": len(transition.completed_requests),
                "total_requests": transition.total_requests,
                "metadata": transition.metadata,
            }

    def list_active_transitions(self) -> list[dict[str, Any]]:
        """
        List all active transitions.

        Returns:
            List of active transition status dictionaries
        """
        with self._lock:
            return [self.get_transition_status(tid) for tid in self.active_transitions.keys()]

    def get_transition_stats(self) -> dict[str, Any]:
        """
        Get transition statistics.

        Returns:
            Dictionary with transition statistics
        """
        with self._lock:
            active_count = len(self.active_transitions)
            completed_count = len(self.completed_transitions)

            total_requests = sum(t.total_requests for t in self.completed_transitions)
            avg_duration = 0.0
            if completed_count > 0:
                durations = [t.duration for t in self.completed_transitions if t.duration]
                if durations:
                    avg_duration = sum(durations) / len(durations)

            return {
                "active_transitions": active_count,
                "completed_transitions": completed_count,
                "total_requests_handled": total_requests,
                "average_transition_duration": avg_duration,
                "max_completed_history": self.max_completed_history,
                "active_routing_rules": len(self.transition_routing),
            }

    def is_transition_active(self, model_id: str) -> bool:
        """
        Check if there's an active transition for a model.

        Args:
            model_id: Model identifier

        Returns:
            True if transition is active
        """
        with self._lock:
            return model_id in self.transition_routing

    def __del__(self):
        """Cleanup on destruction."""
        self.stop_cleanup_task()
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
