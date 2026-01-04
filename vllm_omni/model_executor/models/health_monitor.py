"""
Health Monitor for Real-time Model Switching

This module provides health monitoring capabilities for model instances,
including performance metrics collection, error rate tracking, and automatic alerting.
"""

import asyncio
import logging
import statistics
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set
from concurrent.futures import ThreadPoolExecutor

from vllm_omni.model_executor.models.dynamic_registry import ModelInstance
from vllm.logger import init_logger

logger = init_logger(__name__)


class HealthStatus(Enum):
    """Health status levels for model instances."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetrics:
    """Health metrics for a model instance."""
    model_id: str
    version: str
    timestamp: float = field(default_factory=time.time)

    # Performance metrics
    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')

    # Resource metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_memory_mb: float = 0.0

    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return self.error_count / max(self.request_count, 1)

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        return self.total_latency_ms / max(self.request_count, 1)

    @property
    def throughput_rps(self) -> float:
        """Calculate throughput (requests per second) - requires time window context."""
        # This would be calculated by the monitor over a time window
        return 0.0


@dataclass
class Alert:
    """Alert for health issues."""
    alert_id: str
    model_id: str
    version: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def resolve(self):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = time.time()


class AlertSystem:
    """Alert system for health monitoring."""

    def __init__(self, alert_handlers: Optional[List[Callable[[Alert], None]]] = None):
        """
        Initialize alert system.

        Args:
            alert_handlers: List of functions to call when alerts are triggered
        """
        self.alert_handlers = alert_handlers or []
        self.active_alerts: Dict[str, Alert] = {}
        self.resolved_alerts: List[Alert] = []
        self.max_resolved_history = 1000

    def trigger_alert(self, alert: Alert):
        """
        Trigger a new alert.

        Args:
            alert: Alert to trigger
        """
        # Check if similar alert already exists
        existing_alert = self.active_alerts.get(alert.alert_id)
        if existing_alert and not existing_alert.resolved:
            # Update existing alert
            existing_alert.timestamp = alert.timestamp
            existing_alert.metadata.update(alert.metadata)
            logger.debug(f"Updated existing alert: {alert.alert_id}")
            return

        # Add new alert
        self.active_alerts[alert.alert_id] = alert

        # Call handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

        logger.warning(f"Alert triggered: {alert.title} - {alert.message}")

    def resolve_alert(self, alert_id: str):
        """
        Resolve an alert.

        Args:
            alert_id: Alert identifier
        """
        alert = self.active_alerts.get(alert_id)
        if alert:
            alert.resolve()
            self.resolved_alerts.append(alert)
            del self.active_alerts[alert_id]

            # Maintain history size
            if len(self.resolved_alerts) > self.max_resolved_history:
                self.resolved_alerts.pop(0)

            logger.info(f"Alert resolved: {alert_id}")

    def get_active_alerts(self, model_id: Optional[str] = None) -> List[Alert]:
        """
        Get active alerts.

        Args:
            model_id: Optional model ID filter

        Returns:
            List of active alerts
        """
        alerts = list(self.active_alerts.values())
        if model_id:
            alerts = [a for a in alerts if a.model_id == model_id]
        return alerts

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = 0

        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1

        return {
            "active_alerts": len(self.active_alerts),
            "resolved_alerts": len(self.resolved_alerts),
            "alerts_by_severity": severity_counts
        }


class HealthMonitor:
    """
    Health monitor for model instances.

    This class monitors model performance, detects health issues, and triggers alerts.
    """

    def __init__(self,
                 alert_system: Optional[AlertSystem] = None,
                 monitoring_interval: float = 30.0,
                 metrics_history_size: int = 100,
                 health_check_timeout: float = 10.0):
        """
        Initialize health monitor.

        Args:
            alert_system: Alert system instance
            monitoring_interval: Interval between health checks in seconds
            metrics_history_size: Number of metrics samples to keep in history
            health_check_timeout: Timeout for health checks in seconds
        """
        self.alert_system = alert_system or AlertSystem()
        self.monitoring_interval = monitoring_interval
        self.metrics_history_size = metrics_history_size
        self.health_check_timeout = health_check_timeout

        # Metrics storage
        self.metrics_history: Dict[str, deque] = {}  # model_key -> deque of HealthMetrics
        self.current_metrics: Dict[str, HealthMetrics] = {}  # model_key -> current HealthMetrics

        # Health thresholds
        self.thresholds = {
            "error_rate_warning": 0.05,    # 5%
            "error_rate_critical": 0.10,   # 10%
            "latency_p95_warning": 5000,   # 5 seconds
            "latency_p95_critical": 10000, # 10 seconds
            "memory_usage_warning": 0.8,   # 80%
            "memory_usage_critical": 0.95, # 95%
        }

        # Threading
        self._lock = threading.RLock()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="health-monitor")

        # Health status cache
        self.health_status_cache: Dict[str, tuple] = {}  # model_key -> (status, timestamp)

        logger.info(f"Initialized HealthMonitor with monitoring_interval={monitoring_interval}s")

    def start_monitoring(self):
        """Start the monitoring loop."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    def stop_monitoring(self):
        """Stop the monitoring loop."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval)
                await self._perform_health_checks()
                await self._analyze_health_trends()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    async def _perform_health_checks(self):
        """Perform health checks on all monitored models."""
        # This would be implemented to check actual model instances
        # For now, we'll simulate health checks
        pass

    async def _analyze_health_trends(self):
        """Analyze health trends and trigger alerts."""
        with self._lock:
            for model_key, history in self.metrics_history.items():
                if len(history) < 2:
                    continue

                # Analyze recent metrics
                recent_metrics = list(history)[-10:]  # Last 10 samples

                # Check error rate trends
                error_rates = [m.error_rate for m in recent_metrics]
                if error_rates:
                    avg_error_rate = statistics.mean(error_rates)
                    if avg_error_rate > self.thresholds["error_rate_critical"]:
                        self._trigger_health_alert(
                            model_key, AlertSeverity.CRITICAL,
                            "High Error Rate",
                            f"Error rate is {avg_error_rate:.2%}, above critical threshold"
                        )
                    elif avg_error_rate > self.thresholds["error_rate_warning"]:
                        self._trigger_health_alert(
                            model_key, AlertSeverity.WARNING,
                            "Elevated Error Rate",
                            f"Error rate is {avg_error_rate:.2%}, above warning threshold"
                        )

                # Check latency trends
                latencies = [m.average_latency_ms for m in recent_metrics if m.average_latency_ms > 0]
                if latencies:
                    p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
                    if p95_latency > self.thresholds["latency_p95_critical"]:
                        self._trigger_health_alert(
                            model_key, AlertSeverity.CRITICAL,
                            "High Latency",
                            f"P95 latency is {p95_latency:.0f}ms, above critical threshold"
                        )
                    elif p95_latency > self.thresholds["latency_p95_warning"]:
                        self._trigger_health_alert(
                            model_key, AlertSeverity.WARNING,
                            "Elevated Latency",
                            f"P95 latency is {p95_latency:.0f}ms, above warning threshold"
                        )

    def _trigger_health_alert(self, model_key: str, severity: AlertSeverity,
                             title: str, message: str):
        """
        Trigger a health alert.

        Args:
            model_key: Model key (model_id_version)
            severity: Alert severity
            title: Alert title
            message: Alert message
        """
        model_id, version = model_key.split('_', 1)
        alert_id = f"health_{model_key}_{title.lower().replace(' ', '_')}"

        alert = Alert(
            alert_id=alert_id,
            model_id=model_id,
            version=version,
            severity=severity,
            title=title,
            message=message,
            metadata={"model_key": model_key}
        )

        self.alert_system.trigger_alert(alert)

    def record_request(self, model_key: str, latency_ms: float, error: bool = False):
        """
        Record a request for health monitoring.

        Args:
            model_key: Model key (model_id_version)
            latency_ms: Request latency in milliseconds
            error: Whether the request resulted in an error
        """
        with self._lock:
            # Initialize metrics if needed
            if model_key not in self.current_metrics:
                model_id, version = model_key.split('_', 1)
                self.current_metrics[model_key] = HealthMetrics(
                    model_id=model_id,
                    version=version
                )

            if model_key not in self.metrics_history:
                self.metrics_history[model_key] = deque(maxlen=self.metrics_history_size)

            metrics = self.current_metrics[model_key]

            # Update metrics
            metrics.request_count += 1
            metrics.total_latency_ms += latency_ms
            metrics.max_latency_ms = max(metrics.max_latency_ms, latency_ms)
            metrics.min_latency_ms = min(metrics.min_latency_ms, latency_ms)

            if error:
                metrics.error_count += 1

    def update_resource_metrics(self, model_key: str, memory_mb: float,
                               cpu_percent: float, gpu_memory_mb: float):
        """
        Update resource usage metrics.

        Args:
            model_key: Model key
            memory_mb: Memory usage in MB
            cpu_percent: CPU usage percentage
            gpu_memory_mb: GPU memory usage in MB
        """
        with self._lock:
            if model_key in self.current_metrics:
                metrics = self.current_metrics[model_key]
                metrics.memory_usage_mb = memory_mb
                metrics.cpu_usage_percent = cpu_percent
                metrics.gpu_memory_mb = gpu_memory_mb

    def snapshot_metrics(self, model_key: str):
        """
        Take a snapshot of current metrics and add to history.

        Args:
            model_key: Model key
        """
        with self._lock:
            if model_key in self.current_metrics:
                metrics = self.current_metrics[model_key]

                # Create a copy for history
                history_metrics = HealthMetrics(
                    model_id=metrics.model_id,
                    version=metrics.version,
                    timestamp=metrics.timestamp,
                    request_count=metrics.request_count,
                    error_count=metrics.error_count,
                    total_latency_ms=metrics.total_latency_ms,
                    max_latency_ms=metrics.max_latency_ms,
                    min_latency_ms=metrics.min_latency_ms,
                    memory_usage_mb=metrics.memory_usage_mb,
                    cpu_usage_percent=metrics.cpu_usage_percent,
                    gpu_memory_mb=metrics.gpu_memory_mb,
                    custom_metrics=metrics.custom_metrics.copy()
                )

                # Add to history
                if model_key not in self.metrics_history:
                    self.metrics_history[model_key] = deque(maxlen=self.metrics_history_size)
                self.metrics_history[model_key].append(history_metrics)

                # Reset current metrics for next snapshot period
                self.current_metrics[model_key] = HealthMetrics(
                    model_id=metrics.model_id,
                    version=metrics.version
                )

    def get_health_status(self, model_key: str) -> HealthStatus:
        """
        Get the health status of a model.

        Args:
            model_key: Model key

        Returns:
            Health status
        """
        with self._lock:
            # Check cache first
            if model_key in self.health_status_cache:
                status, timestamp = self.health_status_cache[model_key]
                if time.time() - timestamp < self.monitoring_interval:
                    return status

            # Calculate health status
            status = self._calculate_health_status(model_key)

            # Cache result
            self.health_status_cache[model_key] = (status, time.time())

            return status

    def _calculate_health_status(self, model_key: str) -> HealthStatus:
        """
        Calculate health status based on metrics.

        Args:
            model_key: Model key

        Returns:
            Health status
        """
        if model_key not in self.metrics_history:
            return HealthStatus.UNKNOWN

        history = self.metrics_history[model_key]
        if not history:
            return HealthStatus.UNKNOWN

        # Get recent metrics (last 5 samples)
        recent_metrics = list(history)[-5:]

        # Check error rates
        error_rates = [m.error_rate for m in recent_metrics]
        if error_rates:
            avg_error_rate = statistics.mean(error_rates)
            if avg_error_rate > self.thresholds["error_rate_critical"]:
                return HealthStatus.CRITICAL
            elif avg_error_rate > self.thresholds["error_rate_warning"]:
                return HealthStatus.UNHEALTHY

        # Check latency
        latencies = [m.average_latency_ms for m in recent_metrics if m.average_latency_ms > 0]
        if latencies:
            avg_latency = statistics.mean(latencies)
            if avg_latency > self.thresholds["latency_p95_critical"]:
                return HealthStatus.CRITICAL
            elif avg_latency > self.thresholds["latency_p95_warning"]:
                return HealthStatus.UNHEALTHY

        # Check resource usage
        memory_usages = [m.memory_usage_mb for m in recent_metrics if m.memory_usage_mb > 0]
        if memory_usages:
            # This would need max memory info - simplified check
            pass

        return HealthStatus.HEALTHY

    def get_model_metrics(self, model_key: str, include_history: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get metrics for a model.

        Args:
            model_key: Model key
            include_history: Whether to include full metrics history

        Returns:
            Metrics dictionary or None if not found
        """
        with self._lock:
            if model_key not in self.current_metrics:
                return None

            current = self.current_metrics[model_key]
            result = {
                "model_id": current.model_id,
                "version": current.version,
                "current_metrics": {
                    "request_count": current.request_count,
                    "error_count": current.error_count,
                    "error_rate": current.error_rate,
                    "average_latency_ms": current.average_latency_ms,
                    "max_latency_ms": current.max_latency_ms,
                    "min_latency_ms": current.min_latency_ms,
                    "memory_usage_mb": current.memory_usage_mb,
                    "cpu_usage_percent": current.cpu_usage_percent,
                    "gpu_memory_mb": current.gpu_memory_mb,
                    "custom_metrics": current.custom_metrics
                },
                "health_status": self.get_health_status(model_key).value,
                "history_size": len(self.metrics_history.get(model_key, []))
            }

            if include_history:
                history = self.metrics_history.get(model_key, [])
                result["metrics_history"] = [
                    {
                        "timestamp": m.timestamp,
                        "request_count": m.request_count,
                        "error_rate": m.error_rate,
                        "average_latency_ms": m.average_latency_ms,
                        "memory_usage_mb": m.memory_usage_mb
                    }
                    for m in history
                ]

            return result

    def get_monitor_stats(self) -> Dict[str, Any]:
        """
        Get monitoring statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            total_requests = sum(m.request_count for m in self.current_metrics.values())
            total_errors = sum(m.error_count for m in self.current_metrics.values())

            return {
                "monitored_models": len(self.current_metrics),
                "total_requests": total_requests,
                "total_errors": total_errors,
                "overall_error_rate": total_errors / max(total_requests, 1),
                "alert_stats": self.alert_system.get_alert_stats(),
                "monitoring_interval": self.monitoring_interval,
                "metrics_history_size": self.metrics_history_size
            }

    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """
        Update health thresholds.

        Args:
            new_thresholds: New threshold values
        """
        with self._lock:
            self.thresholds.update(new_thresholds)
            logger.info(f"Updated health thresholds: {new_thresholds}")

    def __del__(self):
        """Cleanup on destruction."""
        self.stop_monitoring()
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
