"""
Alert System for Real-time Model Switching

Provides comprehensive alerting capabilities for model switching operations,
health monitoring, and system issues.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable
from collections import defaultdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import os

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts that can be triggered"""
    MODEL_SWITCH_FAILED = "model_switch_failed"
    MODEL_HEALTH_DEGRADED = "model_health_degraded"
    MODEL_LOAD_FAILED = "model_load_failed"
    CACHE_EVICTION_HIGH = "cache_eviction_high"
    MEMORY_USAGE_HIGH = "memory_usage_high"
    REQUEST_LATENCY_SPIKE = "request_latency_spike"
    TRANSITION_TIMEOUT = "transition_timeout"
    VERSION_ROLLBACK_TRIGGERED = "version_rollback_triggered"
    STRATEGY_EXECUTION_FAILED = "strategy_execution_failed"
    CONFIGURATION_ERROR = "configuration_error"


@dataclass
class Alert:
    """Represents an alert instance"""
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    model_name: Optional[str] = None
    version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    auto_resolve: bool = False
    resolve_after_minutes: Optional[int] = None


class AlertHandler:
    """Base class for alert handlers"""

    async def send_alert(self, alert: Alert) -> bool:
        """Send an alert. Return True if successful."""
        raise NotImplementedError


class ConsoleAlertHandler(AlertHandler):
    """Logs alerts to console"""

    def __init__(self, log_level: int = logging.WARNING):
        self.log_level = log_level

    async def send_alert(self, alert: Alert) -> bool:
        try:
            log_message = f"[{alert.severity.value.upper()}] {alert.title}: {alert.message}"
            if alert.model_name:
                log_message += f" (Model: {alert.model_name})"
            if alert.version:
                log_message += f" (Version: {alert.version})"

            logger.log(self.log_level, log_message)
            return True
        except Exception as e:
            logger.error(f"Failed to send console alert: {e}")
            return False


class EmailAlertHandler(AlertHandler):
    """Sends alerts via email"""

    def __init__(self,
                 smtp_server: str,
                 smtp_port: int,
                 sender_email: str,
                 sender_password: str,
                 recipient_emails: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_emails = recipient_emails

    async def send_alert(self, alert: Alert) -> bool:
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipient_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"

            body = f"""
Model Switching Alert

Type: {alert.type.value}
Severity: {alert.severity.value}
Time: {alert.timestamp.isoformat()}

{alert.message}

{f"Model: {alert.model_name}" if alert.model_name else ""}
{f"Version: {alert.version}" if alert.version else ""}

Metadata:
{json.dumps(alert.metadata, indent=2, default=str)}
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, self.recipient_emails, text)
            server.quit()

            return True
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class WebhookAlertHandler(AlertHandler):
    """Sends alerts via HTTP webhook"""

    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}

    async def send_alert(self, alert: Alert) -> bool:
        try:
            payload = {
                "id": alert.id,
                "type": alert.type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "model_name": alert.model_name,
                "version": alert.version,
                "metadata": alert.metadata,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class SlackAlertHandler(AlertHandler):
    """Sends alerts to Slack via webhook"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send_alert(self, alert: Alert) -> bool:
        try:
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger"
            }

            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "danger"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Type", "value": alert.type.value, "short": True},
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Time", "value": alert.timestamp.isoformat(), "short": True}
                    ]
                }]
            }

            if alert.model_name:
                payload["attachments"][0]["fields"].append({
                    "title": "Model", "value": alert.model_name, "short": True
                })
            if alert.version:
                payload["attachments"][0]["fields"].append({
                    "title": "Version", "value": alert.version, "short": True
                })

            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class AlertManager:
    """Manages alerts and alert handlers"""

    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.handlers: List[AlertHandler] = []
        self.alert_counts: Dict[AlertType, int] = defaultdict(int)
        self.cooldowns: Dict[str, datetime] = {}  # alert_key -> last_sent_time
        self._cleanup_task: Optional[asyncio.Task] = None

    def add_handler(self, handler: AlertHandler):
        """Add an alert handler"""
        self.handlers.append(handler)

    def remove_handler(self, handler: AlertHandler):
        """Remove an alert handler"""
        self.handlers.remove(handler)

    async def raise_alert(self,
                         alert_type: AlertType,
                         severity: AlertSeverity,
                         title: str,
                         message: str,
                         model_name: Optional[str] = None,
                         version: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None,
                         auto_resolve: bool = False,
                         resolve_after_minutes: Optional[int] = None,
                         cooldown_minutes: int = 5) -> str:
        """
        Raise a new alert

        Args:
            alert_type: Type of alert
            severity: Alert severity
            title: Alert title
            message: Alert message
            model_name: Associated model name
            version: Associated version
            metadata: Additional metadata
            auto_resolve: Whether to auto-resolve after timeout
            resolve_after_minutes: Minutes to wait before auto-resolving
            cooldown_minutes: Minutes to wait before sending duplicate alerts

        Returns:
            Alert ID
        """
        # Create alert key for deduplication
        alert_key = f"{alert_type.value}_{model_name or 'none'}_{version or 'none'}"

        # Check cooldown
        now = datetime.now()
        if alert_key in self.cooldowns:
            last_sent = self.cooldowns[alert_key]
            if (now - last_sent).total_seconds() < cooldown_minutes * 60:
                logger.debug(f"Alert {alert_key} is in cooldown, skipping")
                return ""

        # Create alert
        alert_id = f"{alert_type.value}_{now.timestamp()}"
        alert = Alert(
            id=alert_id,
            type=alert_type,
            severity=severity,
            title=title,
            message=message,
            model_name=model_name,
            version=version,
            metadata=metadata or {},
            auto_resolve=auto_resolve,
            resolve_after_minutes=resolve_after_minutes
        )

        self.alerts[alert_id] = alert
        self.alert_counts[alert_type] += 1
        self.cooldowns[alert_key] = now

        # Send alert via all handlers
        success_count = 0
        for handler in self.handlers:
            try:
                if await handler.send_alert(alert):
                    success_count += 1
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        if success_count == 0:
            logger.warning(f"No alert handlers succeeded for alert {alert_id}")

        # Schedule auto-resolution if requested
        if auto_resolve and resolve_after_minutes:
            asyncio.create_task(self._auto_resolve_alert(alert_id, resolve_after_minutes))

        logger.info(f"Alert raised: {alert_type.value} ({severity.value}) - {title}")
        return alert_id

    async def resolve_alert(self, alert_id: str, resolution_message: Optional[str] = None):
        """Resolve an alert"""
        if alert_id not in self.alerts:
            logger.warning(f"Alert {alert_id} not found")
            return

        alert = self.alerts[alert_id]
        if alert.resolved:
            return

        alert.resolved = True
        alert.resolved_at = datetime.now()

        if resolution_message:
            alert.metadata["resolution"] = resolution_message

        logger.info(f"Alert resolved: {alert.type.value} - {alert.title}")

    async def _auto_resolve_alert(self, alert_id: str, delay_minutes: int):
        """Auto-resolve an alert after delay"""
        await asyncio.sleep(delay_minutes * 60)

        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            await self.resolve_alert(alert_id, f"Auto-resolved after {delay_minutes} minutes")

    def get_active_alerts(self,
                         alert_type: Optional[AlertType] = None,
                         model_name: Optional[str] = None) -> List[Alert]:
        """Get active (unresolved) alerts"""
        alerts = [a for a in self.alerts.values() if not a.resolved]

        if alert_type:
            alerts = [a for a in alerts if a.type == alert_type]
        if model_name:
            alerts = [a for a in alerts if a.model_name == model_name]

        return alerts

    def get_alert_history(self,
                         hours: int = 24,
                         alert_type: Optional[AlertType] = None) -> List[Alert]:
        """Get alert history for the specified time period"""
        cutoff = datetime.now() - timedelta(hours=hours)
        alerts = [a for a in self.alerts.values() if a.timestamp >= cutoff]

        if alert_type:
            alerts = [a for a in alerts if a.type == alert_type]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        total_alerts = len(self.alerts)
        active_alerts = len([a for a in self.alerts.values() if not a.resolved])
        resolved_alerts = total_alerts - active_alerts

        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)

        for alert in self.alerts.values():
            severity_counts[alert.severity.value] += 1
            type_counts[alert.type.value] += 1

        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "resolved_alerts": resolved_alerts,
            "severity_breakdown": dict(severity_counts),
            "type_breakdown": dict(type_counts)
        }

    async def start_cleanup_task(self, interval_minutes: int = 60):
        """Start periodic cleanup of old alerts"""
        if self._cleanup_task and not self._cleanup_task.done():
            return

        async def cleanup():
            while True:
                await asyncio.sleep(interval_minutes * 60)
                await self._cleanup_old_alerts()

        self._cleanup_task = asyncio.create_task(cleanup())

    async def _cleanup_old_alerts(self, retention_days: int = 30):
        """Clean up old resolved alerts"""
        cutoff = datetime.now() - timedelta(days=retention_days)
        to_remove = []

        for alert_id, alert in self.alerts.items():
            if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff:
                to_remove.append(alert_id)

        for alert_id in to_remove:
            del self.alerts[alert_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old alerts")


# Global alert manager instance
alert_manager = AlertManager()


def setup_default_alert_handlers():
    """Setup default alert handlers based on environment configuration"""

    # Console handler (always enabled)
    alert_manager.add_handler(ConsoleAlertHandler())

    # Email handler
    smtp_server = os.getenv("ALERT_SMTP_SERVER")
    if smtp_server:
        try:
            email_handler = EmailAlertHandler(
                smtp_server=smtp_server,
                smtp_port=int(os.getenv("ALERT_SMTP_PORT", "587")),
                sender_email=os.getenv("ALERT_SENDER_EMAIL", ""),
                sender_password=os.getenv("ALERT_SENDER_PASSWORD", ""),
                recipient_emails=os.getenv("ALERT_RECIPIENT_EMAILS", "").split(",")
            )
            alert_manager.add_handler(email_handler)
            logger.info("Email alert handler configured")
        except Exception as e:
            logger.error(f"Failed to configure email alert handler: {e}")

    # Slack handler
    slack_webhook = os.getenv("ALERT_SLACK_WEBHOOK")
    if slack_webhook:
        try:
            slack_handler = SlackAlertHandler(slack_webhook)
            alert_manager.add_handler(slack_handler)
            logger.info("Slack alert handler configured")
        except Exception as e:
            logger.error(f"Failed to configure Slack alert handler: {e}")

    # Webhook handler
    webhook_url = os.getenv("ALERT_WEBHOOK_URL")
    if webhook_url:
        try:
            webhook_handler = WebhookAlertHandler(webhook_url)
            alert_manager.add_handler(webhook_handler)
            logger.info("Webhook alert handler configured")
        except Exception as e:
            logger.error(f"Failed to configure webhook alert handler: {e}")


# Initialize default handlers on import
setup_default_alert_handlers()


async def alert_model_switch_failed(model_name: str,
                                   version: str,
                                   error: str,
                                   metadata: Optional[Dict[str, Any]] = None):
    """Alert for model switch failure"""
    return await alert_manager.raise_alert(
        AlertType.MODEL_SWITCH_FAILED,
        AlertSeverity.ERROR,
        f"Model Switch Failed: {model_name}",
        f"Failed to switch {model_name} to version {version}: {error}",
        model_name=model_name,
        version=version,
        metadata=metadata
    )


async def alert_model_health_degraded(model_name: str,
                                     version: str,
                                     metric: str,
                                     current_value: float,
                                     threshold: float,
                                     metadata: Optional[Dict[str, Any]] = None):
    """Alert for degraded model health"""
    return await alert_manager.raise_alert(
        AlertType.MODEL_HEALTH_DEGRADED,
        AlertSeverity.WARNING,
        f"Model Health Degraded: {model_name}",
        f"{model_name} v{version} {metric} degraded: {current_value:.2f} (threshold: {threshold:.2f})",
        model_name=model_name,
        version=version,
        metadata=metadata,
        auto_resolve=True,
        resolve_after_minutes=30
    )


async def alert_model_load_failed(model_name: str,
                                 version: str,
                                 error: str,
                                 metadata: Optional[Dict[str, Any]] = None):
    """Alert for model load failure"""
    return await alert_manager.raise_alert(
        AlertType.MODEL_LOAD_FAILED,
        AlertSeverity.CRITICAL,
        f"Model Load Failed: {model_name}",
        f"Failed to load {model_name} version {version}: {error}",
        model_name=model_name,
        version=version,
        metadata=metadata
    )


async def alert_cache_eviction_high(model_name: str,
                                   eviction_rate: float,
                                   threshold: float,
                                   metadata: Optional[Dict[str, Any]] = None):
    """Alert for high cache eviction rate"""
    return await alert_manager.raise_alert(
        AlertType.CACHE_EVICTION_HIGH,
        AlertSeverity.WARNING,
        f"High Cache Eviction: {model_name}",
        f"Cache eviction rate for {model_name} is {eviction_rate:.2f}% (threshold: {threshold:.2f}%)",
        model_name=model_name,
        metadata=metadata,
        auto_resolve=True,
        resolve_after_minutes=15
    )


async def alert_memory_usage_high(usage_percent: float,
                                 threshold: float,
                                 metadata: Optional[Dict[str, Any]] = None):
    """Alert for high memory usage"""
    return await alert_manager.raise_alert(
        AlertType.MEMORY_USAGE_HIGH,
        AlertSeverity.WARNING,
        "High Memory Usage",
        f"Memory usage is {usage_percent:.1f}% (threshold: {threshold:.1f}%)",
        metadata=metadata,
        auto_resolve=True,
        resolve_after_minutes=10
    )


async def alert_request_latency_spike(model_name: str,
                                     latency_ms: float,
                                     baseline_ms: float,
                                     metadata: Optional[Dict[str, Any]] = None):
    """Alert for request latency spike"""
    return await alert_manager.raise_alert(
        AlertType.REQUEST_LATENCY_SPIKE,
        AlertSeverity.WARNING,
        f"Latency Spike: {model_name}",
        f"Request latency for {model_name} spiked to {latency_ms:.0f}ms (baseline: {baseline_ms:.0f}ms)",
        model_name=model_name,
        metadata=metadata,
        auto_resolve=True,
        resolve_after_minutes=20
    )


async def alert_transition_timeout(model_name: str,
                                  transition_type: str,
                                  timeout_seconds: int,
                                  metadata: Optional[Dict[str, Any]] = None):
    """Alert for transition timeout"""
    return await alert_manager.raise_alert(
        AlertType.TRANSITION_TIMEOUT,
        AlertSeverity.ERROR,
        f"Transition Timeout: {model_name}",
        f"{transition_type} transition for {model_name} timed out after {timeout_seconds}s",
        model_name=model_name,
        metadata=metadata
    )


async def alert_version_rollback_triggered(model_name: str,
                                          from_version: str,
                                          to_version: str,
                                          reason: str,
                                          metadata: Optional[Dict[str, Any]] = None):
    """Alert for automatic version rollback"""
    return await alert_manager.raise_alert(
        AlertType.VERSION_ROLLBACK_TRIGGERED,
        AlertSeverity.CRITICAL,
        f"Version Rollback: {model_name}",
        f"Automatically rolled back {model_name} from {from_version} to {to_version}: {reason}",
        model_name=model_name,
        version=to_version,
        metadata=metadata
    )


async def alert_strategy_execution_failed(model_name: str,
                                         strategy_type: str,
                                         error: str,
                                         metadata: Optional[Dict[str, Any]] = None):
    """Alert for switching strategy execution failure"""
    return await alert_manager.raise_alert(
        AlertType.STRATEGY_EXECUTION_FAILED,
        AlertSeverity.ERROR,
        f"Strategy Execution Failed: {model_name}",
        f"Failed to execute {strategy_type} strategy for {model_name}: {error}",
        model_name=model_name,
        metadata=metadata
    )


async def alert_configuration_error(error: str,
                                   metadata: Optional[Dict[str, Any]] = None):
    """Alert for configuration errors"""
    return await alert_manager.raise_alert(
        AlertType.CONFIGURATION_ERROR,
        AlertSeverity.CRITICAL,
        "Configuration Error",
        f"Model switching configuration error: {error}",
        metadata=metadata
    )