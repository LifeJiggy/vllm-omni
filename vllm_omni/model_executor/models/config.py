"""
Configuration System for Real-time Model Switching

This module provides configuration classes and utilities for the model switching system.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm.config import config


@config
@dataclass
class ModelSwitchingConfig:
    """Configuration for the real-time model switching system."""

    # Registry Configuration
    enable_dynamic_registry: bool = True
    max_cached_models: int = 5
    model_ttl_seconds: int = 3600  # 1 hour
    cleanup_interval_seconds: int = 300  # 5 minutes

    # Cache Configuration
    enable_model_cache: bool = True
    max_cache_size: int = 5
    cache_memory_limit_gb: float = 8.0
    cache_eviction_interval_seconds: float = 60.0

    # Switching Configuration
    max_concurrent_switches: int = 3
    default_switch_strategy: str = "immediate"
    enable_health_checks: bool = True
    switch_timeout_seconds: int = 300  # 5 minutes

    # Transition Configuration
    enable_transitions: bool = True
    max_transition_time_seconds: int = 300
    request_timeout_seconds: float = 60.0
    max_transition_history: int = 100

    # Health Monitoring Configuration
    enable_health_monitoring: bool = True
    monitoring_interval_seconds: float = 30.0
    metrics_history_size: int = 100
    health_check_timeout_seconds: float = 10.0

    # Health Thresholds
    error_rate_warning_threshold: float = 0.05  # 5%
    error_rate_critical_threshold: float = 0.10  # 10%
    latency_p95_warning_threshold: float = 5000  # 5 seconds
    latency_p95_critical_threshold: float = 10000  # 10 seconds
    memory_usage_warning_threshold: float = 0.8  # 80%
    memory_usage_critical_threshold: float = 0.95  # 95%

    # Version Management Configuration
    enable_version_management: bool = True
    version_storage_path: str = "./model_versions"
    max_version_history: int = 50

    # Alert Configuration
    enable_alerts: bool = True
    alert_history_size: int = 1000

    # API Configuration
    enable_management_api: bool = True
    api_host: str = "localhost"
    api_port: int = 8001
    api_base_path: str = "/api/v1/model-switching"

    # Security Configuration
    enable_authentication: bool = False
    api_key: str | None = None
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])

    # Logging Configuration
    log_level: str = "INFO"
    enable_audit_logging: bool = True
    audit_log_path: str | None = None

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate thresholds
        if not (0 <= self.error_rate_warning_threshold <= 1):
            raise ValueError("error_rate_warning_threshold must be between 0 and 1")
        if not (0 <= self.error_rate_critical_threshold <= 1):
            raise ValueError("error_rate_critical_threshold must be between 0 and 1")
        if self.error_rate_warning_threshold >= self.error_rate_critical_threshold:
            raise ValueError("error_rate_warning_threshold must be less than error_rate_critical_threshold")

        if self.latency_p95_warning_threshold >= self.latency_p95_critical_threshold:
            raise ValueError("latency_p95_warning_threshold must be less than latency_p95_critical_threshold")

        if not (0 <= self.memory_usage_warning_threshold <= 1):
            raise ValueError("memory_usage_warning_threshold must be between 0 and 1")
        if not (0 <= self.memory_usage_critical_threshold <= 1):
            raise ValueError("memory_usage_critical_threshold must be between 0 and 1")
        if self.memory_usage_warning_threshold >= self.memory_usage_critical_threshold:
            raise ValueError("memory_usage_warning_threshold must be less than memory_usage_critical_threshold")

        # Create directories if needed
        if self.version_storage_path:
            Path(self.version_storage_path).mkdir(parents=True, exist_ok=True)

        if self.audit_log_path:
            Path(self.audit_log_path).parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "ModelSwitchingConfig":
        """Create configuration from environment variables."""
        return cls(
            # Registry settings
            enable_dynamic_registry=cls._env_bool("VLLM_MODEL_SWITCHING_ENABLE_REGISTRY", True),
            max_cached_models=cls._env_int("VLLM_MODEL_SWITCHING_MAX_CACHED_MODELS", 5),
            model_ttl_seconds=cls._env_int("VLLM_MODEL_SWITCHING_MODEL_TTL_SECONDS", 3600),
            cleanup_interval_seconds=cls._env_int("VLLM_MODEL_SWITCHING_CLEANUP_INTERVAL", 300),
            # Cache settings
            enable_model_cache=cls._env_bool("VLLM_MODEL_SWITCHING_ENABLE_CACHE", True),
            max_cache_size=cls._env_int("VLLM_MODEL_SWITCHING_MAX_CACHE_SIZE", 5),
            cache_memory_limit_gb=cls._env_float("VLLM_MODEL_SWITCHING_CACHE_MEMORY_GB", 8.0),
            cache_eviction_interval_seconds=cls._env_float("VLLM_MODEL_SWITCHING_CACHE_EVICTION_INTERVAL", 60.0),
            # Switching settings
            max_concurrent_switches=cls._env_int("VLLM_MODEL_SWITCHING_MAX_CONCURRENT", 3),
            default_switch_strategy=cls._env_str("VLLM_MODEL_SWITCHING_DEFAULT_STRATEGY", "immediate"),
            enable_health_checks=cls._env_bool("VLLM_MODEL_SWITCHING_ENABLE_HEALTH_CHECKS", True),
            switch_timeout_seconds=cls._env_int("VLLM_MODEL_SWITCHING_TIMEOUT", 300),
            # Transition settings
            enable_transitions=cls._env_bool("VLLM_MODEL_SWITCHING_ENABLE_TRANSITIONS", True),
            max_transition_time_seconds=cls._env_int("VLLM_MODEL_SWITCHING_MAX_TRANSITION_TIME", 300),
            request_timeout_seconds=cls._env_float("VLLM_MODEL_SWITCHING_REQUEST_TIMEOUT", 60.0),
            max_transition_history=cls._env_int("VLLM_MODEL_SWITCHING_MAX_TRANSITION_HISTORY", 100),
            # Health monitoring settings
            enable_health_monitoring=cls._env_bool("VLLM_MODEL_SWITCHING_ENABLE_MONITORING", True),
            monitoring_interval_seconds=cls._env_float("VLLM_MODEL_SWITCHING_MONITORING_INTERVAL", 30.0),
            metrics_history_size=cls._env_int("VLLM_MODEL_SWITCHING_METRICS_HISTORY_SIZE", 100),
            health_check_timeout_seconds=cls._env_float("VLLM_MODEL_SWITCHING_HEALTH_CHECK_TIMEOUT", 10.0),
            # Health thresholds
            error_rate_warning_threshold=cls._env_float("VLLM_MODEL_SWITCHING_ERROR_RATE_WARNING", 0.05),
            error_rate_critical_threshold=cls._env_float("VLLM_MODEL_SWITCHING_ERROR_RATE_CRITICAL", 0.10),
            latency_p95_warning_threshold=cls._env_float("VLLM_MODEL_SWITCHING_LATENCY_WARNING", 5000),
            latency_p95_critical_threshold=cls._env_float("VLLM_MODEL_SWITCHING_LATENCY_CRITICAL", 10000),
            memory_usage_warning_threshold=cls._env_float("VLLM_MODEL_SWITCHING_MEMORY_WARNING", 0.8),
            memory_usage_critical_threshold=cls._env_float("VLLM_MODEL_SWITCHING_MEMORY_CRITICAL", 0.95),
            # Version management settings
            enable_version_management=cls._env_bool("VLLM_MODEL_SWITCHING_ENABLE_VERSION_MGMT", True),
            version_storage_path=cls._env_str("VLLM_MODEL_SWITCHING_VERSION_STORAGE_PATH", "./model_versions"),
            max_version_history=cls._env_int("VLLM_MODEL_SWITCHING_MAX_VERSION_HISTORY", 50),
            # Alert settings
            enable_alerts=cls._env_bool("VLLM_MODEL_SWITCHING_ENABLE_ALERTS", True),
            alert_history_size=cls._env_int("VLLM_MODEL_SWITCHING_ALERT_HISTORY_SIZE", 1000),
            # API settings
            enable_management_api=cls._env_bool("VLLM_MODEL_SWITCHING_ENABLE_API", True),
            api_host=cls._env_str("VLLM_MODEL_SWITCHING_API_HOST", "localhost"),
            api_port=cls._env_int("VLLM_MODEL_SWITCHING_API_PORT", 8001),
            api_base_path=cls._env_str("VLLM_MODEL_SWITCHING_API_BASE_PATH", "/api/v1/model-switching"),
            # Security settings
            enable_authentication=cls._env_bool("VLLM_MODEL_SWITCHING_ENABLE_AUTH", False),
            api_key=cls._env_str("VLLM_MODEL_SWITCHING_API_KEY", None),
            # Logging settings
            log_level=cls._env_str("VLLM_MODEL_SWITCHING_LOG_LEVEL", "INFO"),
            enable_audit_logging=cls._env_bool("VLLM_MODEL_SWITCHING_ENABLE_AUDIT_LOG", True),
            audit_log_path=cls._env_str("VLLM_MODEL_SWITCHING_AUDIT_LOG_PATH", None),
        )

    @staticmethod
    def _env_bool(key: str, default: bool) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    @staticmethod
    def _env_int(key: str, default: int) -> int:
        """Get integer value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @staticmethod
    def _env_float(key: str, default: float) -> float:
        """Get float value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    @staticmethod
    def _env_str(key: str, default: str | None) -> str | None:
        """Get string value from environment variable."""
        return os.getenv(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "enable_dynamic_registry": self.enable_dynamic_registry,
            "max_cached_models": self.max_cached_models,
            "model_ttl_seconds": self.model_ttl_seconds,
            "cleanup_interval_seconds": self.cleanup_interval_seconds,
            "enable_model_cache": self.enable_model_cache,
            "max_cache_size": self.max_cache_size,
            "cache_memory_limit_gb": self.cache_memory_limit_gb,
            "cache_eviction_interval_seconds": self.cache_eviction_interval_seconds,
            "max_concurrent_switches": self.max_concurrent_switches,
            "default_switch_strategy": self.default_switch_strategy,
            "enable_health_checks": self.enable_health_checks,
            "switch_timeout_seconds": self.switch_timeout_seconds,
            "enable_transitions": self.enable_transitions,
            "max_transition_time_seconds": self.max_transition_time_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_transition_history": self.max_transition_history,
            "enable_health_monitoring": self.enable_health_monitoring,
            "monitoring_interval_seconds": self.monitoring_interval_seconds,
            "metrics_history_size": self.metrics_history_size,
            "health_check_timeout_seconds": self.health_check_timeout_seconds,
            "error_rate_warning_threshold": self.error_rate_warning_threshold,
            "error_rate_critical_threshold": self.error_rate_critical_threshold,
            "latency_p95_warning_threshold": self.latency_p95_warning_threshold,
            "latency_p95_critical_threshold": self.latency_p95_critical_threshold,
            "memory_usage_warning_threshold": self.memory_usage_warning_threshold,
            "memory_usage_critical_threshold": self.memory_usage_critical_threshold,
            "enable_version_management": self.enable_version_management,
            "version_storage_path": self.version_storage_path,
            "max_version_history": self.max_version_history,
            "enable_alerts": self.enable_alerts,
            "alert_history_size": self.alert_history_size,
            "enable_management_api": self.enable_management_api,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "api_base_path": self.api_base_path,
            "enable_authentication": self.enable_authentication,
            "log_level": self.log_level,
            "enable_audit_logging": self.enable_audit_logging,
            "audit_log_path": self.audit_log_path,
        }


@dataclass
class SwitchingStrategyConfig:
    """Configuration for a specific switching strategy."""

    strategy_type: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_immediate(cls) -> "SwitchingStrategyConfig":
        """Create immediate switching configuration."""
        return cls(strategy_type="immediate")

    @classmethod
    def create_gradual(cls, duration_minutes: int = 10, steps: int = 10) -> "SwitchingStrategyConfig":
        """Create gradual rollout configuration."""
        return cls(strategy_type="gradual", parameters={"duration_minutes": duration_minutes, "steps": steps})

    @classmethod
    def create_ab_test(cls, traffic_percentage: float = 50, test_duration_hours: int = 24) -> "SwitchingStrategyConfig":
        """Create A/B test configuration."""
        return cls(
            strategy_type="ab_test",
            parameters={"traffic_percentage": traffic_percentage, "test_duration_hours": test_duration_hours},
        )

    @classmethod
    def create_canary(
        cls, initial_percentage: float = 5, step_percentage: float = 10, evaluation_period_minutes: int = 15
    ) -> "SwitchingStrategyConfig":
        """Create canary deployment configuration."""
        return cls(
            strategy_type="canary",
            parameters={
                "initial_percentage": initial_percentage,
                "step_percentage": step_percentage,
                "evaluation_period_minutes": evaluation_period_minutes,
            },
        )


@dataclass
class ModelRegistrationConfig:
    """Configuration for model registration."""

    model_id: str
    model_config: dict[str, Any]
    version: str
    metadata: dict[str, Any] = field(default_factory=dict)
    preload: bool = False
    priority: int = 0


# Global configuration instance
_model_switching_config: ModelSwitchingConfig | None = None


def get_model_switching_config() -> ModelSwitchingConfig:
    """Get the global model switching configuration."""
    global _model_switching_config
    if _model_switching_config is None:
        _model_switching_config = ModelSwitchingConfig.from_env()
    return _model_switching_config


def set_model_switching_config(config: ModelSwitchingConfig):
    """Set the global model switching configuration."""
    global _model_switching_config
    _model_switching_config = config


def reset_model_switching_config():
    """Reset the global configuration to default."""
    global _model_switching_config
    _model_switching_config = None
