"""
API Endpoints for Real-time Model Switching

This module provides REST API endpoints for managing model switching operations,
including model registration, switching, monitoring, and configuration.
"""

import asyncio
from datetime import datetime
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from vllm.logger import init_logger

from vllm_omni.model_executor.models.config import ModelSwitchingConfig, get_model_switching_config
from vllm_omni.model_executor.models.dynamic_registry import DynamicModelRegistry
from vllm_omni.model_executor.models.health_monitor import HealthMonitor
from vllm_omni.model_executor.models.model_switcher import ModelSwitcher
from vllm_omni.model_executor.models.switching_strategies import SwitchingStrategyType

logger = init_logger(__name__)


# Pydantic models for API requests/responses
class ModelRegistrationRequest(BaseModel):
    """Request model for registering a new model."""

    model: str = Field(..., description="Model path or identifier")
    model_arch: str = Field(..., description="Model architecture")
    model_stage: str = Field(..., description="Model stage (thinker, talker, etc.)")
    version: str = Field(..., description="Model version")
    metadata: dict[str, Any] | None = Field(default_factory=dict, description="Additional metadata")


class SwitchRequest(BaseModel):
    """Request model for switching models."""

    model_id: str = Field(..., description="Model identifier")
    target_version: str = Field(..., description="Target version to switch to")
    strategy: SwitchingStrategyType = Field(default=SwitchingStrategyType.IMMEDIATE, description="Switching strategy")
    strategy_config: dict[str, Any] | None = Field(default_factory=dict, description="Strategy-specific configuration")
    metadata: dict[str, Any] | None = Field(default_factory=dict, description="Additional metadata")


class ModelInfo(BaseModel):
    """Model information response."""

    model_id: str
    versions: list[str]
    active_version: str | None
    health_status: str
    metadata: dict[str, Any]


class SwitchOperationResponse(BaseModel):
    """Response model for switch operations."""

    operation_id: str
    success: bool
    model_id: str
    from_version: str
    to_version: str
    strategy: str
    status: str
    progress: float
    duration_seconds: float | None
    error_message: str | None
    metadata: dict[str, Any]


class HealthMetricsResponse(BaseModel):
    """Response model for health metrics."""

    model_id: str
    version: str
    health_status: str
    current_metrics: dict[str, Any]
    history_size: int
    last_updated: float | None


class AlertResponse(BaseModel):
    """Response model for alerts."""

    alert_id: str
    model_id: str
    version: str
    severity: str
    title: str
    message: str
    timestamp: float
    resolved: bool


class ConfigUpdateRequest(BaseModel):
    """Request model for configuration updates."""

    config: dict[str, Any] = Field(..., description="Configuration updates")


class APIStats(BaseModel):
    """API statistics response."""

    total_requests: int
    active_operations: int
    registered_models: int
    alerts_count: int
    uptime_seconds: float


class ModelSwitchingAPI:
    """
    REST API for model switching management.

    This class provides HTTP endpoints for:
    - Model registration and management
    - Switching operations
    - Health monitoring
    - Configuration management
    - Statistics and monitoring
    """

    def __init__(
        self,
        registry: DynamicModelRegistry,
        switcher: ModelSwitcher,
        health_monitor: HealthMonitor,
        config: ModelSwitchingConfig | None = None,
    ):
        """
        Initialize the API.

        Args:
            registry: Model registry instance
            switcher: Model switcher instance
            health_monitor: Health monitor instance
            config: Configuration instance
        """
        self.registry = registry
        self.switchier = switcher
        self.health_monitor = health_monitor
        self.config = config or get_model_switching_config()

        # API statistics
        self.start_time = asyncio.get_event_loop().time()
        self.request_count = 0

        # Create FastAPI app
        self.app = FastAPI(
            title="vLLM-Omni Model Switching API",
            description="API for managing real-time model switching operations",
            version="1.0.0",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add authentication if enabled
        self.security = HTTPBearer() if self.config.enable_authentication else None

        # Register routes
        self._register_routes()

        logger.info(f"Initialized Model Switching API on {self.config.api_host}:{self.config.api_port}")

    def _register_routes(self):
        """Register API routes."""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        @self.app.get("/stats", response_model=APIStats)
        async def get_stats():
            """Get API statistics."""
            self.request_count += 1
            current_time = asyncio.get_event_loop().time()

            return APIStats(
                total_requests=self.request_count,
                active_operations=len(self.switchier.active_operations),
                registered_models=len(self.registry.model_versions),
                alerts_count=len(self.health_monitor.alert_system.active_alerts),
                uptime_seconds=current_time - self.start_time,
            )

        # Model management endpoints
        @self.app.post("/models", status_code=201)
        async def register_model(request: ModelRegistrationRequest):
            """Register a new model version."""
            self.request_count += 1

            try:
                from vllm_omni.config.model import OmniModelConfig

                config = OmniModelConfig(
                    model=request.model, model_arch=request.model_arch, model_stage=request.model_stage
                )

                model_id = self.registry.register_model(config, request.version, request.metadata)

                logger.info(f"Registered model {model_id} version {request.version}")
                return {"model_id": model_id, "version": request.version, "status": "registered"}

            except Exception as e:
                logger.error(f"Failed to register model: {e}")
                raise HTTPException(status_code=400, detail=f"Registration failed: {e}")

        @self.app.get("/models", response_model=list[ModelInfo])
        async def list_models():
            """List all registered models."""
            self.request_count += 1

            models = []
            for model_id, versions in self.registry.model_versions.items():
                active_instance = self.registry.get_active_model(model_id)
                active_version = active_instance.version if active_instance else None

                # Get health status
                health_status = "unknown"
                if active_version:
                    model_key = f"{model_id}_{active_version}"
                    health_status = self.health_monitor.get_health_status(model_key).value

                models.append(
                    ModelInfo(
                        model_id=model_id,
                        versions=[v.version for v in versions],
                        active_version=active_version,
                        health_status=health_status,
                        metadata=versions[0].metadata if versions else {},
                    )
                )

            return models

        @self.app.get("/models/{model_id}")
        async def get_model(model_id: str):
            """Get details for a specific model."""
            self.request_count += 1

            if model_id not in self.registry.model_versions:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

            versions = self.registry.model_versions[model_id]
            active_instance = self.registry.get_active_model(model_id)

            return {
                "model_id": model_id,
                "versions": [{"version": v.version, "status": v.status, "metadata": v.metadata} for v in versions],
                "active_version": active_instance.version if active_instance else None,
                "health_status": self.health_monitor.get_health_status(
                    f"{model_id}_{active_instance.version}" if active_instance else ""
                ).value
                if active_instance
                else "unknown",
            }

        @self.app.delete("/models/{model_id}/versions/{version}")
        async def deregister_model(model_id: str, version: str):
            """Deregister a model version."""
            self.request_count += 1

            success = self.registry.deregister_model(model_id, version)
            if not success:
                raise HTTPException(status_code=404, detail=f"Model {model_id} version {version} not found")

            logger.info(f"Deregistered model {model_id} version {version}")
            return {"status": "deregistered"}

        # Switching endpoints
        @self.app.post("/switch", response_model=SwitchOperationResponse)
        async def switch_model(request: SwitchRequest, background_tasks: BackgroundTasks):
            """Initiate a model switch operation."""
            self.request_count += 1

            try:
                # Validate request
                validation = self.switchier.validate_switch_request(request.model_id, request.target_version)
                if not validation["valid"]:
                    raise HTTPException(status_code=400, detail=f"Invalid request: {validation['errors']}")

                # Start switch operation
                result = await self.switchier.switch_model(
                    model_id=request.model_id,
                    target_version=request.target_version,
                    strategy_type=request.strategy,
                    strategy_config=request.strategy_config,
                    metadata=request.metadata,
                )

                response = SwitchOperationResponse(
                    operation_id=result.operation_id,
                    success=result.success,
                    model_id=result.model_id,
                    from_version=result.from_version,
                    to_version=result.to_version,
                    strategy=result.strategy_type.value,
                    status="completed" if result.success else "failed",
                    progress=1.0 if result.success else 0.0,
                    duration_seconds=result.duration_seconds,
                    error_message=result.error_message,
                    metadata=result.metadata,
                )

                if result.success:
                    logger.info(f"Switch operation {result.operation_id} completed successfully")
                else:
                    logger.error(f"Switch operation {result.operation_id} failed: {result.error_message}")

                return response

            except Exception as e:
                logger.error(f"Switch request failed: {e}")
                raise HTTPException(status_code=500, detail=f"Switch failed: {e}")

        @self.app.get("/switch/operations", response_model=list[dict[str, Any]])
        async def list_switch_operations():
            """List all switch operations."""
            self.request_count += 1

            return self.switchier.list_active_switches()

        @self.app.get("/switch/operations/{operation_id}")
        async def get_switch_operation(operation_id: str):
            """Get details for a specific switch operation."""
            self.request_count += 1

            operation = self.switchier.get_switch_status(operation_id)
            if not operation:
                raise HTTPException(status_code=404, detail=f"Operation {operation_id} not found")

            return operation

        @self.app.post("/switch/operations/{operation_id}/abort")
        async def abort_switch_operation(operation_id: str):
            """Abort a switch operation."""
            self.request_count += 1

            success = self.switchier.abort_switch(operation_id)
            if not success:
                raise HTTPException(status_code=404, detail=f"Operation {operation_id} not found or already completed")

            logger.info(f"Aborted switch operation {operation_id}")
            return {"status": "aborted"}

        # Health monitoring endpoints
        @self.app.get("/health/models", response_model=list[HealthMetricsResponse])
        async def get_models_health():
            """Get health status for all models."""
            self.request_count += 1

            health_data = []
            for model_id in self.registry.model_versions.keys():
                active_instance = self.registry.get_active_model(model_id)
                if active_instance:
                    model_key = f"{model_id}_{active_instance.version}"
                    metrics = self.health_monitor.get_model_metrics(model_key)
                    if metrics:
                        health_data.append(
                            HealthMetricsResponse(
                                model_id=metrics["model_id"],
                                version=metrics["version"],
                                health_status=metrics["health_status"],
                                current_metrics=metrics["current_metrics"],
                                history_size=metrics["history_size"],
                                last_updated=metrics["current_metrics"].get("timestamp"),
                            )
                        )

            return health_data

        @self.app.get("/health/models/{model_id}")
        async def get_model_health(model_id: str):
            """Get health status for a specific model."""
            self.request_count += 1

            active_instance = self.registry.get_active_model(model_id)
            if not active_instance:
                raise HTTPException(status_code=404, detail=f"Active model not found for {model_id}")

            model_key = f"{model_id}_{active_instance.version}"
            metrics = self.health_monitor.get_model_metrics(model_key)
            if not metrics:
                raise HTTPException(status_code=404, detail=f"Health metrics not found for {model_id}")

            return metrics

        @self.app.get("/alerts", response_model=list[AlertResponse])
        async def get_alerts(model_id: str | None = None):
            """Get active alerts."""
            self.request_count += 1

            alerts = self.health_monitor.alert_system.get_active_alerts(model_id)
            return [
                AlertResponse(
                    alert_id=alert.alert_id,
                    model_id=alert.model_id,
                    version=alert.version,
                    severity=alert.severity.value,
                    title=alert.title,
                    message=alert.message,
                    timestamp=alert.timestamp,
                    resolved=alert.resolved,
                )
                for alert in alerts
            ]

        # Configuration endpoints
        @self.app.get("/config")
        async def get_config():
            """Get current configuration."""
            self.request_count += 1

            return self.config.to_dict()

        @self.app.put("/config")
        async def update_config(request: ConfigUpdateRequest):
            """Update configuration (limited to safe parameters)."""
            self.request_count += 1

            # Only allow updating certain safe parameters
            allowed_updates = {"log_level", "enable_audit_logging", "audit_log_path"}

            updates = {}
            for key, value in request.config.items():
                if key in allowed_updates:
                    updates[key] = value
                else:
                    logger.warning(f"Attempted to update restricted config parameter: {key}")

            if updates:
                for key, value in updates.items():
                    setattr(self.config, key, value)
                logger.info(f"Updated configuration: {updates}")
                return {"status": "updated", "updates": updates}
            else:
                return {"status": "no_changes"}

        # Strategy information
        @self.app.get("/strategies")
        async def get_strategies():
            """Get available switching strategies."""
            self.request_count += 1

            return self.switchier.get_available_strategies()

    async def authenticate(self, credentials: HTTPAuthorizationCredentials = Depends(lambda: None)):
        """Authenticate API requests if authentication is enabled."""
        if not self.config.enable_authentication:
            return True

        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        if self.config.api_key and credentials.credentials != self.config.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return True

    def create_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app

    async def start_server(self, host: str | None = None, port: int | None = None):
        """
        Start the API server.

        Args:
            host: Server host (defaults to config)
            port: Server port (defaults to config)
        """
        import uvicorn

        host = host or self.config.api_host
        port = port or self.config.api_port

        logger.info(f"Starting Model Switching API server on {host}:{port}")
        config = uvicorn.Config(self.app, host=host, port=port)
        server = uvicorn.Server(config)
        await server.serve()

    def get_openapi_schema(self) -> dict[str, Any]:
        """Get the OpenAPI schema for the API."""
        return self.app.openapi()


# Factory function for creating API instance
def create_model_switching_api(
    registry: DynamicModelRegistry,
    switcher: ModelSwitcher,
    health_monitor: HealthMonitor,
    config: ModelSwitchingConfig | None = None,
) -> ModelSwitchingAPI:
    """
    Create a ModelSwitchingAPI instance.

    Args:
        registry: Model registry
        switcher: Model switcher
        health_monitor: Health monitor
        config: Configuration (optional)

    Returns:
        Configured API instance
    """
    return ModelSwitchingAPI(registry, switcher, health_monitor, config)
