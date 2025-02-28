"""Base configuration models for the application."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class StorageConfig(BaseModel):
    """Configuration for storage."""

    path: str = Field(default="chain_storage")
    format: str = Field(default="json")

    model_config = ConfigDict(validate_assignment=True)


class VisualizationConfig(BaseModel):
    """Configuration for visualization."""

    host: str = Field(default="localhost")
    port: int = Field(default=8765)
    enabled: bool = Field(default=True)
    debug: bool = Field(default=False)

    model_config = ConfigDict(validate_assignment=True)


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field(default="INFO")
    file: Optional[str] = Field(default="orchestrator.log")

    model_config = ConfigDict(validate_assignment=True)


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""

    requests_per_minute: int = Field(default=60, ge=1)
    max_daily_tokens: int = Field(default=1000000, ge=1)
    alert_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    model_config = ConfigDict(validate_assignment=True)
