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


class DefaultModelConfig(BaseModel):
    """Configuration for default model."""

    provider: str
    name: str
    settings: dict = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)
