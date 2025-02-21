"""Base configuration models for the application."""

from typing import Any, Dict, Optional

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


class LLMProfileReference(BaseModel):
    """Reference to an LLMProfile with optional settings overrides."""

    provider: str = Field(description="Provider name (e.g., 'openai', 'anthropic')")
    name: str = Field(description="Model name (e.g., 'gpt-4-turbo-preview')")
    settings: Dict[str, Any] = Field(
        default_factory=dict, description="Optional settings that override the model's default settings"
    )

    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    def default(cls) -> "LLMProfileReference":
        """Get the default model reference."""
        return cls(
            provider="anthropic", name="claude-3-sonnet-latest", settings={"temperature": 0.7, "max_tokens": 4096}
        )
