"""Core capabilities and limitations of LLM models."""
from typing import Any, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import ModelFamily


class RangeConfig(BaseModel):
    """Configuration for a numeric range."""

    min_value: float
    max_value: float
    default_value: float

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("max_value")
    def max_value_must_be_greater_than_min(cls, v: float, info: Any) -> float:
        if "min_value" in info.data and v < info.data["min_value"]:
            raise ValueError("max_value must be greater than min_value")
        return v

    @field_validator("default_value")
    def default_value_must_be_in_range(cls, v: float, info: Any) -> float:
        if "min_value" in info.data and v < info.data["min_value"]:
            raise ValueError("default_value must be greater than or equal to min_value")
        if "max_value" in info.data and v > info.data["max_value"]:
            raise ValueError("default_value must be less than or equal to max_value")
        return v


class LLMCapabilities(BaseModel):
    """Core capabilities and limitations of a model."""

    # Identity (required)
    name: str
    family: ModelFamily
    version: Optional[str] = None
    description: Optional[str] = None

    # Resource Limits
    max_context_window: int = Field(default=4096, gt=0)
    max_output_tokens: Optional[int] = Field(default=None, gt=0)

    # Core Features
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_embeddings: bool = False
    supports_json_mode: bool = False
    supports_system_prompt: bool = True
    supports_tools: bool = False
    supports_parallel_requests: bool = True

    # Advanced Features
    supports_frequency_penalty: bool = False
    supports_presence_penalty: bool = False
    supports_stop_sequences: bool = True
    supports_message_role: bool = True

    # Performance & Cost
    typical_speed: Optional[float] = None
    supported_languages: Set[str] = Field(default_factory=lambda: {"en"})
    input_cost_per_1k_tokens: float = Field(default=0.0, ge=0)
    output_cost_per_1k_tokens: float = Field(default=0.0, ge=0)

    # Quality Settings
    temperature: RangeConfig = Field(
        default_factory=lambda: RangeConfig(min_value=0.0, max_value=2.0, default_value=1.0)
    )
    top_p: RangeConfig = Field(default_factory=lambda: RangeConfig(min_value=0.0, max_value=1.0, default_value=1.0))

    model_config = ConfigDict(validate_assignment=True)

    def supports_feature(self, feature_name: str) -> bool:
        """Check if a specific feature is supported."""
        attr_name = f"supports_{feature_name}"
        return getattr(self, attr_name, False) if hasattr(self, attr_name) else False

    def get_limit(self, limit_name: str) -> Optional[int]:
        """Get a specific resource limit."""
        if limit_name == "context_window":
            return self.max_context_window
        elif limit_name == "output_tokens":
            return self.max_output_tokens
        return None
