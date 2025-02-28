"""Core capabilities and limitations of LLM models."""
from typing import Any, Optional, Set, List, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class ProviderCapabilities(BaseModel):
    """Provider-level capabilities and features."""

    # API Features
    supports_batch_requests: bool = False
    supports_async_requests: bool = True
    supports_streaming: bool = True
    supports_model_selection: bool = True
    supports_custom_models: bool = False

    # Authentication & Security
    supports_api_key_auth: bool = True
    supports_oauth: bool = False
    supports_organization_ids: bool = False
    supports_custom_endpoints: bool = False

    # Rate Limiting
    supports_concurrent_requests: bool = True
    max_concurrent_requests: Optional[int] = None
    requests_per_minute: Optional[int] = None
    tokens_per_minute: Optional[int] = None

    # Billing & Usage
    supports_usage_tracking: bool = True
    supports_cost_tracking: bool = True
    supports_quotas: bool = True

    # Advanced Features
    supports_fine_tuning: bool = False
    supports_model_deployment: bool = False
    supports_custom_domains: bool = False
    supports_audit_logs: bool = False

    model_config = ConfigDict(validate_assignment=True)


class LLMCapabilities(BaseModel):
    """Core capabilities and limitations of a model."""

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
    supports_temperature: bool = False

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

    model_config = ConfigDict(validate_assignment=True)

    # List of valid capability flags that can be used as requirements
    VALID_CAPABILITY_FLAGS: ClassVar[Set[str]] = {
        "supports_streaming",
        "supports_function_calling",
        "supports_vision",
        "supports_embeddings",
        "supports_json_mode",
        "supports_system_prompt",
        "supports_tools",
        "supports_parallel_requests",
        "supports_frequency_penalty",
        "supports_presence_penalty",
        "supports_stop_sequences",
        "supports_message_role",
    }

    @classmethod
    def validate_capability_flags(cls, flags: Set[str]) -> None:
        """Validate that all flags are valid capability flags.

        Args:
            flags: Set of capability flags to validate

        Raises:
            ValueError: If any flag is not a valid capability flag
        """
        invalid_flags = flags - cls.VALID_CAPABILITY_FLAGS
        if invalid_flags:
            raise ValueError(
                f"Invalid capability flags: {invalid_flags}. " f"Valid flags are: {sorted(cls.VALID_CAPABILITY_FLAGS)}"
            )


class VisionCapabilities(BaseModel):
    """Vision-specific capabilities and limitations."""

    max_images_per_request: int = 1
    supported_formats: List[str] = ["png", "jpeg"]
    max_image_size_mb: int = 20
    max_image_resolution: int = 2048
    supports_image_annotations: bool = False
    supports_image_analysis: bool = False
    supports_image_generation: bool = False
    cost_per_image: float = 0.0

    model_config = ConfigDict(validate_assignment=True)
