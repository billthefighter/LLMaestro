"""Model definitions and capabilities for LLM interfaces."""
import mimetypes
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from llmaestro.config.base import RateLimitConfig
from llmaestro.core.persistence import PersistentModel
from llmaestro.llm.capabilities import LLMCapabilities, ProviderCapabilities, VisionCapabilities
from llmaestro.llm.credentials import APIKey


class TokenUsage(BaseModel):
    """Token usage statistics for LLM responses."""

    prompt_tokens: int = Field(ge=0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(ge=0, description="Number of tokens in the completion")
    total_tokens: int = Field(ge=0, description="Total number of tokens used")

    model_config = ConfigDict(validate_assignment=True)


class ContextMetrics(BaseModel):
    """Metrics about context window usage."""

    max_context_tokens: int = Field(ge=0, description="Maximum context window size in tokens")
    current_context_tokens: int = Field(ge=0, description="Current number of tokens in context")
    available_tokens: int = Field(ge=0, description="Number of tokens available in context")
    context_utilization: float = Field(ge=0.0, le=1.0, description="Context window utilization (0-1)")

    model_config = ConfigDict(validate_assignment=True)


class LLMMetadata(BaseModel):
    """Metadata about a model's lifecycle and status."""

    release_date: Optional[datetime] = None
    is_preview: bool = False
    is_deprecated: bool = False
    end_of_life_date: Optional[datetime] = None
    recommended_replacement: Optional[str] = None
    min_api_version: Optional[str] = None

    model_config = ConfigDict(validate_assignment=True)


class Provider(PersistentModel):
    """Configuration for an LLM provider."""

    family: str
    description: Optional[str] = None

    capabilities: ProviderCapabilities = Field(description="Provider-level capabilities")
    api_base: str = Field(
        description="Base URL for the provider's API",
        pattern=r"^https?://[^\s/$.?#].[^\s]*$",  # Basic URL validation
    )
    rate_limits: RateLimitConfig = Field(default_factory=RateLimitConfig, description="Rate limits for the provider")

    model_config = ConfigDict(validate_assignment=True)

    def __str__(self) -> str:
        return f"Provider ({self.family})"

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration

        Returns:
            Dictionary containing API configuration
        """
        return {
            "api_base": self.api_base,
            "rate_limits": self.rate_limits.model_dump(),
            "features": [
                f for f in dir(self.capabilities) if f.startswith("supports_") and getattr(self.capabilities, f)
            ],
        }

    def validate_api_base(self) -> None:
        """Validate the api_base URL."""
        try:
            result = urlparse(self.api_base)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid API base URL format")
        except Exception as err:
            raise ValueError(f"Invalid API base URL: {str(err)}") from err


class LLMProfile(PersistentModel):
    """Complete profile of an LLM's capabilities and metadata."""

    name: str
    version: Optional[str] = None
    description: Optional[str] = None
    capabilities: LLMCapabilities
    metadata: LLMMetadata = Field(default_factory=LLMMetadata)
    vision_capabilities: Optional[VisionCapabilities] = None

    model_config = ConfigDict(validate_assignment=True)

    def supports_media_type(self, media_type: str) -> bool:
        """Check if a specific media type is supported."""
        if not self.vision_capabilities:
            return False
        mime_type = mimetypes.guess_type(media_type)[0]
        return bool(mime_type and any(mime_type.endswith(fmt) for fmt in self.vision_capabilities.supported_formats))

    def validate_image_request(
        self, image_count: int, formats: List[str], sizes: List[int]
    ) -> tuple[bool, Optional[str]]:
        """Validate if an image request meets the model's capabilities."""
        if not self.vision_capabilities:
            return False, "Model does not support image inputs"

        if image_count > self.vision_capabilities.max_images_per_request:
            return False, f"Too many images. Maximum allowed: {self.vision_capabilities.max_images_per_request}"

        unsupported = [fmt for fmt in formats if fmt not in self.vision_capabilities.supported_formats]
        if unsupported:
            return False, f"Unsupported image formats: {', '.join(unsupported)}"

        oversized = [size for size in sizes if size > self.vision_capabilities.max_image_size_mb]
        if oversized:
            return False, f"Images exceed maximum size of {self.vision_capabilities.max_image_size_mb}MB"

        return True, None


class LLMRuntimeConfig(PersistentModel):
    """Runtime configuration for LLM instances."""

    max_tokens: int = Field(default=2048, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(
        default=None, description="Sampling temperature (None means use model default)"
    )
    max_context_tokens: int = Field(default=4096, description="Maximum context window size")
    stream: bool = Field(default=False, description="Whether to stream the response")
    rate_limit: Optional[RateLimitConfig] = Field(default=None, description="Rate limiting configuration")

    model_config = ConfigDict(validate_assignment=True)


class LLMState(PersistentModel):
    """Complete state container for LLM instances."""

    profile: LLMProfile = Field(description="Model profile containing capabilities and metadata")
    provider: Provider = Field(description="Provider configuration")
    runtime_config: LLMRuntimeConfig = Field(description="Runtime configuration")

    model_config = ConfigDict(validate_assignment=True)

    @property
    def model_family(self) -> str:
        """Get the model family."""
        return self.provider.family

    @property
    def model_name(self):
        """Get the model name."""
        return self.profile.name


if TYPE_CHECKING:
    from llmaestro.llm.interfaces.base import BaseLLMInterface

    InterfaceType = BaseLLMInterface
else:
    InterfaceType = Any


class LLMInstance(PersistentModel):
    """Runtime container that combines interface, state and credentials for an LLM."""

    # Core components
    state: LLMState = Field(description="Configuration and metadata for the LLM")
    interface: InterfaceType = Field(description="Active interface instance for LLM interactions")
    credentials: Optional[APIKey] = Field(default=None, description="API credentials for this instance")

    # Runtime metadata
    is_initialized: bool = Field(default=False, description="Whether this instance has been fully initialized")
    is_healthy: bool = Field(default=True, description="Whether this instance is currently healthy and operational")
    last_error: Optional[str] = Field(default=None, description="Last error encountered by this instance")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        from_attributes=True,
        extra="allow",
        revalidate_instances="always",
    )

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.state.profile.name

    @property
    def model_family(self) -> str:
        """Get the model family."""
        return self.state.model_family

    @property
    def is_ready(self) -> bool:
        """Check if instance is ready for use."""
        return (
            self.is_initialized
            and self.is_healthy
            and self.interface is not None
            and (self.credentials is not None or self.interface.ignore_missing_credentials)
        )

    async def shutdown(self) -> None:
        """Gracefully shutdown this instance."""
        if self.interface and hasattr(self.interface, "shutdown"):
            await self.interface.shutdown()
        self.is_initialized = False
