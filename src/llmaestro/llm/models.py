"""Model definitions and capabilities for LLM interfaces."""
import mimetypes
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Type
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .capabilities import LLMCapabilities
from .capability_detector import BaseCapabilityDetector
from .enums import ModelFamily


class LLMMetadata(BaseModel):
    """Metadata about a model's lifecycle and status."""

    release_date: Optional[datetime] = None
    is_preview: bool = False
    is_deprecated: bool = False
    end_of_life_date: Optional[datetime] = None
    recommended_replacement: Optional[str] = None
    min_api_version: Optional[str] = None

    model_config = ConfigDict(validate_assignment=True)


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


class LLMProfile(BaseModel):
    """Complete profile of an LLM's capabilities and metadata."""

    capabilities: LLMCapabilities
    metadata: LLMMetadata
    vision_capabilities: Optional[VisionCapabilities] = None

    model_config = ConfigDict(validate_assignment=True)

    @property
    def name(self) -> str:
        """LLM name from capabilities."""
        return self.capabilities.name

    @property
    def family(self) -> ModelFamily:
        """LLM family from capabilities."""
        return self.capabilities.family

    def supports_feature(self, feature: str) -> bool:
        """Check if a specific feature is supported."""
        return self.capabilities.supports_feature(feature)

    def get_limit(self, limit_type: str) -> Optional[int]:
        """Get a specific resource limit."""
        return self.capabilities.get_limit(limit_type)

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


class Provider(BaseModel):
    """Configuration for an LLM provider."""

    name: str
    api_base: str = Field(
        description="Base URL for the provider's API",
        pattern=r"^https?://[^\s/$.?#].[^\s]*$",  # Basic URL validation
    )
    capabilities_detector: Optional[Type[BaseCapabilityDetector]] = Field(
        default=None, description="Optional capability detector class for this provider"
    )
    rate_limits: Dict[str, int]
    features: Optional[Set[str]] = None

    def get_api_config(self, model_name: str) -> Dict[str, Any]:
        """Get API configuration for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary containing API configuration
        """
        return {
            "api_base": self.api_base,
            "rate_limits": self.rate_limits,
            "features": self.features or set(),
            "model": model_name,
        }

    @property
    def models(self) -> Dict[str, LLMProfile]:
        """Get models associated with this provider.

        This property ensures we only get models that are actually associated
        with this provider, preventing synchronization issues.
        """
        from .llm_registry import LLMRegistry

        registry = LLMRegistry.create_default()
        return {
            name: model
            for name, model in registry._models.items()
            if model.capabilities.family == ModelFamily(self.name)
        }

    def validate_api_base(self) -> None:
        """Validate the api_base URL."""
        try:
            result = urlparse(self.api_base)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid API base URL format")
        except Exception as e:
            raise ValueError(f"Invalid API base URL: {str(e)}")

    @field_validator("capabilities_detector")
    def validate_capabilities_detector(cls, v: Type[BaseCapabilityDetector]) -> Type[BaseCapabilityDetector]:
        """Validate the capabilities detector is a proper subclass."""
        if not issubclass(v, BaseCapabilityDetector):
            raise ValueError("Capabilities detector must be a subclass of BaseCapabilityDetector")
        return v

    model_config = ConfigDict(validate_assignment=True)
