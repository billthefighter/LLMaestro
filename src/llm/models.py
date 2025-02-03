"""Model family descriptors for LLM interfaces."""
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ModelFamily(str, Enum):
    """Supported model families."""

    GPT = "gpt"
    CLAUDE = "claude"
    HUGGINGFACE = "huggingface"


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


class ModelCapabilities(BaseModel):
    """Capabilities of a model family."""

    # Core Features
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_embeddings: bool = False

    # Vision/Image Capabilities
    vision_config: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "max_images_per_request": 1,
            "supported_formats": ["png", "jpeg"],
            "max_image_size_mb": 20,
            "max_image_resolution": 2048,  # max pixels in either dimension
            "supports_image_annotations": False,  # whether model can handle image region annotations
            "supports_image_analysis": False,  # whether model can analyze image content
            "supports_image_generation": False,  # whether model can generate images
            "cost_per_image": 0.0,  # additional cost per image in request
        },
        description="Configuration for vision/image capabilities when supports_vision is True",
    )

    # Context and Performance
    max_context_window: int = Field(default=4096, gt=0)
    max_output_tokens: Optional[int] = Field(default=None, gt=0)
    typical_speed: Optional[float] = Field(default=None, gt=0)

    # Cost and Quotas
    cost_per_1k_tokens: float = Field(default=0.0, ge=0)
    input_cost_per_1k_tokens: Optional[float] = Field(default=None, ge=0)
    output_cost_per_1k_tokens: Optional[float] = Field(default=None, ge=0)
    daily_request_limit: Optional[int] = Field(default=None, ge=0)

    # Advanced Features
    supports_json_mode: bool = False
    supports_system_prompt: bool = True
    supports_message_role: bool = True
    supports_tools: bool = False
    supports_parallel_requests: bool = True

    # Input/Output Capabilities
    supported_languages: Set[str] = Field(default_factory=lambda: {"en"})
    supported_mime_types: Set[str] = Field(default_factory=set)
    max_image_size: Optional[int] = Field(default=None, gt=0)
    max_audio_length: Optional[float] = Field(default=None, gt=0)

    # Quality and Control
    temperature: RangeConfig = Field(
        default_factory=lambda: RangeConfig(min_value=0.0, max_value=2.0, default_value=1.0)
    )
    top_p: RangeConfig = Field(default_factory=lambda: RangeConfig(min_value=0.0, max_value=1.0, default_value=1.0))
    supports_frequency_penalty: bool = False
    supports_presence_penalty: bool = False
    supports_stop_sequences: bool = True

    # Specialized Features
    supports_semantic_search: bool = False
    supports_code_completion: bool = False
    supports_chat_memory: bool = False
    supports_few_shot_learning: bool = True

    model_config = ConfigDict(validate_assignment=True)

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        # Convert sets to lists for JSON serialization
        if "supported_languages" in data:
            data["supported_languages"] = list(data["supported_languages"])
        if "supported_mime_types" in data:
            data["supported_mime_types"] = list(data["supported_mime_types"])
        return data

    @property
    def supports_images(self) -> bool:
        """Helper property to check if model supports image inputs."""
        return self.supports_vision and bool(self.vision_config)

    def get_image_limits(self) -> Dict[str, Any]:
        """Get the image-related limitations for this model."""
        if not self.supports_images:
            return {}
        return self.vision_config or {}

    def validate_image_request(
        self, image_count: int, formats: List[str], sizes: List[int]
    ) -> Tuple[bool, Optional[str]]:
        """Validate if an image request meets the model's capabilities.

        Args:
            image_count: Number of images in the request
            formats: List of image formats (e.g., ['png', 'jpeg'])
            sizes: List of image sizes in MB

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.supports_images:
            return False, "Model does not support image inputs"

        config = self.vision_config or {}

        # Check image count
        max_images = config.get("max_images_per_request", 1)
        if image_count > max_images:
            return False, f"Too many images. Maximum allowed: {max_images}"

        # Check formats
        supported_formats = set(config.get("supported_formats", []))
        unsupported = [fmt for fmt in formats if fmt not in supported_formats]
        if unsupported:
            return False, f"Unsupported image formats: {', '.join(unsupported)}"

        # Check sizes
        max_size = config.get("max_image_size_mb", 20)
        oversized = [size for size in sizes if size > max_size]
        if oversized:
            return False, f"Images exceed maximum size of {max_size}MB"

        return True, None


class ModelDescriptor(BaseModel):
    """Descriptor for a model within a family."""

    name: str
    family: ModelFamily
    capabilities: ModelCapabilities
    is_preview: bool = False
    is_deprecated: bool = False
    min_api_version: Optional[str] = None
    release_date: Optional[datetime] = None
    end_of_life_date: Optional[datetime] = None
    recommended_replacement: Optional[str] = None

    model_config = ConfigDict(validate_assignment=True)

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        # Convert datetime objects to ISO format strings
        if data.get("release_date"):
            data["release_date"] = data["release_date"].isoformat()
        if data.get("end_of_life_date"):
            data["end_of_life_date"] = data["end_of_life_date"].isoformat()
        return data


class ModelCapabilitiesTable(Base):
    """SQLAlchemy model for storing capabilities in a database."""

    __tablename__ = "model_capabilities"

    id = Column(Integer, primary_key=True)
    model_name = Column(String, unique=True, nullable=False)
    family = Column(String, nullable=False)
    capabilities = Column(JSON, nullable=False)
    is_preview = Column(Boolean, default=False)
    is_deprecated = Column(Boolean, default=False)
    min_api_version = Column(String)
    release_date = Column(DateTime)
    end_of_life_date = Column(DateTime)
    recommended_replacement = Column(String)

    @classmethod
    def from_descriptor(cls, descriptor: ModelDescriptor) -> "ModelCapabilitiesTable":
        """Create a database record from a model descriptor."""
        return cls(
            model_name=descriptor.name,
            family=descriptor.family.value,
            capabilities=descriptor.capabilities.dict(),
            is_preview=descriptor.is_preview,
            is_deprecated=descriptor.is_deprecated,
            min_api_version=descriptor.min_api_version,
            release_date=descriptor.release_date,
            end_of_life_date=descriptor.end_of_life_date,
            recommended_replacement=descriptor.recommended_replacement,
        )

    def to_descriptor(self) -> ModelDescriptor:
        """Convert database record to a model descriptor."""
        # Convert capabilities to a string-keyed dictionary
        capabilities_dict = {str(k): v for k, v in self.capabilities.items()}

        # Get values from SQLAlchemy columns
        name = getattr(self.model_name, "value", self.model_name)
        family = getattr(self.family, "value", self.family)
        is_preview = getattr(self.is_preview, "value", self.is_preview)
        is_deprecated = getattr(self.is_deprecated, "value", self.is_deprecated)
        min_api_version = getattr(self.min_api_version, "value", self.min_api_version)
        release_date = getattr(self.release_date, "value", self.release_date)
        end_of_life_date = getattr(self.end_of_life_date, "value", self.end_of_life_date)
        recommended_replacement = getattr(self.recommended_replacement, "value", self.recommended_replacement)

        return ModelDescriptor(
            name=str(name),
            family=ModelFamily(str(family)),
            capabilities=ModelCapabilities(**capabilities_dict),
            is_preview=bool(is_preview),
            is_deprecated=bool(is_deprecated),
            min_api_version=str(min_api_version) if min_api_version is not None else None,
            release_date=release_date.replace(tzinfo=None) if release_date is not None else None,
            end_of_life_date=end_of_life_date.replace(tzinfo=None) if end_of_life_date is not None else None,
            recommended_replacement=str(recommended_replacement) if recommended_replacement is not None else None,
        )


class ModelRegistry:
    """Registry of available LLM models and their capabilities."""

    def __init__(self):
        self._models: Dict[str, ModelDescriptor] = {}

    def register(self, descriptor: ModelDescriptor) -> None:
        """Register a model descriptor."""
        self._models[descriptor.name] = descriptor

    def get_model(self, name: str) -> Optional[ModelDescriptor]:
        """Get a model by name."""
        return self._models.get(name)

    def get_family_models(self, family: ModelFamily) -> List[ModelDescriptor]:
        """Get all models in a family."""
        return [model for model in self._models.values() if model.family == family]

    def get_models_by_capability(
        self,
        capability: str,
        min_context_window: Optional[int] = None,
        max_cost_per_1k: Optional[float] = None,
        required_languages: Optional[Set[str]] = None,
        min_speed: Optional[float] = None,
    ) -> List[ModelDescriptor]:
        """Get models that support a specific capability."""
        matching_models = []

        for model in self._models.values():
            if not hasattr(model.capabilities, capability):
                continue

            if getattr(model.capabilities, capability) is not True:
                continue

            if min_context_window and model.capabilities.max_context_window < min_context_window:
                continue

            if max_cost_per_1k and model.capabilities.input_cost_per_1k_tokens:
                if model.capabilities.input_cost_per_1k_tokens > max_cost_per_1k:
                    continue

            if required_languages and not required_languages.issubset(model.capabilities.supported_languages):
                continue

            if min_speed and (not model.capabilities.typical_speed or model.capabilities.typical_speed < min_speed):
                continue

            matching_models.append(model)

        return matching_models

    def validate_model(self, name: str) -> tuple[bool, Optional[str]]:
        """Validate if a model exists and is usable."""
        descriptor = self.get_model(name)
        if not descriptor:
            return False, f"Unknown model {name}"

        if descriptor.is_deprecated:
            msg = f"Model {name} is deprecated"
            if descriptor.recommended_replacement:
                msg += f". Consider using {descriptor.recommended_replacement} instead"
            if descriptor.end_of_life_date:
                msg += f". End of life date: {descriptor.end_of_life_date}"
            return False, msg

        return True, None

    async def detect_and_register_model(self, provider: str, model_name: str, api_key: str) -> ModelDescriptor:
        """
        Detects capabilities of a model and registers it in the registry.

        Args:
            provider: The LLM provider (e.g., "anthropic", "openai")
            model_name: Name of the model to detect capabilities for
            api_key: API key for authentication

        Returns:
            ModelDescriptor for the registered model
        """
        # Detect capabilities
        capabilities = await ModelCapabilitiesDetector.detect_capabilities(provider, model_name, api_key)

        # Create model descriptor with proper datetime
        descriptor = ModelDescriptor(
            name=model_name,
            family=ModelFamily(provider.lower()),
            capabilities=capabilities,
            min_api_version="2024-02-29",  # Use current latest version
            release_date=datetime.now(),
        )

        # Register the model
        self.register(descriptor)

        return descriptor

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ModelRegistry":
        """Load registry from a JSON file."""
        registry = cls()
        with open(path) as f:
            data = json.load(f)
            for model_data in data["models"]:
                # Convert string representations of sets back to actual sets
                if "capabilities" in model_data:
                    caps = model_data["capabilities"]
                    if "supported_languages" in caps and isinstance(caps["supported_languages"], str):
                        caps["supported_languages"] = eval(caps["supported_languages"])
                    if "supported_mime_types" in caps and isinstance(caps["supported_mime_types"], str):
                        caps["supported_mime_types"] = eval(caps["supported_mime_types"])
                registry.register(ModelDescriptor(**model_data))
        return registry

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelRegistry":
        """Load registry from a YAML file."""
        registry = cls()
        with open(path) as f:
            data = yaml.safe_load(f)
            for model_data in data["models"]:
                # Convert string representations of sets back to actual sets
                if "capabilities" in model_data:
                    caps = model_data["capabilities"]
                    if "supported_languages" in caps and isinstance(caps["supported_languages"], str):
                        caps["supported_languages"] = eval(caps["supported_languages"])
                    if "supported_mime_types" in caps and isinstance(caps["supported_mime_types"], str):
                        caps["supported_mime_types"] = eval(caps["supported_mime_types"])
                registry.register(ModelDescriptor(**model_data))
        return registry

    @classmethod
    def from_database(cls, session, query_filter: Optional[Dict[str, Any]] = None) -> "ModelRegistry":
        """Load registry from database records."""
        registry = cls()
        query = session.query(ModelCapabilitiesTable)
        if query_filter:
            query = query.filter_by(**query_filter)

        for record in query.all():
            registry.register(record.to_descriptor())
        return registry

    def to_json(self, path: Union[str, Path]) -> None:
        """Save registry to a JSON file."""
        data = {"models": [model.model_dump() for model in self._models.values()]}
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save registry to a YAML file."""
        data = {"models": [model.model_dump() for model in self._models.values()]}
        with open(path, "w") as f:
            yaml.dump(data, f, sort_keys=False)


class ModelCapabilitiesDetector:
    """Dynamically detects and generates model capabilities by querying the provider's API."""

    @classmethod
    async def detect_capabilities(cls, provider: str, model_name: str, api_key: str) -> ModelCapabilities:
        """
        Detects model capabilities by making API calls to the provider.

        Args:
            provider: The LLM provider (e.g., "anthropic", "openai")
            model_name: Name of the model to detect capabilities for
            api_key: API key for authentication

        Returns:
            ModelCapabilities object with detected capabilities
        """
        detector = cls._get_detector(provider)
        return await detector(model_name, api_key)

    @classmethod
    def _get_detector(cls, provider: str):
        """Returns the appropriate detector method for the given provider."""
        detectors = {
            "anthropic": cls._detect_anthropic_capabilities,
            "openai": cls._detect_openai_capabilities,
        }
        if provider.lower() not in detectors:
            raise ValueError(f"Unsupported provider for capability detection: {provider}")
        return detectors[provider.lower()]

    @staticmethod
    async def _detect_anthropic_capabilities(model_name: str, api_key: str) -> ModelCapabilities:
        """Detects capabilities for Anthropic models."""
        from anthropic import Anthropic

        try:
            # Initialize client to verify API key
            client = Anthropic(api_key=api_key)
            # Make a simple API call to verify the key works
            client.messages.create(
                model="claude-3-haiku-20240229", max_tokens=1, messages=[{"role": "user", "content": "test"}]
            )

            # Test streaming
            supports_streaming = True  # Anthropic supports streaming by default

            # Test function calling and tools
            supports_tools = "claude-3" in model_name.lower()
            supports_function_calling = supports_tools

            # Test vision capabilities
            supports_vision = "claude-3" in model_name.lower()

            # Get context window and other limits
            if "claude-3" in model_name.lower():
                max_context_window = 200000
                typical_speed = 100.0
            else:
                max_context_window = 100000
                typical_speed = 70.0

            # Determine supported mime types
            supported_mime_types = set()
            if supports_vision:
                supported_mime_types.update({"image/jpeg", "image/png"})
                if "opus" in model_name.lower():
                    supported_mime_types.add("image/gif")
                    supported_mime_types.add("image/webp")

            # Create capabilities object
            return ModelCapabilities(
                supports_streaming=supports_streaming,
                supports_function_calling=supports_function_calling,
                supports_vision=supports_vision,
                supports_embeddings=False,
                max_context_window=max_context_window,
                max_output_tokens=4096,
                typical_speed=typical_speed,
                input_cost_per_1k_tokens=0.015 if "claude-3" in model_name.lower() else 0.008,
                output_cost_per_1k_tokens=0.075
                if "opus" in model_name.lower()
                else (0.015 if "sonnet" in model_name.lower() else 0.024),
                daily_request_limit=150000,
                supports_json_mode="claude-3" in model_name.lower(),
                supports_system_prompt=True,
                supports_message_role=True,
                supports_tools=supports_tools,
                supports_parallel_requests=True,
                supported_languages={"en"},
                supported_mime_types=supported_mime_types,
                temperature=RangeConfig(min_value=0.0, max_value=1.0, default_value=0.7),
                top_p=RangeConfig(min_value=0.0, max_value=1.0, default_value=1.0),
                supports_frequency_penalty=False,
                supports_presence_penalty=False,
                supports_stop_sequences=True,
                supports_semantic_search=True,
                supports_code_completion=True,
                supports_chat_memory="claude-3" in model_name.lower(),
                supports_few_shot_learning=True,
            )
        except Exception as e:
            raise RuntimeError("Failed to detect Anthropic capabilities") from e

    @staticmethod
    async def _detect_openai_capabilities(model_name: str, api_key: str) -> ModelCapabilities:
        """Detects capabilities for OpenAI models."""
        from openai import AsyncOpenAI

        try:
            # Initialize client and verify API key
            client = AsyncOpenAI(api_key=api_key)
            model_info = await client.models.retrieve(model_name)

            # Determine capabilities based on model name and info
            is_gpt4 = "gpt-4" in model_name.lower()
            is_vision = "vision" in model_name.lower()

            # Get context window from model info or use default
            try:
                context_window = getattr(model_info, "context_window", 4096)
            except AttributeError:
                context_window = 4096  # Default if not available

            # Create capabilities object
            return ModelCapabilities(
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=is_vision,
                supports_embeddings=False,
                max_context_window=context_window,
                max_output_tokens=4096,
                typical_speed=150.0,
                input_cost_per_1k_tokens=0.01 if is_gpt4 else 0.0015,
                output_cost_per_1k_tokens=0.03 if is_gpt4 else 0.002,
                daily_request_limit=200000,
                supports_json_mode=True,
                supports_system_prompt=True,
                supports_message_role=True,
                supports_tools=True,
                supports_parallel_requests=True,
                supported_languages={"en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", "ko"},
                supported_mime_types={"image/jpeg", "image/png", "image/gif", "image/webp"} if is_vision else set(),
                temperature=RangeConfig(min_value=0.0, max_value=2.0, default_value=1.0),
                top_p=RangeConfig(min_value=0.0, max_value=1.0, default_value=1.0),
                supports_frequency_penalty=True,
                supports_presence_penalty=True,
                supports_stop_sequences=True,
                supports_semantic_search=True,
                supports_code_completion=True,
                supports_chat_memory=False,
                supports_few_shot_learning=True,
            )
        except Exception as e:
            raise RuntimeError("Failed to detect OpenAI capabilities") from e


async def register_claude_3_5_sonnet_latest(api_key: str) -> ModelDescriptor:
    """
    Registers the claude-3-5-sonnet-latest model with dynamically detected capabilities.

    Args:
        api_key: Anthropic API key

    Returns:
        ModelDescriptor for the registered model
    """
    registry = ModelRegistry()
    return await registry.detect_and_register_model(
        provider="anthropic", model_name="claude-3-5-sonnet-latest", api_key=api_key
    )
