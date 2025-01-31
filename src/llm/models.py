"""Model family descriptors for LLM interfaces."""
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

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

    class Config:
        """Pydantic config."""

        json_encoders = {set: list, datetime: lambda v: v.isoformat()}


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

    class Config:
        """Pydantic config."""

        json_encoders = {datetime: lambda v: v.isoformat()}


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
        return ModelDescriptor(
            name=self.model_name,
            family=ModelFamily(self.family),
            capabilities=ModelCapabilities(**self.capabilities),
            is_preview=self.is_preview,
            is_deprecated=self.is_deprecated,
            min_api_version=self.min_api_version,
            release_date=self.release_date,
            end_of_life_date=self.end_of_life_date,
            recommended_replacement=self.recommended_replacement,
        )


class ModelRegistry:
    """Registry of model families and their capabilities."""

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


ANTHROPIC_MODELS = {
    "claude-3-sonnet": ModelDescriptor(
        name="claude-3-sonnet",
        provider="anthropic",
        min_api_version="2024-03-07",
        release_date="2024-03-07",
        context_window=200000,
        max_tokens=4096,
        token_encoding="cl100k_base",
        capabilities=[
            "text",
            "code",
            "analysis",
            "math",
            "extraction",
            "classification",
            "json",
        ],
        description="Claude 3 Sonnet model from Anthropic, released March 2024",
    ),
}
