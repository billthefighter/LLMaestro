"""Registry for managing LLM models and their capabilities."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml

from llmaestro.llm.models import ModelCapabilities, ModelCapabilitiesTable, ModelDescriptor, ModelFamily, RangeConfig
from llmaestro.llm.provider_registry import ProviderConfig, ProviderRegistry


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
            client.messages.create(model=model_name, max_tokens=1, messages=[{"role": "user", "content": "test"}])

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
            supported_media_types = set()
            if supports_vision:
                supported_media_types.update({"image/jpeg", "image/png"})
                if "opus" in model_name.lower():
                    supported_media_types.add("image/gif")
                    supported_media_types.add("image/webp")

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
                supported_media_types=supported_media_types,
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
                supported_media_types={"image/jpeg", "image/png", "image/gif", "image/webp"} if is_vision else set(),
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


class ModelRegistry:
    """Registry of available LLM models and their capabilities."""

    def __init__(self, provider_registry: Optional[ProviderRegistry] = None):
        self._models: Dict[str, ModelDescriptor] = {}
        self._provider_registry = provider_registry or ProviderRegistry()

    def register(self, descriptor: ModelDescriptor) -> None:
        """Register a model descriptor."""
        self._models[descriptor.name] = descriptor

    def register_from_provider(self, provider: str, model_name: str) -> None:
        """Register a model using provider configuration.

        Args:
            provider: Provider name
            model_name: Model name to register

        Raises:
            ValueError: If provider or model configuration not found
        """
        model_config = self._provider_registry.get_provider_model_config(provider, model_name)
        if not model_config:
            raise ValueError(f"No configuration found for model {model_name} from provider {provider}")

        # Create basic capabilities from provider model config
        capabilities = ModelCapabilities(
            max_context_window=model_config.context_window,
            typical_speed=model_config.typical_speed,
            input_cost_per_1k_tokens=model_config.cost.get("input_per_1k", 0.0),
            output_cost_per_1k_tokens=model_config.cost.get("output_per_1k", 0.0),
            supported_languages={"en"},  # Default to English
            supports_streaming=True,  # Most modern models support this
        )

        # Add features from provider config
        for feature in model_config.features:
            if hasattr(capabilities, feature):
                setattr(capabilities, feature, True)

        descriptor = ModelDescriptor(
            name=model_name,
            family=model_config.family,
            capabilities=capabilities,
        )
        self.register(descriptor)

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

    def get_provider_config(self, provider: str) -> Optional[ProviderConfig]:
        """Get configuration for a provider."""
        return self._provider_registry.get_provider(provider)

    async def detect_and_register_model(self, provider: str, model_name: str, api_key: str) -> ModelDescriptor:
        """Detects capabilities of a model and registers it in the registry.

        Args:
            provider: The LLM provider (e.g., "anthropic", "openai")
            model_name: Name of the model to detect capabilities for
            api_key: API key for authentication

        Returns:
            ModelDescriptor for the registered model
        """
        # First check if provider configuration exists
        provider_config = self.get_provider_config(provider)
        if provider_config:
            try:
                self.register_from_provider(provider, model_name)
                model = self.get_model(model_name)
                if model:
                    return model
            except ValueError:
                pass  # Fall through to dynamic detection if provider registration fails

        # Detect capabilities dynamically
        capabilities = await ModelCapabilitiesDetector.detect_capabilities(provider, model_name, api_key)

        # Create and register descriptor
        descriptor = ModelDescriptor(
            name=model_name,
            family=provider,  # Use provider as family when not in provider registry
            capabilities=capabilities,
            min_api_version="2024-02-29",
            release_date=datetime.now(),
        )
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
                    if "supported_media_types" in caps and isinstance(caps["supported_media_types"], str):
                        caps["supported_media_types"] = eval(caps["supported_media_types"])
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
                    if "supported_media_types" in caps and isinstance(caps["supported_media_types"], str):
                        caps["supported_media_types"] = eval(caps["supported_media_types"])
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
