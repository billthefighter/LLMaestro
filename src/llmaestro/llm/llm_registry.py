"""Registry for managing LLM models."""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import yaml

from .capability_factory import ModelCapabilityDetectorFactory
from .models import LLMCapabilities, LLMMetadata, LLMProfile, ModelFamily
from .provider_registry import Provider, ProviderRegistry


@dataclass
class LLMRegistry:
    """Registry for managing LLM models and their configurations."""

    provider_registry: ProviderRegistry = field(default_factory=ProviderRegistry)
    strict_capability_detection: bool = False
    _models: Dict[str, LLMProfile] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        pass  # No longer needed since provider_registry is properly initialized

    def register(self, descriptor: LLMProfile) -> None:
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
        provider_config = self.provider_registry.get_provider(provider)
        if not provider_config:
            raise ValueError(f"Unknown provider: {provider}")

        model_config = provider_config.models.get(model_name)
        if not model_config:
            raise ValueError(f"No configuration found for model {model_name} from provider {provider}")

        # Create basic capabilities from provider model config
        capabilities = LLMCapabilities(
            name=model_name,
            family=model_config.capabilities.family,
            max_context_window=model_config.capabilities.max_context_window,
            typical_speed=model_config.capabilities.typical_speed,
            input_cost_per_1k_tokens=model_config.capabilities.input_cost_per_1k_tokens,
            output_cost_per_1k_tokens=model_config.capabilities.output_cost_per_1k_tokens,
            supported_languages={"en"},  # Default to English
            supports_streaming=True,  # Most modern models support this
        )

        # Add features from provider config if available
        if provider_config.features:
            for feature in provider_config.features:
                feature_name = feature.replace("supports_", "")
                feature_attr = f"supports_{feature_name}"
                if hasattr(capabilities, feature_attr):
                    setattr(capabilities, feature_attr, True)

        # Create metadata with defaults
        metadata = LLMMetadata(
            is_preview=False,
            is_deprecated=False,
        )

        descriptor = LLMProfile(
            capabilities=capabilities,
            metadata=metadata,
        )
        self.register(descriptor)

    async def detect_and_register_model(self, provider: str, model_name: str, api_key: str) -> LLMProfile:
        """Detects capabilities of a model and registers it in the registry.

        Args:
            provider: The LLM provider (e.g., "anthropic", "openai")
            model_name: Name of the model to detect capabilities for
            api_key: API key for authentication

        Returns:
            LLMProfile for the registered model

        Raises:
            ValueError: If strict_capability_detection is True and capability detection fails
        """
        logger = logging.getLogger(__name__)

        # First check if provider configuration exists
        provider_config = self.provider_registry.get_provider(provider)
        if provider_config:
            try:
                self.register_from_provider(provider, model_name)
                model = self.get_model(model_name)
                if model:
                    return model
            except ValueError:
                pass  # Fall through to dynamic detection if provider registration fails

        try:
            # Get the appropriate detector and detect capabilities
            capabilities = await ModelCapabilityDetectorFactory.detect_capabilities(provider, model_name, api_key)
            capabilities.name = model_name
            capabilities.family = ModelFamily(provider)  # Convert provider string to ModelFamily enum
        except (ValueError, RuntimeError) as e:
            if self.strict_capability_detection:
                raise ValueError(f"Failed to detect capabilities for {model_name}: {str(e)}") from e
            else:
                logger.warning(
                    f"Capability detection failed for {model_name}, using default capabilities. Error: {str(e)}"
                )
                # Create basic capabilities with conservative defaults
                capabilities = LLMCapabilities(
                    name=model_name,
                    family=ModelFamily(provider),
                    max_context_window=4096,  # Conservative default
                    typical_speed=50.0,  # Conservative estimate
                    input_cost_per_1k_tokens=0.01,  # Default cost
                    output_cost_per_1k_tokens=0.02,  # Default cost
                    supports_streaming=True,  # Most modern models support this
                )

        # Create metadata with release info
        metadata = LLMMetadata(
            release_date=datetime.now(),
            min_api_version="2024-02-29",
        )

        # Create and register descriptor
        descriptor = LLMProfile(
            capabilities=capabilities,
            metadata=metadata,
        )
        self.register(descriptor)
        return descriptor

    def get_model(self, name: str) -> Optional[LLMProfile]:
        """Get a model by name."""
        return self._models.get(name)

    def get_family_models(self, family: ModelFamily) -> List[LLMProfile]:
        """Get all models in a family."""
        return [model for model in self._models.values() if model.family == family]

    def get_models_by_capability(
        self,
        capability: str,
        min_context_window: Optional[int] = None,
        max_cost_per_1k: Optional[float] = None,
        required_languages: Optional[Set[str]] = None,
        min_speed: Optional[float] = None,
    ) -> List[LLMProfile]:
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

        if descriptor.metadata.is_deprecated:
            msg = f"Model {name} is deprecated"
            if descriptor.metadata.recommended_replacement:
                msg += f". Consider using {descriptor.metadata.recommended_replacement} instead"
            if descriptor.metadata.end_of_life_date:
                msg += f". End of life date: {descriptor.metadata.end_of_life_date}"
            return False, msg

        return True, None

    def get_provider_config(self, provider: str) -> Optional[Provider]:
        """Get configuration for a provider."""
        return self.provider_registry.get_provider(provider)

    def list_models(self) -> List[str]:
        """Get a list of all registered model names.

        Returns:
            List of model names in the registry
        """
        return list(self._models.keys())

    @property
    def models(self) -> List[str]:
        """Get a list of all registered model names.

        Returns:
            List of model names in the registry
        """
        return self.list_models()

    @classmethod
    def load_from_file(cls, path: Union[str, Path], strict_capability_detection: bool = False) -> "LLMRegistry":
        """Load registry from a JSON or YAML file.

        The file can either be:
        1. A single configuration file with "providers" and "models" sections
        2. A provider-specific file with "provider" and "models" sections

        Args:
            path: Path to configuration file
            strict_capability_detection: Whether to enforce strict capability detection

        Returns:
            Configured LLMRegistry instance
        """
        registry = cls(strict_capability_detection=strict_capability_detection)

        with open(path) as f:
            data = json.load(f) if str(path).endswith(".json") else yaml.safe_load(f)

            # Handle provider configuration
            if "provider" in data:
                # Provider-specific file format
                provider = Provider(**data["provider"])
                registry.provider_registry.register_provider(provider.name, provider)
            elif "providers" in data:
                # Multi-provider configuration format
                for provider_data in data["providers"]:
                    provider = Provider(**provider_data)
                    registry.provider_registry.register_provider(provider.name, provider)

            # Load model configurations
            for model_data in data.get("models", []):
                # Convert string representations of sets back to actual sets
                if "capabilities" in model_data:
                    caps = model_data["capabilities"]
                    for set_field in ["supported_languages", "supported_media_types"]:
                        if set_field in caps and isinstance(caps[set_field], str):
                            caps[set_field] = eval(caps[set_field])

                # If this is a provider-specific file, ensure family matches provider
                if "provider" in data:
                    if "capabilities" not in model_data:
                        model_data["capabilities"] = {}
                    model_data["capabilities"]["family"] = ModelFamily(data["provider"]["provider_name"])

                registry.register(LLMProfile(**model_data))

        return registry

    @classmethod
    def create_default(cls, strict_capability_detection: bool = False) -> "LLMRegistry":
        """Create a LLMRegistry with default configurations from the model library.

        Args:
            strict_capability_detection: Whether to enforce strict capability detection

        Returns:
            LLMRegistry instance with default configurations loaded from all provider files
        """
        registry = cls(strict_capability_detection=strict_capability_detection)
        library_path = Path(__file__).parent / "model_library"

        # Load all YAML files from the model library
        yaml_files = list(library_path.glob("*.yaml"))
        if not yaml_files:
            raise ValueError(f"No provider configuration files found in {library_path}")

        # Load each provider file
        for config_file in yaml_files:
            try:
                loaded_registry = cls.load_from_file(config_file, strict_capability_detection)
                # Merge configurations
                for provider in loaded_registry.provider_registry.list_providers():
                    registry.provider_registry.register_provider(provider.name, provider)
                for model in loaded_registry._models.values():
                    registry.register(model)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load {config_file}: {str(e)}")

        return registry

    def to_file(self, path: Union[str, Path]) -> None:
        """Save registry to a JSON or YAML file.

        Args:
            path: Path to save configuration to
        """
        data = {
            "providers": [provider.model_dump() for provider in self.provider_registry.list_providers()],
            "models": [model.model_dump() for model in self._models.values()],
        }

        with open(path, "w") as f:
            if str(path).endswith(".json"):
                json.dump(data, f, indent=2, default=str)
            else:
                yaml.dump(data, f, sort_keys=False)
