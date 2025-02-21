"""Registry for managing LLM models."""
import asyncio
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
    auto_update_capabilities: bool = True
    _models: Dict[str, LLMProfile] = field(default_factory=dict)

    @classmethod
    def create_default(
        cls,
        strict_capability_detection: bool = False,
        auto_update_capabilities: bool = True,
        api_keys: Optional[Dict[str, str]] = None,
    ) -> "LLMRegistry":
        """Create a LLMRegistry with default configurations from the model library.
        This is the primary way to instantiate an LLMRegistry.

        Args:
            strict_capability_detection: Whether to enforce strict capability detection
            auto_update_capabilities: Whether to automatically detect and update capabilities
            api_keys: Dictionary of provider API keys for capability detection

        Returns:
            LLMRegistry instance with default configurations loaded
        """
        registry = cls(
            strict_capability_detection=strict_capability_detection, auto_update_capabilities=auto_update_capabilities
        )
        library_path = Path(__file__).parent / "model_library"

        # Load all YAML files from the model library
        yaml_files = list(library_path.glob("*.yaml"))
        if not yaml_files:
            raise ValueError(f"No provider configuration files found in {library_path}")

        # Load each provider file
        for config_file in yaml_files:
            try:
                registry._load_provider_file(config_file)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load {config_file}: {str(e)}")

        # Update capabilities if auto-update is enabled and API keys are provided
        if auto_update_capabilities and api_keys:
            asyncio.run(registry._update_all_capabilities(api_keys))

        return registry

    def _load_provider_file(self, path: Union[str, Path]) -> None:
        """Load a single provider configuration file.

        Args:
            path: Path to provider YAML file
        """
        with open(path) as f:
            data = yaml.safe_load(f)

            # Register provider
            provider = Provider(**data["provider"])
            self.provider_registry.register_provider(provider.name, provider)

            # Register models
            for model_data in data["models"]:
                self.register(LLMProfile(**model_data))

    def register(self, descriptor: LLMProfile) -> None:
        """Register a model descriptor."""
        self._models[descriptor.name] = descriptor

    async def _update_all_capabilities(self, api_keys: Dict[str, str]) -> None:
        """Update capabilities for all models where possible.

        Args:
            api_keys: Dictionary of provider API keys for capability detection
        """
        logger = logging.getLogger(__name__)
        update_tasks = []

        for model in list(self._models.values()):
            provider = model.capabilities.family.value
            if provider not in api_keys:
                logger.warning(f"No API key provided for provider {provider}, skipping capability update")
                continue

            update_tasks.append(
                self._update_model_capabilities(
                    provider=provider, model_name=model.name, api_key=api_keys[provider], current_profile=model
                )
            )

        if update_tasks:
            # Run all updates concurrently
            await asyncio.gather(*update_tasks)

    async def _update_model_capabilities(
        self, provider: str, model_name: str, api_key: str, current_profile: LLMProfile
    ) -> None:
        """Update capabilities for a single model.

        Args:
            provider: The provider name
            model_name: The model name
            api_key: API key for the provider
            current_profile: Current model profile to update
        """
        logger = logging.getLogger(__name__)

        try:
            # Detect current capabilities
            capabilities = await ModelCapabilityDetectorFactory.detect_capabilities(provider, model_name, api_key)

            # Update only if capabilities differ from current
            if self._capabilities_need_update(current_profile.capabilities, capabilities):
                logger.info(f"Updating capabilities for {model_name}")
                # Preserve metadata while updating capabilities
                updated_profile = LLMProfile(
                    capabilities=capabilities,
                    metadata=current_profile.metadata,
                    vision_capabilities=current_profile.vision_capabilities,
                )
                self.register(updated_profile)

        except Exception as e:
            logger.warning(f"Failed to update capabilities for {model_name}: {str(e)}")

    def _capabilities_need_update(self, current: LLMCapabilities, detected: LLMCapabilities) -> bool:
        """Check if capabilities need to be updated.

        Args:
            current: Current capabilities
            detected: Newly detected capabilities

        Returns:
            True if capabilities need update, False otherwise
        """
        # Compare relevant fields that might change
        fields_to_compare = [
            "max_context_window",
            "max_output_tokens",
            "typical_speed",
            "input_cost_per_1k_tokens",
            "output_cost_per_1k_tokens",
            "supports_streaming",
            "supports_vision",
            "supports_json_mode",
            "supports_tools",
            "supports_function_calling",
        ]

        return any(
            getattr(current, field) != getattr(detected, field)
            for field in fields_to_compare
            if hasattr(current, field) and hasattr(detected, field)
        )

    async def detect_and_register_model(self, provider: str, model_name: str, api_key: str) -> LLMProfile:
        """Detects capabilities of a model and registers it in the registry.
        Used for models not in the YAML configurations or when forcing capability update.

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

        try:
            capabilities = await ModelCapabilityDetectorFactory.detect_capabilities(provider, model_name, api_key)
            capabilities.name = model_name
            capabilities.family = ModelFamily(provider)
        except (ValueError, RuntimeError) as e:
            if self.strict_capability_detection:
                raise ValueError(f"Failed to detect capabilities for {model_name}: {str(e)}") from e
            logger.warning(f"Capability detection failed for {model_name}, using default capabilities. Error: {str(e)}")
            capabilities = LLMCapabilities(
                name=model_name,
                family=ModelFamily(provider),
                max_context_window=4096,
                typical_speed=50.0,
                input_cost_per_1k_tokens=0.01,
                output_cost_per_1k_tokens=0.02,
                supports_streaming=True,
            )

        descriptor = LLMProfile(
            capabilities=capabilities,
            metadata=LLMMetadata(
                release_date=datetime.now(),
                min_api_version="2024-02-29",
            ),
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

    def to_provider_files(self, directory: Union[str, Path]) -> None:
        """Save registry to individual YAML files for each provider.

        Args:
            directory: Directory to save provider YAML files to
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Group models by provider
        provider_models: Dict[str, List[LLMProfile]] = {}
        for model in self._models.values():
            provider = model.capabilities.family.value
            if provider not in provider_models:
                provider_models[provider] = []
            provider_models[provider].append(model)

        # Save each provider to its own file
        for provider_name, models in provider_models.items():
            provider = self.provider_registry.get_provider(provider_name)
            if not provider:
                continue

            data = {"provider": provider.model_dump(), "models": [model.model_dump() for model in models]}

            file_path = directory / f"{provider_name}.yaml"
            with open(file_path, "w") as f:
                yaml.dump(data, f, sort_keys=False)
