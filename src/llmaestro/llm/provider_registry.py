"""Registry for LLM providers and their configurations."""
from typing import Dict, List, Optional

from .models import LLMProfile, Provider


class ProviderRegistry:
    """Registry of LLM providers and their configurations."""

    def __init__(self):
        self._providers: Dict[str, Provider] = {}

    def register_provider(self, name: str, config: Provider) -> None:
        """Register a provider configuration.

        Args:
            name: Provider name (e.g., "anthropic")
            config: Provider configuration
        """
        self._providers[name.lower()] = config

    def get_provider(self, name: str) -> Optional[Provider]:
        """Get a provider configuration by name."""
        return self._providers.get(name.lower())

    def list_providers(self) -> List[Provider]:
        """Get a list of all registered providers."""
        return list(self._providers.values())

    def get_provider_model_config(self, provider: str, model_name: str) -> Optional[LLMProfile]:
        """Get a model configuration from a provider.

        Args:
            provider: Provider name
            model_name: Model name

        Returns:
            LLMProfile if found, None otherwise
        """
        provider_config = self.get_provider(provider)
        if not provider_config:
            return None
        return provider_config.models.get(model_name)

    def get_model(self, provider: str, model_name: str) -> Optional[LLMProfile]:
        """Get a model descriptor from a provider.

        Args:
            provider: Provider name
            model_name: Model name

        Returns:
            LLMProfile if found, None otherwise
        """
        provider_config = self.get_provider(provider)
        if not provider_config:
            return None
        return provider_config.models.get(model_name)

    def get_provider_api_config(self, provider: str, model_name: str, api_key: Optional[str] = None) -> dict:
        """Get the complete API configuration for a provider and model.

        Args:
            provider: Provider name
            model_name: Model name
            api_key: Optional API key to include in config

        Returns:
            Complete provider API configuration

        Raises:
            ValueError: If provider or model not found
        """
        provider_config = self.get_provider(provider)
        if not provider_config:
            raise ValueError(f"Unknown provider: {provider}")

        model = provider_config.models.get(model_name)
        if not model:
            raise ValueError(f"Unknown model {model_name} for provider {provider}")

        return {
            "provider": provider,
            "name": model_name,
            "api_base": provider_config.api_base,
            "api_key": api_key,
            "capabilities": model.capabilities.model_dump(),
            "rate_limits": provider_config.rate_limits,
        }

    @classmethod
    def create_default(cls) -> "ProviderRegistry":
        """Create a ProviderRegistry with default configurations loaded from the model library.

        Returns:
            ProviderRegistry instance with default configurations loaded
        """
        from .llm_registry import LLMRegistry

        registry = LLMRegistry.create_default()
        return registry.provider_registry
