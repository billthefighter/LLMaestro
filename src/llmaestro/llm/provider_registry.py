"""Registry for LLM providers and their configurations."""
from typing import Dict, Optional, Set

from pydantic import BaseModel, ConfigDict


class ModelConfig(BaseModel):
    """Configuration for a specific model within a provider."""

    family: str
    context_window: int
    typical_speed: float
    features: Set[str]
    cost: Dict[str, float]

    model_config = ConfigDict(validate_assignment=True)


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    api_base: str
    capabilities_detector: str
    models: Dict[str, ModelConfig]
    rate_limits: Dict[str, int]
    features: Optional[Set[str]] = None  # Provider-wide features

    model_config = ConfigDict(validate_assignment=True)


class ProviderRegistry:
    """Registry of LLM providers and their configurations."""

    def __init__(self):
        self._providers: Dict[str, ProviderConfig] = {}

    def register_provider(self, name: str, config: ProviderConfig) -> None:
        """Register a provider configuration.

        Args:
            name: Provider name (e.g., "anthropic")
            config: Provider configuration
        """
        self._providers[name.lower()] = config

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Get a provider configuration by name."""
        return self._providers.get(name.lower())

    def get_provider_model_config(self, provider: str, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model from a provider.

        Args:
            provider: Provider name
            model_name: Model name

        Returns:
            ModelConfig if found, None otherwise
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

        model_config = provider_config.models.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown model {model_name} for provider {provider}")

        return {
            "provider": provider,
            "name": model_name,
            "api_base": provider_config.api_base,
            "api_key": api_key,
            "capabilities": model_config.model_dump(),
            "rate_limits": provider_config.rate_limits,
        }
