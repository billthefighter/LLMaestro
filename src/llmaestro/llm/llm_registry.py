"""Registry for LLM providers and models."""
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict


class ModelConfig(BaseModel):
    """Configuration for a specific model."""

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


class ModelRegistry:
    """Registry of available LLM models and their capabilities."""

    def __init__(self, providers: Dict[str, ProviderConfig]):
        self._providers = providers
        self._active_models: Dict[str, ModelConfig] = {}

    def register_model(self, provider: str, model_name: str) -> None:
        """Register a model from the provider configuration.

        Args:
            provider: Provider name (e.g., "anthropic")
            model_name: Model name to register

        Raises:
            ValueError: If provider or model not found
        """
        provider_config = self._providers.get(provider)
        if not provider_config:
            raise ValueError(f"Unknown provider: {provider}")

        model_config = provider_config.models.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown model {model_name} for provider {provider}")

        self._active_models[model_name] = model_config

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get a model configuration by name."""
        return self._active_models.get(name)

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Get a provider configuration by name."""
        return self._providers.get(name.lower())

    def get_models_by_feature(self, feature: str) -> List[str]:
        """Get all models that support a specific feature."""
        return [name for name, config in self._active_models.items() if feature in config.features]

    def get_models_by_cost(self, max_cost_per_1k: float) -> List[str]:
        """Get all models within a cost threshold."""
        return [name for name, config in self._active_models.items() if config.cost["input_per_1k"] <= max_cost_per_1k]

    def validate_model(self, name: str) -> tuple[bool, Optional[str]]:
        """Validate if a model exists and is registered."""
        if name not in self._active_models:
            return False, f"Model {name} not registered"
        return True, None

    def get_model_config(self, provider: str, model_name: str, api_key: Optional[str] = None) -> dict:
        """Get the complete configuration for a model.

        Args:
            provider: Provider name
            model_name: Model name
            api_key: Optional API key to include in config

        Returns:
            Complete model configuration including provider settings

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
