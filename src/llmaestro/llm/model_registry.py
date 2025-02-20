"""Registry for managing LLM models."""
from typing import Dict, Optional, Set

from .provider_registry import ProviderRegistry


class ModelRegistry:
    """Registry for managing LLM models and their configurations."""

    def __init__(self, provider_registry: ProviderRegistry):
        """Initialize the model registry.

        Args:
            provider_registry: Registry of LLM providers
        """
        self._provider_registry = provider_registry
        self._registered_models: Dict[str, Dict[str, dict]] = {}

    def register_from_provider(self, provider: str, model_name: str) -> None:
        """Register a model from a provider.

        Args:
            provider: Provider name
            model_name: Model name

        Raises:
            ValueError: If provider or model not found
        """
        model_config = self._provider_registry.get_provider_model_config(provider, model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not found for provider {provider}")

        if provider not in self._registered_models:
            self._registered_models[provider] = {}
        self._registered_models[provider][model_name] = model_config.model_dump()

    def get_model_config(self, provider: str, model_name: str) -> Optional[dict]:
        """Get configuration for a specific model.

        Args:
            provider: Provider name
            model_name: Model name

        Returns:
            Model configuration if found, None otherwise
        """
        return self._registered_models.get(provider, {}).get(model_name)

    def get_registered_models(self, provider: Optional[str] = None) -> Dict[str, Set[str]]:
        """Get all registered models, optionally filtered by provider.

        Args:
            provider: Optional provider name to filter by

        Returns:
            Dictionary mapping provider names to sets of model names
        """
        if provider:
            return {provider: set(self._registered_models.get(provider, {}).keys())}
        return {p: set(models.keys()) for p, models in self._registered_models.items()}
