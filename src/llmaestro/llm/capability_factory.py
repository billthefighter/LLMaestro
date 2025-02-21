"""Factory for creating model capability detectors."""
from typing import Dict, Type

from .capability_detector import AnthropicCapabilityDetector, BaseCapabilityDetector, OpenAICapabilityDetector
from .models import LLMCapabilities


class ModelCapabilityDetectorFactory:
    """Factory for creating capability detectors."""

    _detectors: Dict[str, Type[BaseCapabilityDetector]] = {
        "anthropic": AnthropicCapabilityDetector,
        "openai": OpenAICapabilityDetector,
    }

    @classmethod
    def get_detector(cls, provider: str) -> Type[BaseCapabilityDetector]:
        """Get the appropriate detector for a provider."""
        detector = cls._detectors.get(provider.lower())
        if not detector:
            raise ValueError(f"Unsupported provider for capability detection: {provider}")
        return detector

    @classmethod
    def register_detector(cls, provider: str, detector: Type[BaseCapabilityDetector]) -> None:
        """Register a new detector for a provider."""
        cls._detectors[provider.lower()] = detector

    @classmethod
    async def detect_capabilities(cls, provider: str, model_name: str, api_key: str) -> LLMCapabilities:
        """Detect capabilities for a model.

        Args:
            provider: The LLM provider (e.g., "anthropic", "openai")
            model_name: Name of the model to detect capabilities for
            api_key: API key for authentication

        Returns:
            LLMCapabilities for the model
        """
        detector = cls.get_detector(provider)
        return await detector.detect_capabilities(model_name, api_key)
