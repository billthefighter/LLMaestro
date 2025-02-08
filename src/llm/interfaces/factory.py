"""Factory for creating LLM interfaces."""

from typing import Optional

from src.core.models import AgentConfig
from src.llm.interfaces.base import BaseLLMInterface
from src.llm.interfaces.provider_interfaces.anthropic import AnthropicLLM
from src.llm.interfaces.provider_interfaces.gemini import GeminiLLM
from src.llm.interfaces.provider_interfaces.openai import OpenAIInterface
from src.llm.models import ModelDescriptor, ModelFamily, ModelRegistry


def create_llm_interface(config: AgentConfig) -> BaseLLMInterface:
    """Create an LLM interface based on the provider specified in config.

    Args:
        config: Configuration containing provider and model information

    Returns:
        BaseLLMInterface: The appropriate interface for the provider

    Raises:
        ValueError: If the provider is not supported
    """
    match config.provider.lower():
        case "openai":
            return OpenAIInterface(config)
        case "anthropic":
            return AnthropicLLM(config)
        case "google":
            return GeminiLLM(config)
        case _:
            raise ValueError(f"Unsupported provider: {config.provider}")


def create_interface_for_model(
    model: ModelDescriptor,
    config: AgentConfig,
    registry: Optional[ModelRegistry] = None,
) -> BaseLLMInterface:
    """Create an LLM interface for a specific model.

    Args:
        model: Model descriptor
        config: Agent configuration
        registry: Optional model registry

    Returns:
        BaseLLMInterface: The appropriate interface for the model

    Raises:
        ValueError: If the model family is not supported
    """
    if model.family == ModelFamily.CLAUDE:
        return AnthropicLLM(config, registry)
    elif model.family == ModelFamily.GPT:
        return OpenAIInterface(config, registry)
    elif model.family == ModelFamily.GEMINI:
        return GeminiLLM(config, registry)
    else:
        raise ValueError(f"Unsupported model family: {model.family}")
