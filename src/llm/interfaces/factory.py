"""Factory for creating LLM interfaces."""

from src.core.models import AgentConfig
from src.llm.interfaces.base import BaseLLMInterface
from src.llm.interfaces.provider_interfaces.anthropic import AnthropicLLM
from src.llm.interfaces.provider_interfaces.openai import OpenAIInterface
from src.llm.models import ModelDescriptor, ModelFamily


def create_llm_interface(config: AgentConfig) -> BaseLLMInterface:
    """Create an LLM interface based on the configuration.

    This factory function creates interfaces based on the provider specified in the config.
    For more fine-grained control over model selection, use create_interface_for_model.

    Args:
        config: Agent configuration containing provider and model settings

    Returns:
        An instance of the appropriate LLM interface

    Raises:
        ValueError: If the provider is not supported
    """
    match config.provider.lower():
        case "openai":
            return OpenAIInterface(config)
        case "anthropic":
            return AnthropicLLM(config)
        case _:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")


def create_interface_for_model(model: ModelDescriptor, config: AgentConfig, model_registry=None) -> BaseLLMInterface:
    """Create an appropriate LLM interface based on the model family.

    This factory function creates interfaces based on the model's family, ensuring
    the correct interface is used regardless of how the model is configured.

    Args:
        model: The model descriptor containing model information
        config: The agent configuration including API key and model settings
        model_registry: Optional model registry to use for model validation

    Returns:
        An instance of the appropriate LLM interface

    Raises:
        ValueError: If the model family is not supported
        NotImplementedError: If the model family is recognized but not yet implemented
    """
    if model.family == ModelFamily.CLAUDE:
        return AnthropicLLM(config=config, model_registry=model_registry)
    elif model.family == ModelFamily.GPT:
        return OpenAIInterface(config=config, model_registry=model_registry)
    elif model.family == ModelFamily.HUGGINGFACE:
        raise NotImplementedError("HuggingFace interface not yet implemented")
    else:
        raise ValueError(f"Unsupported model family: {model.family}")
