"""Factory for creating LLM interfaces."""

from typing import Optional

from llmaestro.config.agent import AgentTypeConfig
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.llm.interfaces.provider_interfaces.anthropic import AnthropicLLM
from llmaestro.llm.interfaces.provider_interfaces.gemini import GeminiLLM
from llmaestro.llm.interfaces.provider_interfaces.openai import OpenAIInterface
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.models import LLMProfile, ModelFamily
from llmaestro.llm.provider_registry import ProviderRegistry


async def create_llm_interface(
    config: AgentTypeConfig,
    provider_registry: Optional[ProviderRegistry] = None,
    llm_registry: Optional[LLMRegistry] = None,
) -> BaseLLMInterface:
    """Create an LLM interface based on the provider specified in config.

    Args:
        config: Configuration containing provider and model information
        provider_registry: Optional provider registry for API configuration
        llm_registry: Optional model registry for capabilities

    Returns:
        BaseLLMInterface: The appropriate interface for the provider

    Raises:
        ValueError: If the provider is not supported
    """
    # Get API configuration from provider registry if available
    api_config = None
    if provider_registry:
        try:
            api_config = provider_registry.get_provider_api_config(provider=config.provider, model_name=config.model)
        except ValueError:
            pass

    # Get API key from config
    api_key = api_config.get("api_key") if api_config else None
    if not api_key:
        raise ValueError(f"API key not found for provider {config.provider}")

    # Common initialization parameters
    init_params = {
        "provider": config.provider,
        "model": config.model,
        "api_key": api_key,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "rate_limit": config.runtime.rate_limit if hasattr(config.runtime, "rate_limit") else None,
        "max_context_tokens": config.runtime.max_context_tokens
        if hasattr(config.runtime, "max_context_tokens")
        else None,
        "stream": config.runtime.stream if hasattr(config.runtime, "stream") else True,
    }

    # Create interface based on provider
    interface: BaseLLMInterface
    match config.provider.lower():
        case "openai":
            interface = OpenAIInterface(**init_params)
        case "anthropic":
            interface = AnthropicLLM(**init_params)
        case "google":
            interface = GeminiLLM(**init_params)
        case _:
            raise ValueError(f"Unsupported provider: {config.provider}")

    # Initialize async components
    await interface.initialize()
    return interface


async def create_interface_for_model(
    model: LLMProfile,
    config: AgentTypeConfig,
    provider_registry: Optional[ProviderRegistry] = None,
    llm_registry: Optional[LLMRegistry] = None,
) -> BaseLLMInterface:
    """Create an LLM interface for a specific model.

    Args:
        model: Model descriptor
        config: Agent configuration
        provider_registry: Optional provider registry for API configuration
        llm_registry: Optional model registry for capabilities

    Returns:
        BaseLLMInterface: The appropriate interface for the model

    Raises:
        ValueError: If the model family is not supported
    """
    # Get API configuration from provider registry if available
    api_config = None
    if provider_registry:
        try:
            api_config = provider_registry.get_provider_api_config(provider=config.provider, model_name=model.name)
        except ValueError:
            pass

    # Get API key from config
    api_key = api_config.get("api_key") if api_config else None
    if not api_key:
        raise ValueError(f"API key not found for provider {config.provider}")

    # Common initialization parameters
    init_params = {
        "provider": config.provider,
        "model": model.name,
        "api_key": api_key,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "rate_limit": config.runtime.rate_limit if hasattr(config.runtime, "rate_limit") else None,
        "max_context_tokens": config.runtime.max_context_tokens
        if hasattr(config.runtime, "max_context_tokens")
        else None,
        "stream": config.runtime.stream if hasattr(config.runtime, "stream") else True,
    }

    # Create interface based on model family
    interface: BaseLLMInterface
    if model.family == ModelFamily.CLAUDE:
        interface = AnthropicLLM(**init_params)
    elif model.family == ModelFamily.GPT:
        interface = OpenAIInterface(**init_params)
    elif model.family == ModelFamily.GEMINI:
        interface = GeminiLLM(**init_params)
    else:
        raise ValueError(f"Unsupported model family: {model.family}")

    # Initialize async components
    await interface.initialize()
    return interface
