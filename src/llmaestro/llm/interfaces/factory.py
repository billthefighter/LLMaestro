"""Factory for creating LLM interfaces."""

from typing import Optional

from llmaestro.config.agent import AgentTypeConfig
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.llm.interfaces.provider_interfaces.anthropic import AnthropicLLM
from llmaestro.llm.interfaces.provider_interfaces.gemini import GeminiLLM
from llmaestro.llm.interfaces.provider_interfaces.openai import OpenAIInterface
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.models import LLMProfile, ModelFamily, Provider



async def create_llm_interface(
    config: AgentTypeConfig,
    llm_registry: Optional[LLMRegistry] = None,
) -> BaseLLMInterface:
    """Create an LLM interface based on the provider specified in config.

    Args:
        config: Configuration containing provider and model information
        llm_registry: Optional LLMRegistry instance for model capabilities and provider configs

    Returns:
        BaseLLMInterface: The appropriate interface for the provider

    Raises:
        ValueError: If the provider is not supported or model not found
    """
    # Get model profile and provider config from registry if available
    model_profile = None
    api_config = None
    if llm_registry:
        model_profile = llm_registry.get_model(config.model)
        api_config = llm_registry.get_provider_api_config(config.model)

    # Validate model exists if registry is available
    if llm_registry and not model_profile:
        valid, msg = llm_registry.validate_model(config.model)
        if not valid:
            raise ValueError(f"Invalid model configuration: {msg}")

    # Common initialization parameters
    init_params = {
        "provider": config.provider,
        "model": config.model,
        "api_key": api_config.get("api_key") if api_config else None,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "stream": config.runtime.stream if hasattr(config.runtime, "stream") else True,
    }

    # Add rate limits and context window from provider config or runtime config
    if api_config:
        init_params.update(
            {
                "rate_limit": api_config.get("rate_limits", {}).get("requests_per_minute", 60),
                "max_context_tokens": (
                    model_profile.capabilities.max_context_window
                    if model_profile
                    else config.runtime.max_context_tokens
                    if hasattr(config.runtime, "max_context_tokens")
                    else None
                ),
            }
        )
    else:
        init_params.update(
            {
                "rate_limit": config.runtime.rate_limit if hasattr(config.runtime, "rate_limit") else None,
                "max_context_tokens": config.runtime.max_context_tokens
                if hasattr(config.runtime, "max_context_tokens")
                else None,
            }
        )

    # Create interface based on model family or provider
    if model_profile:
        interface = _create_interface_by_family(model_profile.family, init_params)
    else:
        interface = _create_interface_by_provider(config.provider, init_params)

    # Initialize async components
    await interface.initialize()
    return interface


def _create_interface_by_family(family: ModelFamily, init_params: dict) -> BaseLLMInterface:
    """Create interface based on model family."""
    match family:
        case ModelFamily.CLAUDE:
            return AnthropicLLM(**init_params)
        case ModelFamily.GPT:
            return OpenAIInterface(**init_params)
        case ModelFamily.GEMINI:
            return GeminiLLM(**init_params)
        case _:
            raise ValueError(f"Unsupported model family: {family}")


def _create_interface_by_provider(provider: str, init_params: dict) -> BaseLLMInterface:
    """Create interface based on provider name (fallback)."""
    match provider.lower():
        case "openai":
            return OpenAIInterface(**init_params)
        case "anthropic":
            return AnthropicLLM(**init_params)
        case "google":
            return GeminiLLM(**init_params)
        case _:
            raise ValueError(f"Unsupported provider: {provider}")


async def create_interface_for_model(
    model: LLMProfile,
    config: AgentTypeConfig,
    llm_registry: Optional[LLMRegistry] = None,
) -> BaseLLMInterface:
    """Create an LLM interface for a specific model.

    Args:
        model: Model descriptor
        config: Agent configuration
        llm_registry: Optional model registry for capabilities and API configuration

    Returns:
        BaseLLMInterface: The appropriate interface for the model

    Raises:
        ValueError: If the model family is not supported
    """
    # Get API configuration from registry if available
    api_config = None
    if llm_registry:
        try:
            api_config = llm_registry.get_provider_api_config(model_name=model.name)
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
