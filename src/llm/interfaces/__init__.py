from .base import BaseLLMInterface, LLMResponse, ConversationContext
from .openai import OpenAIInterface
from .anthropic import AnthropicLLM

from src.core.models import AgentConfig

def create_llm_interface(config: AgentConfig) -> BaseLLMInterface:
    """Create an LLM interface based on the configuration."""
    match config.provider.lower():
        case "openai":
            return OpenAIInterface(config)
        case "anthropic":
            return AnthropicLLM(config)
        case _:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

__all__ = [
    'BaseLLMInterface',
    'LLMResponse',
    'ConversationContext',
    'OpenAIInterface',
    'AnthropicLLM',
    'create_llm_interface'
] 