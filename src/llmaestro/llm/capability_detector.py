"""Capability detection for LLM models."""

from .capabilities import LLMCapabilities, RangeConfig
from .enums import ModelFamily


class BaseCapabilityDetector:
    """Base class for model capability detection."""

    @classmethod
    async def detect_capabilities(cls, model_name: str, api_key: str) -> LLMCapabilities:
        """Detect capabilities for a model."""
        raise NotImplementedError


class AnthropicCapabilityDetector(BaseCapabilityDetector):
    """Capability detector for Anthropic models."""

    @classmethod
    async def detect_capabilities(cls, model_name: str, api_key: str) -> LLMCapabilities:
        """Detect capabilities for an Anthropic model."""
        is_claude3 = "claude-3" in model_name.lower()

        return LLMCapabilities(
            name=model_name,
            family=ModelFamily.CLAUDE,
            max_context_window=200000 if is_claude3 else 100000,
            max_output_tokens=4096,
            typical_speed=150.0,  # Tokens per second estimate
            input_cost_per_1k_tokens=0.01,
            output_cost_per_1k_tokens=0.03,
            # Core Features
            supports_streaming=True,
            supports_vision=is_claude3,
            supports_embeddings=False,
            supports_json_mode=is_claude3,
            supports_system_prompt=True,
            supports_tools=is_claude3,
            supports_parallel_requests=True,
            supports_function_calling=is_claude3,
            # Advanced Features
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            supports_stop_sequences=True,
            supports_message_role=True,
            # Quality Settings
            temperature=RangeConfig(min_value=0.0, max_value=1.0, default_value=0.7),
            top_p=RangeConfig(min_value=0.0, max_value=1.0, default_value=1.0),
        )


class OpenAICapabilityDetector(BaseCapabilityDetector):
    """Capability detector for OpenAI models."""

    @classmethod
    async def detect_capabilities(cls, model_name: str, api_key: str) -> LLMCapabilities:
        """Detect capabilities for an OpenAI model."""
        from openai import AsyncOpenAI

        try:
            client = AsyncOpenAI(api_key=api_key)
            model = await client.models.retrieve(model_name)

            is_gpt4 = "gpt-4" in model_name.lower()
            is_vision = "vision" in model_name.lower()

            return LLMCapabilities(
                name=model_name,
                family=ModelFamily.GPT,
                max_context_window=getattr(model, "context_window", 32000),
                max_output_tokens=4096,
                typical_speed=150.0,  # Tokens per second estimate
                input_cost_per_1k_tokens=0.01 if is_gpt4 else 0.0015,
                output_cost_per_1k_tokens=0.03 if is_gpt4 else 0.002,
                # Core Features
                supports_streaming=True,
                supports_vision=is_vision,
                supports_embeddings=False,
                supports_json_mode=True,
                supports_system_prompt=True,
                supports_tools=True,
                supports_parallel_requests=True,
                supports_function_calling=True,
                # Advanced Features
                supports_frequency_penalty=True,
                supports_presence_penalty=True,
                supports_stop_sequences=True,
                supports_message_role=True,
                # Quality Settings
                temperature=RangeConfig(min_value=0.0, max_value=2.0, default_value=0.7),
                top_p=RangeConfig(min_value=0.0, max_value=1.0, default_value=1.0),
            )
        except Exception as e:
            raise RuntimeError("Failed to detect OpenAI capabilities") from e
