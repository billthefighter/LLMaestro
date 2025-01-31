from typing import Any, Optional

from litellm import acompletion

from src.llm.models import ModelFamily

from .base import BaseLLMInterface, LLMResponse


class OpenAIInterface(BaseLLMInterface):
    """OpenAI-specific implementation of the LLM interface."""

    @property
    def model_family(self) -> ModelFamily:
        """Get the model family for this interface."""
        return ModelFamily.GPT

    async def process(self, input_data: Any, system_prompt: Optional[str] = None) -> LLMResponse:
        """Process input using OpenAI's API via LiteLLM."""
        try:
            messages = self._format_messages(input_data, system_prompt)

            # Check rate limits
            can_proceed, error_msg = await self._check_rate_limits(messages)
            if not can_proceed:
                return LLMResponse(
                    content=f"Rate limit exceeded: {error_msg}", metadata={"error": "rate_limit_exceeded"}
                )

            # Check if model supports streaming
            supports_streaming = self.capabilities and self.capabilities.supports_streaming

            # Make API call with appropriate streaming setting
            response = await acompletion(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stream=supports_streaming,
            )

            return await self._handle_response(response, messages)

        except Exception as e:
            return self._handle_error(e)
