from typing import Any, Dict, List, Optional, cast

from anthropic import Anthropic
from anthropic.types import Message

from src.core.models import TokenUsage
from src.llm.models import ModelFamily

from .base import BaseLLMInterface, LLMResponse


class AnthropicLLM(BaseLLMInterface):
    """Anthropic Claude LLM implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = Anthropic(api_key=self.config.api_key)

    @property
    def model_family(self) -> ModelFamily:
        """Get the model family for this interface."""
        return ModelFamily.CLAUDE

    def estimate_tokens(self, messages: List[Dict[str, str]]) -> Dict[str, int]:
        """Estimate token usage for a list of messages."""
        total_tokens = 0
        for msg in messages:
            total_tokens += len(msg["content"].split())  # Simple word count approximation
        return {
            "prompt_tokens": total_tokens,
            "completion_tokens": self.config.max_tokens,
            "total_tokens": total_tokens + self.config.max_tokens,
        }

    async def process(self, input_data: Any, system_prompt: Optional[str] = None) -> LLMResponse:
        """Process input data through Claude and return a standardized response."""
        try:
            messages = self._format_messages(input_data, system_prompt)

            # Check rate limits
            can_proceed, error_msg = await self._check_rate_limits(messages)
            if not can_proceed:
                return LLMResponse(
                    content=f"Rate limit exceeded: {error_msg}", metadata={"error": "rate_limit_exceeded"}
                )

            # Convert messages to Anthropic format
            anthropic_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    anthropic_messages.append({"role": "assistant", "content": msg["content"]})
                else:
                    anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

            # Make API call
            response = self.client.messages.create(
                model=self.config.model_name,
                messages=anthropic_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            # Extract text from the response
            response_message = cast(Message, response)
            content = str(response_message.content[0])

            # Estimate token usage
            token_usage = TokenUsage(
                prompt_tokens=len(" ".join(msg["content"] for msg in messages).split()),  # Simple word count
                completion_tokens=len(content.split()),  # Simple word count
                total_tokens=len(" ".join(msg["content"] for msg in messages).split()) + len(content.split()),
            )

            return LLMResponse(content=content, metadata={"id": response.id}, token_usage=token_usage)

        except Exception as e:
            return self._handle_error(e)
