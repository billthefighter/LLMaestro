import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from litellm import acompletion
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

from src.core.models import AgentConfig, ContextMetrics, TokenUsage


@dataclass
class ConversationContext:
    """Represents the current conversation context."""

    messages: List[Dict[str, str]]
    summary: Optional[Dict[str, Any]] = None
    initial_task: Optional[str] = None
    message_count: int = 0


class LLMResponse(BaseModel):
    """Standardized response from LLM processing."""

    content: str
    metadata: Dict[str, Any] = {}
    token_usage: Optional[TokenUsage] = None
    context_metrics: Optional[ContextMetrics] = None

    def model_dump(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "token_usage": self.token_usage.model_dump() if self.token_usage else None,
            "context_metrics": self.context_metrics.model_dump() if self.context_metrics else None,
        }

    def model_dump_json(self) -> str:
        return json.dumps(self.model_dump())


class BaseLLMInterface(ABC):
    """Base interface for LLM interactions."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.context = ConversationContext([])
        self._total_tokens = 0

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens

    def set_initial_task(self, task: str) -> None:
        """Set the initial task for the conversation."""
        self.context.initial_task = task

    async def _maybe_add_reminder(self) -> bool:
        """Add a reminder of the initial task if needed."""
        if not self.context.initial_task:
            return False

        # Add reminder every N messages
        if (
            self.context.message_count > 0
            and self.config.summarization.reminder_frequency > 0
            and self.context.message_count % self.config.summarization.reminder_frequency == 0
        ):
            reminder_message = {
                "role": "system",
                "content": self.config.summarization.reminder_template.format(task=self.context.initial_task),
            }
            self.context.messages.append(reminder_message)
            return True
        return False

    async def _maybe_summarize_context(self, current_metrics: ContextMetrics) -> bool:
        """Summarize the conversation context if needed."""
        if not self.config.summarization.enabled:
            return False

        # Check if we need to summarize based on token count or message count
        if (
            current_metrics.current_context_tokens < self.config.max_context_tokens
            and len(self.context.messages) < self.config.summarization.preserve_last_n_messages
        ):
            return False

        # Prepare summarization prompt
        summary_prompt = {
            "role": "system",
            "content": "Please provide a brief summary of the conversation so far, focusing on the key points and decisions made.",
        }

        # Get summary from LLM
        messages_for_summary = [
            msg
            for msg in self.context.messages
            if msg.get("role") != "system" or "Remember, your initial task was" not in msg.get("content", "")
        ]

        summary_messages = [summary_prompt] + messages_for_summary[
            -self.config.summarization.preserve_last_n_messages :
        ]

        try:
            stream = await acompletion(
                model=self.config.model_name,
                messages=summary_messages,
                max_tokens=self.config.max_tokens,
                stream=True,
            )

            # Collect all chunks from the stream
            content = ""
            async for chunk in stream:
                if hasattr(chunk, "choices") and chunk.choices and hasattr(chunk.choices[0], "delta"):
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        content += delta.content

            # Update context with summary
            self.context.summary = {"content": content, "message_count": len(self.context.messages)}

            # Clear old messages except system prompts and recent ones
            system_messages = [msg for msg in self.context.messages if msg["role"] == "system"]
            recent_messages = self.context.messages[-self.config.summarization.preserve_last_n_messages :]

            summary_message = {
                "role": "system",
                "content": f"Previous conversation summary: {content}",
            }

            self.context.messages = system_messages + [summary_message] + recent_messages
            return True

        except Exception as e:
            print(f"Failed to generate summary: {str(e)}")
            return False

    def _update_metrics(self, response: ModelResponse) -> Tuple[Optional[TokenUsage], Optional[ContextMetrics]]:
        """Update token usage and context metrics."""
        try:
            if hasattr(response, "usage"):
                usage = response.usage
                token_usage = TokenUsage(
                    completion_tokens=usage.completion_tokens,
                    prompt_tokens=usage.prompt_tokens,
                    total_tokens=usage.total_tokens,
                )
                self._total_tokens += token_usage.total_tokens

                # Calculate context metrics
                total_context_tokens = sum(len(msg.get("content", "")) for msg in self.context.messages)

                context_metrics = ContextMetrics(
                    max_context_tokens=self.config.max_context_tokens,
                    current_context_tokens=total_context_tokens,
                    available_tokens=self.config.max_context_tokens - total_context_tokens,
                    context_utilization=total_context_tokens / self.config.max_context_tokens,
                )

                return token_usage, context_metrics
            return None, None
        except Exception as e:
            print(f"Failed to update metrics: {str(e)}")
            return None, None

    def _format_messages(self, input_data: Any, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Format input data and context into messages for the LLM."""
        messages = self.context.messages.copy()

        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, dict) and "role" in input_data and "content" in input_data:
            messages.append(input_data)
        elif isinstance(input_data, list):
            messages.extend(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        return messages

    async def _handle_response(
        self, stream: AsyncGenerator[ModelResponse, None], messages: List[Dict[str, str]]
    ) -> LLMResponse:
        """Process the LLM response and update context."""
        try:
            # Collect all chunks from the stream
            content = ""
            async for chunk in stream:
                if hasattr(chunk, "choices") and chunk.choices and hasattr(chunk.choices[0], "delta"):
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        content += delta.content

            # Get the last chunk for usage information
            token_usage, context_metrics = self._update_metrics(last_chunk) if last_chunk else (None, None)

            # Update conversation context
            self.context.messages = messages
            self.context.messages.append({"role": "assistant", "content": content})
            self.context.message_count += 1

            # Handle context management
            if context_metrics:
                await self._maybe_summarize_context(context_metrics)
            await self._maybe_add_reminder()

            return LLMResponse(content=content, token_usage=token_usage, context_metrics=context_metrics)

        except Exception as e:
            return self._handle_error(e)

    def _handle_error(self, e: Exception) -> LLMResponse:
        """Handle errors in LLM processing."""
        error_message = f"Error processing LLM request: {str(e)}"
        print(error_message)  # Log the error
        return LLMResponse(content=error_message, metadata={"error": str(e)})

    @abstractmethod
    async def process(self, input_data: Any, system_prompt: Optional[str] = None) -> LLMResponse:
        """Process input data and return a response."""
        pass
