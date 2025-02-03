import base64
import json
import mimetypes
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from litellm import acompletion
from pydantic import BaseModel

from src.core.models import AgentConfig, ContextMetrics, TokenUsage
from src.llm.models import ModelDescriptor, ModelFamily, ModelRegistry
from src.llm.rate_limiter import RateLimitConfig, RateLimiter, SQLiteQuotaStorage
from src.llm.token_utils import TokenCounter


@dataclass
class ConversationContext:
    """Represents the current conversation context."""

    messages: List[Dict[str, str]]
    summary: Optional[Dict[str, Any]] = None
    initial_task: Optional[str] = None
    message_count: int = 0


@dataclass
class ImageInput:
    """Represents an image input for LLM processing."""

    content: Union[str, bytes]  # Base64 encoded string or raw bytes
    mime_type: str
    file_name: Optional[str] = None

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "ImageInput":
        """Create an ImageInput from a file path."""
        path = Path(file_path)
        with open(path, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return cls(content=content, mime_type=mime_type, file_name=path.name)

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str, file_name: Optional[str] = None) -> "ImageInput":
        """Create an ImageInput from bytes."""
        content = base64.b64encode(data).decode()
        return cls(content=content, mime_type=mime_type, file_name=file_name)


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
        self._token_counter = TokenCounter()

        # Validate API key
        if not self.config.api_key:
            raise ValueError(f"API key is required for provider {self.config.provider}")

        # Create registry instance
        self._registry = ModelRegistry()

        # Try to get model from registry
        self._model_descriptor = self._registry.get_model(self.config.model_name)
        if not self._model_descriptor:
            raise ValueError(f"Could not find model {self.config.model_name} in registry")

        # Initialize storage and rate limiter
        db_path = os.path.join("data", f"rate_limiter_{self.config.provider}.db")
        os.makedirs("data", exist_ok=True)
        self.storage = SQLiteQuotaStorage(db_path)

        # Initialize rate limiter if enabled
        if self.config.rate_limit.enabled:
            self.rate_limiter = RateLimiter(
                config=RateLimitConfig(
                    requests_per_minute=self.config.rate_limit.requests_per_minute,
                    requests_per_hour=self.config.rate_limit.requests_per_hour,
                    max_daily_tokens=self.config.rate_limit.max_daily_tokens,
                    alert_threshold=self.config.rate_limit.alert_threshold,
                ),
                storage=self.storage,
            )
        else:
            self.rate_limiter = None

    @property
    @abstractmethod
    def model_family(self) -> ModelFamily:
        """Get the model family for this interface."""
        pass

    @property
    def model_descriptor(self) -> Optional[ModelDescriptor]:
        """Get the descriptor for the current model."""
        return self._model_descriptor

    @property
    def capabilities(self):
        """Get capabilities of the current model."""
        return self.model_descriptor.capabilities if self.model_descriptor else None

    @property
    def total_tokens_used(self) -> int:
        """Get total tokens used by this interface."""
        return self._total_tokens

    def set_initial_task(self, task: str) -> None:
        """Set the initial task for the conversation."""
        self.context.initial_task = task

    def estimate_tokens(self, messages: List[Dict[str, str]]) -> Dict[str, int]:
        """Estimate token usage for a list of messages."""
        return self._token_counter.estimate_messages(messages, self.model_family, self.config.model_name)

    def estimate_cost(self, token_usage: TokenUsage) -> float:
        """Estimate cost for token usage."""
        if not self.capabilities:
            return 0.0

        input_cost = token_usage.prompt_tokens * (self.capabilities.input_cost_per_1k_tokens or 0.0) / 1000
        output_cost = token_usage.completion_tokens * (self.capabilities.output_cost_per_1k_tokens or 0.0) / 1000
        return input_cost + output_cost

    async def _check_rate_limits(
        self, messages: List[Dict[str, str]], estimated_tokens: Optional[int] = None
    ) -> tuple[bool, Optional[str]]:
        """Check rate limits before processing a request"""
        if not self.rate_limiter:
            return True, None

        # Use provided token estimate or calculate it
        if estimated_tokens is None:
            estimates = self.estimate_tokens(messages)
            estimated_tokens = estimates["total_tokens"]

        can_proceed, error_msg = await self.rate_limiter.check_and_update(estimated_tokens)
        if not can_proceed:
            # Get current quota status for better error reporting
            status = await self.rate_limiter.get_quota_status()
            return False, f"{error_msg}. Current status: {status}"

        # Check if we're approaching limits
        status = await self.rate_limiter.get_quota_status()
        if status["quota_used_percentage"] >= self.config.rate_limit.alert_threshold * 100:
            print(f"Warning: Approaching rate limits. Current usage: {status['quota_used_percentage']:.1f}%")

        return True, None

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

    async def _maybe_summarize_context(self) -> bool:
        """Summarize the conversation context if needed."""
        if not self.config.summarization.enabled:
            return False

        # Get current token count
        estimates = self.estimate_tokens(self.context.messages)
        current_context_tokens = estimates["total_tokens"]

        # Check if we need to summarize based on token count or message count
        if (
            current_context_tokens < self.config.max_context_tokens
            and len(self.context.messages) < self.config.summarization.preserve_last_n_messages
        ):
            return False

        # Prepare summarization prompt
        summary_prompt = {
            "role": "system",
            "content": (
                "Please provide a brief summary of the conversation so far, "
                "focusing on the key points and decisions made."
            ),
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

    def _update_metrics(self, response: Any) -> Tuple[Optional[TokenUsage], Optional[ContextMetrics]]:
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

                # Calculate context metrics using token counter
                estimates = self.estimate_tokens(self.context.messages)
                total_context_tokens = estimates["total_tokens"]

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

    def _format_messages(
        self, input_data: Any, system_prompt: Optional[str] = None, images: Optional[List[ImageInput]] = None
    ) -> List[Dict[str, Any]]:
        """Format input data and optional system prompt into messages."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, dict):
            messages.append(input_data)
        elif isinstance(input_data, list):
            messages.extend(input_data)

        # Add images if provided
        if images:
            # Convert the last user message to include images
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    messages[i]["images"] = [
                        {"content": img.content, "mime_type": img.mime_type, "file_name": img.file_name}
                        for img in images
                    ]
                    break

        return messages

    async def _handle_response(
        self,
        stream: Any,  # Using Any to handle all litellm response types
        messages: List[Dict[str, str]],
    ) -> LLMResponse:
        """Process the LLM response and update context."""
        try:
            # Collect all chunks from the stream
            content = ""
            last_chunk = None

            # Handle both streaming and non-streaming responses
            if hasattr(stream, "__aiter__"):  # Check for async iterator protocol
                async for chunk in stream:  # type: ignore
                    last_chunk = chunk
                    if hasattr(chunk, "choices") and chunk.choices:
                        delta = getattr(chunk.choices[0], "delta", None)
                        if delta and hasattr(delta, "content"):
                            content += delta.content
            else:  # If it's a single response
                last_chunk = stream
                if hasattr(last_chunk, "choices") and last_chunk.choices:
                    content = getattr(last_chunk.choices[0], "content", "")

            # Get usage information if available
            token_usage = None
            if last_chunk and hasattr(last_chunk, "usage"):
                usage = last_chunk.usage
                token_usage = TokenUsage(
                    completion_tokens=getattr(usage, "completion_tokens", 0),
                    prompt_tokens=getattr(usage, "prompt_tokens", 0),
                    total_tokens=getattr(usage, "total_tokens", 0),
                )
                self._total_tokens += token_usage.total_tokens

            # Calculate context metrics using token counter
            estimates = self.estimate_tokens(self.context.messages)
            total_context_tokens = estimates["total_tokens"]
            context_metrics = ContextMetrics(
                max_context_tokens=self.config.max_context_tokens,
                current_context_tokens=total_context_tokens,
                available_tokens=self.config.max_context_tokens - total_context_tokens,
                context_utilization=total_context_tokens / self.config.max_context_tokens,
            )

            # Update conversation context
            self.context.messages = messages
            self.context.messages.append({"role": "assistant", "content": content})
            self.context.message_count += 1

            # Handle context management
            await self._maybe_summarize_context()
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
    async def process(
        self, input_data: Any, system_prompt: Optional[str] = None, images: Optional[List[ImageInput]] = None
    ) -> LLMResponse:
        """Process input data and return a response."""
        pass

    async def initialize(self) -> None:
        """Initialize async components of the interface."""
        # If model not in registry, detect capabilities and register
        if not self._model_descriptor:
            self._model_descriptor = await self._registry.detect_and_register_model(
                provider=self.config.provider,
                model_name=self.config.model_name,
                api_key=str(self.config.api_key),  # Ensure string type
            )
            if not self._model_descriptor:
                raise ValueError(f"Could not detect capabilities for model {self.config.model_name}")
