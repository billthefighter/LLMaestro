"""Base interfaces for LLM providers."""

import base64
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from litellm import acompletion
from pydantic import BaseModel, Field, ConfigDict

from llmaestro.core.models import (
    ContextMetrics,
    LLMResponse,  # Updated to import from core.models
    TokenUsage,
)
from llmaestro.llm.enums import MediaType
from llmaestro.llm.rate_limiter import RateLimiter
from llmaestro.llm.token_utils import TokenCounter
from llmaestro.prompts.base import BasePrompt

from llmaestro.core.conversations import ConversationContext
from llmaestro.llm.enums import ModelFamily

import logging
logger = logging.getLogger(__name__)




@dataclass
class ImageInput:
    """Input image data for vision models."""

    content: Union[str, bytes]  # Base64 encoded string or raw bytes
    media_type: MediaType  # Media type of the image
    file_name: Optional[str] = None

    def __init__(
        self,
        content: Union[str, bytes],
        media_type: Union[str, MediaType] = MediaType.JPEG,
        file_name: Optional[str] = None,
    ):
        self.content = content
        self.media_type = media_type if isinstance(media_type, MediaType) else MediaType.from_mime_type(media_type)
        self.file_name = file_name

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "ImageInput":
        """Create an ImageInput from a file path."""
        path = Path(file_path)
        with open(path, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        media_type = MediaType.from_file_extension(str(path))  # Convert Path to str
        return cls(content=content, media_type=media_type, file_name=path.name)

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: Union[str, MediaType], file_name: Optional[str] = None) -> "ImageInput":
        """Create an ImageInput from bytes."""
        content = base64.b64encode(data).decode()
        media_type = mime_type if isinstance(mime_type, MediaType) else MediaType.from_mime_type(mime_type)
        return cls(content=content, media_type=media_type, file_name=file_name)


class BaseLLMInterface(BaseModel, ABC):
    """Base interface for LLM interactions."""

    # Default supported media types (can be overridden by specific implementations)
    SUPPORTED_MEDIA_TYPES: Set[MediaType] = {MediaType.JPEG, MediaType.PNG, MediaType.PDF}

    # Pydantic fields
    context: ConversationContext = Field(default_factory=lambda: ConversationContext(), description="Current conversation context")
    _total_tokens: int = Field(default=0, description="Total tokens used by this interface")
    token_counter: Optional[TokenCounter] = Field(default=None, description="Token counter instance")
    rate_limiter: Optional[RateLimiter] = Field(default=None, description="Rate limiter instance")
    ignore_missing_credentials: bool = Field(default=False, description="Ignore missing credentials")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Initialize the interface after Pydantic model validation."""

        # Initialize token counter with registry
        self.token_counter = TokenCounter(
            api_key=self.state.credentials,
            llm_registry=self.llm_registry
        )

        # Set up rate limiter if configured
        if self.state.runtime_config.rate_limit:
            self.rate_limiter = RateLimiter(
                config=self.state.runtime_config.rate_limit
            )
    @abstractproperty
    def model_family(self) -> ModelFamily:
        """Get the model family for the LLM provider."""
        pass

    @property
    def total_tokens_used(self) -> int:
        """Get total tokens used by this interface."""
        return self._total_tokens

    def set_initial_task(self, task: str) -> None:
        """Set the initial task for the conversation."""
        self.context.initial_task = task

    def estimate_tokens(self, messages: List[Dict[str, str]]) -> Dict[str, int]:
        """Estimate token usage for a list of messages."""
        return self.token_counter.estimate_messages(
            messages,
            self.state.model_family,
            str(self.state.profile)
        )

    def validate_credentials(self) -> None:
        """Validate the credentials for the LLM provider."""
        if not self.apikey and not self.ignore_missing_credentials:
            raise ValueError(f"API key is required for provider {self.apikey.provider}")
        elif not self.apikey and self.ignore_missing_credentials:
            logger.warning("WARNING: No API key provided for provider {self.apikey.provider}. Some functionality may be limited.")

    def estimate_cost(self, token_usage: TokenUsage) -> float:
        """Estimate cost for token usage.
        
        Args:
            token_usage: Token usage information
            
        Returns:
            Estimated cost in USD
        """
        if not self.capabilities:
            raise ValueError(f"Capabilities not found for model {self.state.profile}, cannot return estimated cost.")

        token_counts = {
            "prompt_tokens": token_usage.prompt_tokens,
            "completion_tokens": token_usage.completion_tokens,
            "total_tokens": token_usage.total_tokens
        }
        
        # Use token counter's implementation
        return self.token_counter.estimate_cost(
            token_counts=token_counts,
            image_count=0,  # No images in token usage
            model_family=self.state.model_family,
            model_name=str(self.state.profile)
        )

    async def _check_rate_limits(
        self, messages: List[Dict[str, Any]], estimated_tokens: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """Check if the request would exceed rate limits."""
        if not self.rate_limiter:
            return True, None

        return await self.rate_limiter.check_and_update(estimated_tokens or 0)

    async def _maybe_add_reminder(self) -> bool:
        """Add a reminder of the initial task if needed."""
        if not self.context.initial_task:
            return False

        # Add reminder every N messages based on config
        if self.context.message_count > 0:
            reminder_message = {
                "role": "system",
                "content": f"Remember, your initial task was: {self.context.initial_task}",
            }
            self.context.messages.append(reminder_message)
            return True
        return False

    async def _maybe_summarize_context(self) -> bool:
        """Summarize the conversation context if needed."""
        # Get current token count
        estimates = self.estimate_tokens(self.context.messages)
        current_context_tokens = estimates["total_tokens"]

        # Check if we need to summarize based on token count
        if current_context_tokens < self.state.runtime_config.max_context_tokens:
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

        summary_messages = [summary_prompt] + messages_for_summary[-10:]  # Keep last 10 messages

        try:
            stream = await acompletion(
                model=self.state.profile.name,
                messages=summary_messages,
                max_tokens=self.state.runtime_config.max_tokens,
                stream=True,
            )

            # Collect all chunks from the stream
            content = ""
            async for chunk in stream:  # type: ignore
                if hasattr(chunk, "choices") and chunk.choices and hasattr(chunk.choices[0], "delta"):
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        content += delta.content

            # Update context with summary
            self.context.summary = {"content": content, "message_count": len(self.context.messages)}

            # Clear old messages except system prompts and recent ones
            system_messages = [msg for msg in self.context.messages if msg["role"] == "system"]
            recent_messages = self.context.messages[-10:]  # Keep last 10 messages

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
                    completion_tokens=getattr(usage, "completion_tokens", 0),
                    prompt_tokens=getattr(usage, "prompt_tokens", 0),
                    total_tokens=getattr(usage, "total_tokens", 0),
                )
                self._total_tokens += token_usage.total_tokens

                # Calculate context metrics using token counter
                estimates = self.estimate_tokens(self.context.messages)
                total_context_tokens = estimates["total_tokens"]

                context_metrics = ContextMetrics(
                    max_context_tokens=self.state.runtime_config.max_context_tokens,
                    current_context_tokens=total_context_tokens,
                    available_tokens=self.state.runtime_config.max_context_tokens - total_context_tokens,
                    context_utilization=total_context_tokens / self.state.runtime_config.max_context_tokens,
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
                        {"content": img.content, "mime_type": str(img.media_type), "file_name": img.file_name}
                        for img in images
                    ]
                    break

        return messages

    async def _handle_response(
        self,
        stream: Any,
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
                max_context_tokens=self.state.runtime_config.max_context_tokens,
                current_context_tokens=total_context_tokens,
                available_tokens=self.state.runtime_config.max_context_tokens - total_context_tokens,
                context_utilization=total_context_tokens / self.state.runtime_config.max_context_tokens,
            )

            # Update conversation context
            self.context.messages = messages
            self.context.messages.append({"role": "assistant", "content": content})
            self.context.message_count += 1

            # Handle context management
            await self._maybe_summarize_context()
            await self._maybe_add_reminder()

            return LLMResponse(
                content=content,
                success=True,
                model=self.state.profile,
                token_usage=token_usage or TokenUsage(
                    completion_tokens=0,
                    prompt_tokens=0,
                    total_tokens=0
                ),
                context_metrics=context_metrics
            )

        except Exception as e:
            return self._handle_error(e)

    def _handle_error(self, e: Exception) -> LLMResponse:
        """Handle errors and return a standardized error response."""
        error_msg = str(e)
        return LLMResponse(
            content="",
            success=False,
            model=self.state.profile,
            token_usage=TokenUsage(
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0
            ),
            error=error_msg
        )

    @abstractmethod
    async def process(
        self,
        prompt: Union[BasePrompt, "BasePrompt"],
        variables: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Process a prompt through the LLM and return a standardized response."""
        pass

    @abstractmethod
    async def batch_process(
        self,
        prompts: List[Union[BasePrompt, "BasePrompt"]],
        variables: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> List[LLMResponse]:
        """Process multiple BasePrompts in a batch."""
        pass

    @abstractmethod
    async def process_async(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Asynchronous prompt processing."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize async components of the interface.
        
        This method should be called after construction to set up any async resources
        like rate limiters, token counters, or provider-specific clients.
        
        All implementations must call super().initialize() to ensure proper initialization
        of base components.
        """
        # Initialize rate limiter storage if needed
        if self.rate_limiter and hasattr(self.rate_limiter, "initialize"):
            await self.rate_limiter.initialize()

    async def process_prompt(self, prompt: BasePrompt, variables: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Process a BasePrompt object and return a response."""
        # Validate prompt against model capabilities
        if prompt.attachments:
            for attachment in prompt.attachments:
                if not self.supports_media_type(attachment.media_type):
                    raise ValueError(
                        f"Media type {attachment.media_type} not supported by model {self.state.profile}"
                    )

        # Render the prompt with variables
        system_prompt, user_prompt, attachments = prompt.render(**(variables or {}))

        # Convert attachments to ImageInput objects
        image_inputs = [
            ImageInput(content=att["content"], media_type=att["mime_type"], file_name=att["file_name"])
            for att in (attachments or [])
        ]

        # Format messages
        messages = self._format_messages(
            input_data=user_prompt,
            system_prompt=system_prompt,
            images=image_inputs if image_inputs else None
        )

        # Check rate limits
        token_estimate = self.estimate_tokens(messages)
        can_proceed, error = await self._check_rate_limits(messages, token_estimate["total_tokens"])
        if not can_proceed:
            return LLMResponse(
                content=error or "Rate limit exceeded",
                success=False,
                model=self.state.profile,
                token_usage=TokenUsage(
                    completion_tokens=0,
                    prompt_tokens=0,
                    total_tokens=0
                ),
                metadata={"error": error}
            )

        try:
            # Process with model
            stream = await acompletion(
                model=self.state.profile.name,
                messages=messages,
                max_tokens=self.state.runtime_config.max_tokens,
                stream=True,
            )

            return await self._handle_response(stream, messages)

        except Exception as e:
            return self._handle_error(e)

    def validate_media_type(self, media_type: Union[str, MediaType]) -> MediaType:
        """Validate and convert media type to a supported format."""
        if isinstance(media_type, str):
            media_type = MediaType.from_mime_type(media_type)

        if not self.state.validate_media_type(media_type):
            # Default to JPEG if unsupported type
            return MediaType.JPEG
        return media_type

    def supports_media_type(self, media_type: Union[str, MediaType]) -> bool:
        """Check if a media type is supported by this LLM."""
        if isinstance(media_type, str):
            media_type = MediaType.from_mime_type(media_type)
        return self.state.validate_media_type(media_type)
