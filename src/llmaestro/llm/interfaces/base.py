"""Base interfaces for LLM providers."""

import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, TYPE_CHECKING, AsyncIterator

from pydantic import BaseModel, Field, ConfigDict

from llmaestro.core.conversations import ConversationContext

from llmaestro.core.models import (
    ContextMetrics,
    LLMResponse,
    TokenUsage,
)
from llmaestro.llm.credentials import APIKey
from llmaestro.prompts.base import BasePrompt
from llmaestro.llm.enums import MediaType

from .tokenizers import BaseTokenizer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..models import LLMState


@dataclass
class ImageInput:
    """Input image data for vision models."""

    content: Union[str, bytes]  # Base64 encoded string or raw bytes
    media_type: MediaType  # Media type of the image
    file_name: Optional[str] = None

    def __init__(
        self,
        content: Union[str, bytes],
        # TODO: this should not be a string, and should be iniitalized as none
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
    context: ConversationContext = Field(
        default_factory=ConversationContext, description="Current conversation context"
    )
    tokenizer: Optional[BaseTokenizer] = Field(default=None, description="Tokenizer instance")
    ignore_missing_credentials: bool = Field(default=False, description="Ignore missing credentials")
    state: Optional["LLMState"] = Field(default=None, description="LLM state")
    credentials: Optional[APIKey] = Field(default=None, description="API credentials")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Initialize the interface after Pydantic model validation."""
        if not self.state:
            return

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Basic token counting."""
        if not self.tokenizer:
            return 0
        return self.tokenizer.count_messages(messages)

    def validate_credentials(self) -> None:
        """Validate the credentials for the LLM provider."""
        if not self.credentials and not self.ignore_missing_credentials:
            raise ValueError(f"API key is required for provider {self.state.model_family if self.state else 'unknown'}")

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

    @abstractmethod
    async def process(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Process a prompt through the LLM and return a standardized response.

        This is the main entry point for processing prompts. It should:
        1. Convert string prompts to BasePrompt if needed
        2. Add the prompt to the conversation context
        3. Process the prompt and get a response
        4. Add the response to the conversation context
        5. Return a standardized LLMResponse
        """
        pass

    @abstractmethod
    async def batch_process(
        self,
        prompts: List[Union[BasePrompt, str]],
        variables: Optional[List[Optional[Dict[str, Any]]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[LLMResponse]:
        """Process multiple prompts in a batch.

        This method should:
        1. Handle batching efficiently using provider-specific optimizations
        2. Maintain conversation context for each prompt
        3. Respect rate limits across the batch
        4. Optionally process in parallel if supported by provider

        Args:
            prompts: List of prompts to process
            variables: Optional list of variable dictionaries, one per prompt
            batch_size: Optional batch size for processing chunks of prompts
        """
        pass

    @abstractmethod
    async def stream(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[LLMResponse]:
        """Stream responses from the LLM one token at a time.

        This method provides a streaming interface to get responses as they are generated.
        Each yielded LLMResponse will contain a partial completion that should be concatenated
        with previous responses to form the complete response.

        Args:
            prompt: The prompt to process
            variables: Optional variables for prompt templating

        Returns:
            An async iterator yielding partial LLMResponses as they are generated
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize async components of the interface.

        This method should be called after construction to set up any async resources
        like rate limiters, token counters, or provider-specific clients.

        All implementations must call super().initialize() to ensure proper initialization
        of base components.
        """
        pass

    async def _handle_response(
        self,
        stream: Any,
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

            # Create response node
            response = LLMResponse(
                content=content,
                success=True,
                token_usage=token_usage or TokenUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0),
                context_metrics=self._calculate_context_metrics(),
                metadata={
                    "model": self.state.profile.name if self.state and self.state.profile else None,
                },
            )

            # Add response to conversation context
            self.context.add_node(
                content=response,
                node_type="response",
                metadata={
                    "model": self.state.profile.name if self.state and self.state.profile else None,
                },
            )

            return response

        except Exception as e:
            return self._handle_error(e)

    def _calculate_context_metrics(self) -> Optional[ContextMetrics]:
        """Calculate context metrics based on current state."""
        if not self.state or not self.state.runtime_config:
            return None

        return ContextMetrics(
            max_context_tokens=self.state.runtime_config.max_context_tokens,
            current_context_tokens=self.context.message_count,
            available_tokens=self.state.runtime_config.max_context_tokens - self.context.message_count,
            context_utilization=self.context.message_count / self.state.runtime_config.max_context_tokens,
        )

    def _handle_error(self, e: Exception) -> LLMResponse:
        """Handle errors and return a standardized error response."""
        error_msg = str(e)
        return LLMResponse(
            content="",
            success=False,
            token_usage=TokenUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0),
            error=error_msg,
            metadata={"error_type": type(e).__name__},
        )

    def validate_media_type(self, media_type: Union[str, MediaType]) -> MediaType:
        """Validate and convert media type to a supported format."""
        if isinstance(media_type, str):
            media_type = MediaType.from_mime_type(media_type)

        if not self.supports_media_type(media_type):
            # Default to JPEG if unsupported type
            return MediaType.JPEG
        return media_type

    def supports_media_type(self, media_type: Union[str, MediaType]) -> bool:
        """Check if a media type is supported by this LLM."""
        if isinstance(media_type, str):
            media_type = MediaType.from_mime_type(media_type)
        return media_type in self.SUPPORTED_MEDIA_TYPES
