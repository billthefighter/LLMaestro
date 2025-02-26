"""Base interfaces for LLM providers."""
from __future__ import annotations

import base64
import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, TYPE_CHECKING, AsyncIterator

from pydantic import BaseModel, Field, ConfigDict

from llmaestro.core.conversations import ConversationContext
from llmaestro.core.models import (
    LLMResponse,
)
from llmaestro.llm.credentials import APIKey
from llmaestro.prompts.base import BasePrompt
from llmaestro.llm.enums import MediaType
from .tokenizers import BaseTokenizer
from llmaestro.llm.models import LLMState  # Direct import instead of TYPE_CHECKING

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


class BaseLLMInterface(BaseModel):
    """Base interface for LLM interactions."""

    # Default supported media types (can be overridden by specific implementations)
    SUPPORTED_MEDIA_TYPES: Set[MediaType] = {MediaType.JPEG, MediaType.PNG, MediaType.PDF}

    # Pydantic fields
    context: ConversationContext = Field(
        default_factory=ConversationContext, description="Current conversation context"
    )
    tokenizer: Optional[BaseTokenizer] = Field(default=None, description="Tokenizer instance")
    state: LLMState = Field(description="Complete state container for LLM instances")
    ignore_missing_credentials: bool = Field(default=False, description="Whether to ignore missing credentials")
    credentials: Optional[APIKey] = Field(default=None, description="API credentials")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        from_attributes=True,
        extra="allow",
        validate_default=False,
        frozen=False,
        populate_by_name=True,
    )

    def __init__(self, **data):
        logger.debug("Initializing BaseLLMInterface")
        logger.debug(f"Base init data: {data}")
        try:
            super().__init__(**data)
            logger.debug("BaseLLMInterface initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BaseLLMInterface: {str(e)}", exc_info=True)
            raise

    def model_post_init(self, __context: Any) -> None:
        """Initialize the interface after Pydantic model validation."""
        logger.debug("Running BaseLLMInterface post-init")
        if not self.state:
            logger.warning("No state provided in base post-init")
            return
        logger.debug("BaseLLMInterface post-init completed")

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
        """Process a prompt through the LLM and return a standardized response."""
        pass

    @abstractmethod
    async def batch_process(
        self,
        prompts: List[Union[BasePrompt, str]],
        variables: Optional[List[Optional[Dict[str, Any]]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[LLMResponse]:
        """Process multiple prompts in a batch."""
        pass

    @abstractmethod
    async def stream(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[LLMResponse]:
        """Stream responses from the LLM one token at a time."""
        pass

    async def initialize(self) -> None:
        """Initialize the interface."""
        pass

    async def shutdown(self) -> None:
        """Shutdown the interface."""
        pass
