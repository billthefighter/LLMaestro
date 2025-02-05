import base64
import io
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageStreamEvent,
)
from PIL import Image

from src.llm.models import ModelFamily

from .base import BaseLLMInterface, ImageInput, LLMResponse, MediaType, TokenUsage

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class AnthropicLLM(BaseLLMInterface):
    """Anthropic Claude LLM implementation."""

    # Override supported media types for Anthropic-specific support
    SUPPORTED_MEDIA_TYPES = {MediaType.JPEG, MediaType.PNG, MediaType.GIF, MediaType.WEBP}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = AsyncAnthropic(api_key=self.config.api_key)
        self.stream = getattr(self.config, "stream", False)  # Default to False if not specified
        logger.info(f"Initialized AnthropicLLM with model: {self.config.model_name}")

    @property
    def model_family(self) -> ModelFamily:
        """Get the model family for this interface."""
        return ModelFamily.CLAUDE

    def _validate_media_type(self, media_type: Union[str, MediaType]) -> MediaType:
        """Validate and convert media type to supported Anthropic media type."""
        if isinstance(media_type, str):
            media_type = MediaType.from_mime_type(media_type)

        if media_type not in self.SUPPORTED_MEDIA_TYPES:
            # Default to JPEG if unsupported type
            return MediaType.JPEG
        return media_type

    def _get_image_dimensions(self, image_input: ImageInput) -> Dict[str, int]:
        """Get image dimensions from an ImageInput."""
        if isinstance(image_input.content, str):
            # Decode base64 string
            image_data = base64.b64decode(image_input.content)
        else:
            image_data = image_input.content

        # Open image and get dimensions
        img = Image.open(io.BytesIO(image_data))
        return {"width": img.width, "height": img.height}

    def _create_image_source(self, media_type: str, data: str) -> Dict[str, str]:
        """Create a properly typed image source."""
        return {
            "type": "base64",
            "media_type": str(self._validate_media_type(media_type)),
            "data": data,
        }

    async def _create_message_content(
        self, message: Dict[str, Any], images: Optional[List[ImageInput]] = None
    ) -> List[Dict[str, Any]]:
        """Create message content blocks including text and images."""
        content_blocks: List[Dict[str, Any]] = []

        # Add text content
        if "content" in message:
            content_blocks.append({"type": "text", "text": message["content"]})

        # Add image content if present and this is a user message
        if images and message.get("role") == "user":
            for img in images:
                source = {
                    "type": "base64",
                    "media_type": str(self._validate_media_type(img.media_type)),
                    "data": img.content if isinstance(img.content, str) else base64.b64encode(img.content).decode(),
                }
                content_blocks.append({"type": "image", "source": source})

        return content_blocks

    async def process(
        self, input_data: Any, system_prompt: Optional[str] = None, images: Optional[List[ImageInput]] = None
    ) -> LLMResponse:
        """Process input data through Claude and return a standardized response."""
        try:
            logger.debug(f"Processing input data: {input_data}")
            logger.debug(f"System prompt: {system_prompt}")

            messages = self._format_messages(input_data, system_prompt=None, images=images)
            logger.debug(f"Formatted messages: {messages}")

            # Extract system prompt from input_data if present
            if isinstance(input_data, list):
                system_messages = [msg for msg in input_data if msg.get("role") == "system"]
                if system_messages and not system_prompt:
                    system_prompt = system_messages[0].get("content")
                    logger.debug(f"Extracted system prompt: {system_prompt}")
                # Remove system messages from the list
                messages = [msg for msg in messages if msg.get("role") != "system"]
                logger.debug(f"Messages after system removal: {messages}")

            # Get image dimensions for token counting
            image_dimensions = []
            if images:
                image_dimensions = [self._get_image_dimensions(img) for img in images]

            # Estimate tokens including images
            token_estimates = self._token_counter.estimate_messages_with_images(
                messages=messages,
                image_data=image_dimensions,
                model_family=self.model_family,
                model_name=self.config.model_name,
            )

            # Check rate limits with image tokens included
            can_proceed, error_msg = await self._check_rate_limits(
                messages, estimated_tokens=token_estimates["total_tokens"]
            )
            if not can_proceed:
                return LLMResponse(
                    content=f"Rate limit exceeded: {error_msg}",
                    metadata={"error": "rate_limit_exceeded", "estimated_tokens": token_estimates["total_tokens"]},
                )

            # Convert messages to Anthropic format with proper content blocks
            anthropic_messages = []
            for msg in messages:
                content_blocks = await self._create_message_content(msg, images if msg.get("role") == "user" else None)
                anthropic_messages.append({"role": msg["role"], "content": content_blocks})

            # Prepare API call parameters
            create_params = {
                "model": self.config.model_name,
                "messages": anthropic_messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": self.stream,
            }

            # Only add system parameter if we have a system prompt
            if system_prompt:
                create_params["system"] = system_prompt

            logger.debug(f"API call parameters: {create_params}")

            try:
                logger.info("Making API call to Anthropic...")

                if self.stream:
                    # Handle streaming response
                    content = ""
                    final_message: Any = None

                    async for message in self.client.messages.stream(**create_params):  # type: ignore
                        if isinstance(message, MessageStreamEvent):
                            delta = getattr(message, "delta", None)
                            if not delta:
                                continue

                            if message.type == "content_block_delta":
                                text = getattr(delta, "text", None)
                                if text:
                                    content += text
                            elif message.type == "message_delta":
                                msg = getattr(delta, "message", None)
                                if msg:
                                    final_message = msg

                    # Create response with accumulated content
                    return LLMResponse(
                        content=content,
                        metadata={
                            "id": getattr(final_message, "id", "stream"),
                            "cost": 0.0,
                            "image_tokens": token_estimates.get("image_tokens", 0),
                        },
                        token_usage=TokenUsage(
                            prompt_tokens=getattr(final_message.usage, "input_tokens", 0)
                            if hasattr(final_message, "usage")
                            else 0,
                            completion_tokens=getattr(final_message.usage, "output_tokens", 0)
                            if hasattr(final_message, "usage")
                            else 0,
                            total_tokens=sum(
                                [
                                    getattr(final_message.usage, "input_tokens", 0)
                                    if hasattr(final_message, "usage")
                                    else 0,
                                    getattr(final_message.usage, "output_tokens", 0)
                                    if hasattr(final_message, "usage")
                                    else 0,
                                ]
                            ),
                        ),
                    )
                else:
                    # Handle non-streaming response
                    response = await self.client.messages.create(**create_params)
                    content = "".join(block.text for block in response.content if hasattr(block, "text"))

                logger.debug(f"Extracted content: {content}")

                # If the content is already JSON, return it as is
                try:
                    json.loads(content)
                    return LLMResponse(
                        content=content,
                        metadata={
                            "id": getattr(response, "id", "unknown"),
                            "cost": 0.0,
                            "image_tokens": token_estimates.get("image_tokens", 0),
                        },
                        token_usage=TokenUsage(
                            prompt_tokens=getattr(response.usage, "input_tokens", 0)
                            if hasattr(response, "usage")
                            else 0,
                            completion_tokens=getattr(response.usage, "output_tokens", 0)
                            if hasattr(response, "usage")
                            else 0,
                            total_tokens=sum(
                                [
                                    getattr(response.usage, "input_tokens", 0) if hasattr(response, "usage") else 0,
                                    getattr(response.usage, "output_tokens", 0) if hasattr(response, "usage") else 0,
                                ]
                            ),
                        ),
                    )
                except json.JSONDecodeError:
                    # If not JSON, wrap it in our expected format
                    formatted_response = {"message": content.strip(), "timestamp": datetime.utcnow().isoformat() + "Z"}
                    return LLMResponse(
                        content=json.dumps(formatted_response),
                        metadata={
                            "id": getattr(response, "id", "unknown"),
                            "cost": 0.0,
                            "image_tokens": token_estimates.get("image_tokens", 0),
                        },
                        token_usage=TokenUsage(
                            prompt_tokens=getattr(response.usage, "input_tokens", 0)
                            if hasattr(response, "usage")
                            else 0,
                            completion_tokens=getattr(response.usage, "output_tokens", 0)
                            if hasattr(response, "usage")
                            else 0,
                            total_tokens=sum(
                                [
                                    getattr(response.usage, "input_tokens", 0) if hasattr(response, "usage") else 0,
                                    getattr(response.usage, "output_tokens", 0) if hasattr(response, "usage") else 0,
                                ]
                            ),
                        ),
                    )

            except Exception as e:
                logger.error(f"API call failed: {str(e)}", exc_info=True)
                return self._handle_error(e)

        except Exception as e:
            logger.error(f"Process failed: {str(e)}", exc_info=True)
            return self._handle_error(e)

    def _handle_error(self, e: Exception) -> LLMResponse:
        """Handle errors in LLM processing."""
        error_message = f"Error processing LLM request: {str(e)}"
        logger.error(error_message, exc_info=True)
        return LLMResponse(
            content=json.dumps(
                {
                    "error": str(e),
                    "message": "An error occurred while processing the request",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            ),
            metadata={"error": str(e)},
        )
