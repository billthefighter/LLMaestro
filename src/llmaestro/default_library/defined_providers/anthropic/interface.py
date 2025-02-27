import base64
import io
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageStreamEvent,
)
from llmaestro.core.models import LLMResponse, TokenUsage
from llmaestro.llm.interfaces.base import BaseLLMInterface, BasePrompt, ImageInput, MediaType
from llmaestro.prompts.types import PromptMetadata, ResponseFormat, ResponseFormatType, VersionInfo
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class AnthropicLLM(BaseLLMInterface):
    """Anthropic Claude LLM implementation."""

    # Override supported media types for Anthropic-specific support
    SUPPORTED_MEDIA_TYPES: Set[MediaType] = {MediaType.JPEG, MediaType.PNG, MediaType.GIF, MediaType.WEBP}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = AsyncAnthropic(api_key=self.state.credentials)
        self.stream = self.state.runtime_config.stream
        logger.info(f"Initialized AnthropicLLM with model: {self.state.profile.name}")

    async def initialize(self) -> None:
        """Initialize async components of the interface."""
        await super().initialize()
        # Additional Anthropic-specific initialization can go here

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

        # Process attachments
        if "attachments" in message:
            for att in message["attachments"]:
                img_input = ImageInput(content=att["content"], media_type=att["media_type"], file_name=att["file_name"])
                source = {
                    "type": "base64",
                    "media_type": str(self._validate_media_type(img_input.media_type)),
                    "data": img_input.content
                    if isinstance(img_input.content, str)
                    else base64.b64encode(img_input.content).decode(),
                }
                content_blocks.append({"type": "image", "source": source})

        return content_blocks

    def _format_messages(self, input_data: str) -> List[Dict[str, Any]]:
        """Format messages for Anthropic API."""
        messages = []

        # Add user message
        messages.append({"role": "user", "content": input_data})

        return messages

    async def process(
        self, prompt: Union[BasePrompt, "BasePrompt"], variables: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Process input data through Claude and return a standardized response."""
        try:
            # Ensure model descriptor is available (it should be from base class __init__)
            logger.debug(f"Processing prompt: {prompt}")
            logger.debug(f"Variables: {variables}")

            # Render the prompt with optional variables
            system_prompt, user_prompt, attachments = prompt.render(**(variables or {}))
            logger.debug(f"Rendered system prompt: {system_prompt}")
            logger.debug(f"Rendered user prompt: {user_prompt}")

            # Convert attachments to ImageInput objects if any
            images = (
                [
                    ImageInput(content=att["content"], media_type=att["media_type"], file_name=att["file_name"])
                    for att in attachments
                ]
                if attachments
                else None
            )

            # Format messages - only include user message, system prompt is handled separately
            messages = self._format_messages(input_data=user_prompt)
            logger.debug(f"Formatted messages: {messages}")

            # Get image dimensions for token counting
            image_dimensions = []
            if images:
                image_dimensions = [self._get_image_dimensions(img) for img in images]

            # Estimate tokens including images
            token_estimates = self.token_counter.estimate_messages_with_images(
                messages=messages,
                image_data=image_dimensions,
                model_family=self.model_family,
                model_name=str(self.state.profile.name),
            )

            # Check rate limits with image tokens included
            can_proceed, error_msg = await self._check_rate_limits(
                messages, estimated_tokens=token_estimates["total_tokens"]
            )
            if not can_proceed:
                return LLMResponse(
                    content=f"Rate limit exceeded: {error_msg}",
                    success=False,
                    model=self.state.profile,
                    metadata={"error": "rate_limit_exceeded", "estimated_tokens": token_estimates["total_tokens"]},
                    token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                )

            # Convert messages to Anthropic format with proper content blocks
            anthropic_messages = []
            for msg in messages:
                content_blocks = await self._create_message_content(msg, images if msg.get("role") == "user" else None)
                anthropic_messages.append({"role": msg["role"], "content": content_blocks})

            # Prepare API call parameters
            create_params = self._create_message_params()

            # Add system prompt as a top-level parameter
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
                        success=True,
                        model=self.state.profile,
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

                    return LLMResponse(
                        content=content,
                        success=True,
                        model=self.state.profile,
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
            return self._handle_error(e)

    async def batch_process(
        self, prompts: List[Union[BasePrompt, "BasePrompt"]], variables: Optional[List[Optional[Dict[str, Any]]]] = None
    ) -> List[LLMResponse]:
        """Batch process prompts using Anthropic's API."""
        # Ensure variables list matches prompts length if provided
        if variables is not None and len(variables) != len(prompts):
            raise ValueError("Number of variable sets must match number of prompts")

        # Process each prompt
        results = []
        for i, prompt in enumerate(prompts):
            # Use corresponding variables if provided, otherwise None
            prompt_vars = variables[i] if variables is not None else None
            result = await self.process(prompt, prompt_vars)
            results.append(result)

        return results

    async def process_async(
        self, prompt: Union[BasePrompt, str], variables: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Process prompt asynchronously."""
        if isinstance(prompt, str):
            # Convert string prompt to BasePrompt
            prompt = BasePrompt(
                name="direct_prompt",
                description="Direct string prompt",
                system_prompt="",
                user_prompt=prompt,
                metadata=PromptMetadata(
                    type="direct_input",
                    expected_response=ResponseFormat(format=ResponseFormatType.TEXT, response_schema=None),
                    tags=[],
                ),
                current_version=VersionInfo(
                    number="1.0.0",
                    author="system",
                    description="Direct string prompt",
                    timestamp=datetime.now(),
                    change_type="initial",
                ),
            )
        return await self.process(prompt, variables)

    def _handle_error(self, e: Exception) -> LLMResponse:
        """Handle errors in LLM processing."""
        error_message = f"Error processing LLM request: {str(e)}"
        logger.error(error_message, exc_info=True)
        return LLMResponse(
            content="",
            success=False,
            model=self.state.profile,
            error=str(e),
            metadata={"error_type": type(e).__name__},
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    def _create_message_params(self) -> Dict[str, Any]:
        """Create message parameters for Anthropic."""
        return {
            "model": self.state.profile.name,
            "max_tokens": self.state.runtime_config.max_tokens,
            "temperature": self.state.runtime_config.temperature,
            "stream": self.stream,
        }
