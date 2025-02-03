import io
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from anthropic import Anthropic
from anthropic.types import (
    ContentBlockParam,
)
from PIL import Image

from src.llm.models import ModelFamily

from .base import BaseLLMInterface, ImageInput, LLMResponse, TokenUsage

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class AnthropicLLM(BaseLLMInterface):
    """Anthropic Claude LLM implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = Anthropic(api_key=self.config.api_key)
        logger.info(f"Initialized AnthropicLLM with model: {self.config.model_name}")

    @property
    def model_family(self) -> ModelFamily:
        """Get the model family for this interface."""
        return ModelFamily.CLAUDE

    def _get_image_dimensions(self, image_input: ImageInput) -> Dict[str, int]:
        """Get image dimensions from an ImageInput."""
        if isinstance(image_input.content, str):
            # Decode base64 string
            import base64

            image_data = base64.b64decode(image_input.content)
        else:
            image_data = image_input.content

        # Open image and get dimensions
        img = Image.open(io.BytesIO(image_data))
        return {"width": img.width, "height": img.height}

    def _create_message_param(
        self, role: Literal["user", "assistant"], content_blocks: List[ContentBlockParam]
    ) -> Dict[str, Any]:
        """Create a message parameter dictionary."""
        return {"role": role, "content": content_blocks}

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

            # Prepare API call parameters
            create_params = {
                "model": self.config.model_name,
                "messages": [{"role": msg["role"], "content": msg["content"]} for msg in messages],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": False,
            }

            # Only add system parameter if we have a system prompt
            if system_prompt:
                create_params["system"] = system_prompt

            logger.debug(f"API call parameters: {create_params}")

            # Make API call with parameters and await the response
            try:
                logger.info("Making API call to Anthropic...")
                response = await self.client.messages.create(**create_params)
                logger.debug(f"Received response: {response}")

                # Extract text from the response
                content = ""
                if hasattr(response, "content"):
                    for block in response.content:
                        if hasattr(block, "text"):
                            content += block.text

                logger.debug(f"Extracted content: {content}")

                # If the content is already JSON, return it as is
                try:
                    json.loads(content)
                    return LLMResponse(
                        content=content,
                        metadata={"id": response.id, "cost": 0.0, "image_tokens": 0},
                        token_usage=TokenUsage(
                            prompt_tokens=response.usage.input_tokens,
                            completion_tokens=response.usage.output_tokens,
                            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                        ),
                    )
                except json.JSONDecodeError:
                    # If not JSON, wrap it in our expected format
                    formatted_response = {"message": content.strip(), "timestamp": datetime.utcnow().isoformat() + "Z"}
                    return LLMResponse(
                        content=json.dumps(formatted_response),
                        metadata={"id": response.id, "cost": 0.0, "image_tokens": 0},
                        token_usage=TokenUsage(
                            prompt_tokens=response.usage.input_tokens,
                            completion_tokens=response.usage.output_tokens,
                            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
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
