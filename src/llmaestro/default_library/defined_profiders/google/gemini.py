import base64
import io
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import google.generativeai as genai
from llmaestro.core.models import TokenUsage
from llmaestro.llm.enums import MediaType
from llmaestro.llm.interfaces.base import BaseLLMInterface, BasePrompt, ImageInput, LLMResponse
from llmaestro.prompts.types import PromptMetadata, ResponseFormat, ResponseFormatType, VersionInfo
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLMInterface):
    """Google Gemini LLM implementation."""

    # Override supported media types for Gemini-specific support
    SUPPORTED_MEDIA_TYPES = {MediaType.JPEG, MediaType.PNG, MediaType.WEBP}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configure Gemini
        genai.configure(api_key=self.state.credentials)
        self._model = genai.GenerativeModel(str(self.state.profile.name))
        self.stream = self.state.runtime_config.stream
        logger.info(f"Initialized GeminiLLM with model: {self.state.profile.name}")

    async def initialize(self) -> None:
        """Initialize async components of the interface."""
        await super().initialize()
        # Additional Gemini-specific initialization can go here

    def _validate_media_type(self, media_type: Union[str, MediaType]) -> MediaType:
        """Validate and convert media type to supported Gemini media type."""
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

    def _create_image_part(self, image_input: ImageInput) -> Any:
        """Create a properly formatted image part for Gemini."""
        if isinstance(image_input.content, str):
            # Decode base64 string
            image_data = base64.b64decode(image_input.content)
        else:
            image_data = image_input.content

        # Convert to PIL Image
        img = Image.open(io.BytesIO(image_data))
        return img

    async def process(
        self, prompt: Union[BasePrompt, "BasePrompt"], variables: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Process input data through Gemini and return a standardized response."""
        try:
            logger.debug(f"Processing prompt: {prompt}")
            logger.debug(f"Variables: {variables}")

            # Render the prompt with optional variables
            system_prompt, user_prompt, attachments = prompt.render(**(variables or {}))
            logger.debug(f"Rendered system prompt: {system_prompt}")
            logger.debug(f"Rendered user prompt: {user_prompt}")

            # Convert attachments to PIL Images if any
            images = []
            if attachments:
                for att in attachments:
                    img_input = ImageInput(
                        content=att["content"], media_type=att["mime_type"], file_name=att["file_name"]
                    )
                    images.append(self._create_image_part(img_input))

            # Get image dimensions for token counting
            image_dimensions = []
            if images:
                image_dimensions = [
                    self._get_image_dimensions(ImageInput(content=img.tobytes(), media_type=MediaType.JPEG))
                    for img in images
                ]

            # Combine system prompt and user prompt if system prompt exists
            # Since Gemini doesn't support system prompts directly
            full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

            # Estimate tokens including images
            token_estimates = self.token_counter.estimate_messages_with_images(
                messages=[{"role": "user", "content": full_prompt}],
                image_data=image_dimensions,
                model_family=self.model_family,
                model_name=str(self.state.profile.name),
            )

            # Check rate limits with image tokens included
            can_proceed, error_msg = await self._check_rate_limits(
                [{"role": "user", "content": full_prompt}], estimated_tokens=token_estimates["total_tokens"]
            )
            if not can_proceed:
                return LLMResponse(
                    content=f"Rate limit exceeded: {error_msg}",
                    success=False,
                    model=self.state.profile,
                    metadata={"error": "rate_limit_exceeded", "estimated_tokens": token_estimates["total_tokens"]},
                    token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                )

            try:
                logger.info("Making API call to Google Gemini...")

                # Create generation config
                generation_config = self._create_generation_config()

                if images:
                    # For vision model
                    response = await self._model.generate_content_async(
                        contents=[full_prompt, *images], generation_config=generation_config, stream=self.stream
                    )
                else:
                    # For text-only model
                    response = await self._model.generate_content_async(
                        contents=full_prompt, generation_config=generation_config, stream=self.stream
                    )

                if self.stream:
                    # Handle streaming response
                    content = ""
                    async for chunk in response:
                        if chunk.text:
                            content += chunk.text

                    # Create response with accumulated content
                    return LLMResponse(
                        content=content,
                        success=True,
                        model=self.state.profile,
                        metadata={
                            "id": "stream",
                            "cost": 0.0,
                            "image_tokens": token_estimates.get("image_tokens", 0),
                        },
                        token_usage=TokenUsage(
                            prompt_tokens=token_estimates["prompt_tokens"],
                            completion_tokens=len(content.split()) * 4,  # Rough estimate
                            total_tokens=token_estimates["total_tokens"] + (len(content.split()) * 4),
                        ),
                    )
                else:
                    # Handle non-streaming response
                    content = response.text

                    # Create response with content
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
                            prompt_tokens=token_estimates["prompt_tokens"],
                            completion_tokens=len(content.split()) * 4,  # Rough estimate
                            total_tokens=token_estimates["total_tokens"] + (len(content.split()) * 4),
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
        """Batch process prompts using Gemini's API."""
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
                    expected_response=ResponseFormat(format=ResponseFormatType.TEXT, schema=None),
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

    def _create_generation_config(self) -> genai.types.GenerationConfig:
        """Create a generation config for Gemini."""
        return genai.types.GenerationConfig(
            temperature=self.state.runtime_config.temperature, max_output_tokens=self.state.runtime_config.max_tokens
        )
