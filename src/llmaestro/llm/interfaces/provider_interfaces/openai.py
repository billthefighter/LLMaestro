import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from litellm import acompletion
from llmaestro.core.models import TokenUsage
from llmaestro.llm.interfaces.base import BaseLLMInterface, BasePrompt, ImageInput, LLMResponse
from llmaestro.llm.models import ModelFamily
from llmaestro.prompts.types import PromptMetadata, ResponseFormat, VersionInfo

logger = logging.getLogger(__name__)


class OpenAIInterface(BaseLLMInterface):
    """OpenAI-specific implementation of the LLM interface."""

    @property
    def model_family(self) -> ModelFamily:
        """Get the model family for this interface."""
        return ModelFamily.GPT

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
                    type="direct_input", expected_response=ResponseFormat(format="text", schema=None), tags=[]
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

    async def process(
        self, prompt: Union[BasePrompt, "BasePrompt"], variables: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Process input using OpenAI's API via LiteLLM."""
        try:
            # Render the prompt with optional variables
            system_prompt, user_prompt, attachments = prompt.render(**(variables or {}))

            # Convert attachments to ImageInput objects if any
            images = (
                [
                    ImageInput(content=att["content"], media_type=att["mime_type"], file_name=att["file_name"])
                    for att in attachments
                ]
                if attachments
                else None
            )

            # Format messages
            messages = self._format_messages(input_data=user_prompt, system_prompt=system_prompt, images=images)

            # Check rate limits
            can_proceed, error_msg = await self._check_rate_limits(messages)
            if not can_proceed:
                return LLMResponse(
                    content=f"Rate limit exceeded: {error_msg}",
                    success=False,
                    model=self._model_descriptor,
                    token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                    metadata={"error": "rate_limit_exceeded"},
                )

            # Check if model supports streaming
            supports_streaming = self.capabilities and self.capabilities.supports_streaming

            # Make API call with appropriate streaming setting
            response = await acompletion(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=supports_streaming,
            )

            return await self._handle_response(response, messages)

        except Exception as e:
            return self._handle_error(e)

    async def batch_process(
        self, prompts: List[Union[BasePrompt, "BasePrompt"]], variables: Optional[List[Optional[Dict[str, Any]]]] = None
    ) -> List[LLMResponse]:
        """Batch process prompts using OpenAI's API."""
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

    def _handle_error(self, e: Exception) -> LLMResponse:
        """Handle errors in LLM processing."""
        error_message = f"Error processing LLM request: {str(e)}"
        logger.error(error_message, exc_info=True)
        # Model descriptor is guaranteed to be non-None by base class __init__
        assert self._model_descriptor is not None, "Model descriptor not initialized"
        return LLMResponse(
            content="",
            success=False,
            model=self._model_descriptor,
            error=str(e),
            metadata={"error_type": type(e).__name__},
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
