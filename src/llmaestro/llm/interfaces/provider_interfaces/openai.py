from typing import Any, Dict, List, Optional, Union

from litellm import acompletion

from llmaestro.llm.interfaces.base import BaseLLMInterface, BasePrompt, ImageInput, LLMResponse
from llmaestro.llm.models import ModelFamily


class OpenAIInterface(BaseLLMInterface):
    """OpenAI-specific implementation of the LLM interface."""

    @property
    def model_family(self) -> ModelFamily:
        """Get the model family for this interface."""
        return ModelFamily.GPT

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
                    content=f"Rate limit exceeded: {error_msg}", metadata={"error": "rate_limit_exceeded"}
                )

            # Check if model supports streaming
            supports_streaming = self.capabilities and self.capabilities.supports_streaming

            # Make API call with appropriate streaming setting
            response = await acompletion(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
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
