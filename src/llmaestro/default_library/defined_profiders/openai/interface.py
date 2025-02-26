"""OpenAI interface implementation."""

import logging
from typing import Any, Dict, List, Optional, Union, AsyncIterator, TYPE_CHECKING, cast, overload

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam

from llmaestro.core.models import LLMResponse, TokenUsage, ContextMetrics
from llmaestro.llm.interfaces.base import BaseLLMInterface, ImageInput
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.types import PromptMetadata, ResponseFormat, ResponseFormatType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SimplePrompt(BasePrompt):
    """A simple prompt implementation for handling string prompts."""

    async def load(self) -> None:
        """Implement abstract load method."""
        pass

    async def save(self) -> None:
        """Implement abstract save method."""
        pass


class OpenAIInterface(BaseLLMInterface):
    """OpenAI-specific implementation of the LLM interface."""

    def __init__(self, **data):
        logger.debug("Initializing OpenAIInterface")
        logger.debug(f"Init data: {data}")
        try:
            super().__init__(**data)
            self.client = AsyncOpenAI(api_key=self.credentials.key if self.credentials else None)
            logger.debug("OpenAIInterface initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAIInterface: {str(e)}", exc_info=True)
            raise

    def model_post_init(self, __context: Any) -> None:
        """Initialize the interface after Pydantic model validation."""
        logger.debug("Running OpenAIInterface post-init")
        super().model_post_init(__context)
        if not self.state:
            logger.warning("No state provided in post-init")
            return
        logger.debug("OpenAIInterface post-init completed")

    @property
    def model_family(self) -> str:
        """Get the model family."""
        return "openai"

    async def initialize(self) -> None:
        """Initialize async components of the interface."""
        await super().initialize()
        # Additional OpenAI-specific initialization can go here

    @overload
    async def process(self, prompt: str, variables: Optional[Dict[str, Any]] = None) -> LLMResponse:
        ...

    @overload
    async def process(self, prompt: BasePrompt, variables: Optional[Dict[str, Any]] = None) -> LLMResponse:
        ...

    async def process(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Process input using OpenAI's API."""
        try:
            # Convert string prompt to SimplePrompt if needed
            if isinstance(prompt, str):
                prompt = SimplePrompt(
                    name="direct_prompt",
                    description="Direct string prompt",
                    system_prompt="",
                    user_prompt=prompt,
                    metadata=PromptMetadata(
                        type="direct_input",
                        expected_response=ResponseFormat(format=ResponseFormatType.TEXT, schema=None),
                        tags=[],
                    ),
                )

            # Validate credentials
            self.validate_credentials()
            if not self.credentials:
                raise ValueError("No credentials provided")

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

            if self.state:
                model_name = self.state.profile.name
                max_tokens = self.state.runtime_config.max_tokens
                temperature = self.state.runtime_config.temperature
            else:
                raise ValueError("No state provided, a LLMState must be provided to the LLMInterface")

            # Make API call with credentials
            response = await self.client.chat.completions.create(
                messages=messages,
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )

            return await self._handle_response(response)

        except Exception as e:
            return self._handle_error(e)

    async def batch_process(
        self,
        prompts: List[Union[BasePrompt, str]],
        variables: Optional[List[Optional[Dict[str, Any]]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[LLMResponse]:
        """Process multiple prompts in a batch."""
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

    async def stream(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[LLMResponse]:
        """Stream responses from OpenAI's API one token at a time."""
        try:
            # Convert string prompt to SimplePrompt if needed
            if isinstance(prompt, str):
                prompt = SimplePrompt(
                    name="direct_prompt",
                    description="Direct string prompt",
                    system_prompt="",
                    user_prompt=prompt,
                    metadata=PromptMetadata(
                        type="direct_input",
                        expected_response=ResponseFormat(format=ResponseFormatType.TEXT, schema=None),
                        tags=[],
                    ),
                )

            # Validate credentials
            self.validate_credentials()
            if not self.credentials:
                raise ValueError("No credentials provided")

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

            if not self.state:
                raise ValueError("No state provided, a LLMState must be provided to the LLMInterface")

            # Get model configuration
            model_name = self.state.profile.name
            max_tokens = self.state.runtime_config.max_tokens
            temperature = self.state.runtime_config.temperature

            # Make streaming API call with credentials
            stream = await self.client.chat.completions.create(
                messages=messages,
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            # Process the streaming response
            try:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        yield LLMResponse(
                            content=content,
                            success=True,
                            token_usage=TokenUsage(
                                completion_tokens=1,  # Each chunk is roughly one token
                                prompt_tokens=0,  # We don't know prompt tokens in streaming
                                total_tokens=1,
                            ),
                            context_metrics=self._calculate_context_metrics(),
                            metadata={
                                "model": self.state.profile.name,
                                "is_streaming": True,
                                "is_partial": True,
                            },
                        )
            except Exception as e:
                yield self._handle_error(e)

        except Exception as e:
            yield self._handle_error(e)

    def _handle_error(self, e: Exception) -> LLMResponse:
        """Handle errors in LLM processing."""
        error_message = f"Error processing LLM request: {str(e)}"
        logger.error(error_message, exc_info=True)
        return LLMResponse(
            content="",
            success=False,
            error=str(e),
            metadata={"error_type": type(e).__name__},
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    def _format_messages(
        self, input_data: Any, system_prompt: Optional[str] = None, images: Optional[List[ImageInput]] = None
    ) -> List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]]:
        """Format input data and optional system prompt into messages."""
        messages: List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]] = []

        if system_prompt:
            system_message: ChatCompletionSystemMessageParam = {"role": "system", "content": system_prompt}
            messages.append(system_message)

        if isinstance(input_data, str):
            user_message: ChatCompletionUserMessageParam = {"role": "user", "content": input_data}
            messages.append(user_message)
        elif isinstance(input_data, dict):
            messages.append(cast(Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam], input_data))
        elif isinstance(input_data, list):
            messages.extend(
                cast(List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]], input_data)
            )

        # Add images if provided
        if images:
            # Convert the last user message to include images
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if msg["role"] == "user":
                    # Create a new message with images
                    content = msg["content"]
                    if not isinstance(content, str):
                        content = str(content)

                    messages[i] = cast(
                        ChatCompletionUserMessageParam,
                        {
                            "role": "user",
                            "content": content,
                            "images": [
                                {
                                    "content": str(img.content),
                                    "mime_type": str(img.media_type),
                                    "file_name": img.file_name or "",
                                }
                                for img in images
                            ],
                        },
                    )
                    break

        return messages

    async def _handle_response(self, response: ChatCompletion) -> LLMResponse:
        """Handle the response from OpenAI's API."""
        try:
            # Extract content and metadata from response
            content = response.choices[0].message.content if response.choices else ""
            if content is None:
                content = ""  # Ensure content is never None

            # Create token usage info
            usage = response.usage
            if usage is None:
                # Fallback if usage is not available
                token_usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            else:
                token_usage = TokenUsage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                )

            return LLMResponse(
                content=content,
                success=True,
                token_usage=token_usage,
                context_metrics=self._calculate_context_metrics(),
                metadata={
                    "model": self.state.profile.name if self.state else "unknown",
                    "is_streaming": False,
                    "is_partial": False,
                },
            )
        except Exception as e:
            return self._handle_error(e)

    def _calculate_context_metrics(self) -> ContextMetrics:
        """Calculate context window metrics."""
        if not self.state:
            return ContextMetrics(
                max_context_tokens=0,
                current_context_tokens=0,
                available_tokens=0,
                context_utilization=0.0,
            )

        max_tokens = self.state.runtime_config.max_context_tokens
        current_tokens = self.context.total_tokens.total_tokens if self.context and self.context.total_tokens else 0
        available = max(0, max_tokens - current_tokens)
        utilization = float(current_tokens) / float(max_tokens) if max_tokens > 0 else 0.0

        return ContextMetrics(
            max_context_tokens=max_tokens,
            current_context_tokens=current_tokens,
            available_tokens=available,
            context_utilization=utilization,
        )
