"""OpenAI interface implementation."""

import logging
from typing import Any, Dict, List, Optional, Union, AsyncIterator, TYPE_CHECKING, cast, overload, BinaryIO
import base64

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.file_object import FileObject
from openai.types import FilePurpose

from llmaestro.core.models import LLMResponse, TokenUsage, ContextMetrics
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.core.attachments import BaseAttachment, ImageAttachment, AttachmentConverter, FileAttachment
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.prompts.types import PromptMetadata, ResponseFormat
from llmaestro.llm.responses import ResponseFormatType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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

    async def _prepare_request(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]], str, float, int]:
        """Prepare common request parameters for both streaming and non-streaming calls."""
        logger.debug("Starting _prepare_request")
        # Convert string prompt to MemoryPrompt if needed
        if isinstance(prompt, str):
            prompt = MemoryPrompt(
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
            logger.debug("Converted string prompt to MemoryPrompt")

        # Validate credentials
        self.validate_credentials()
        if not self.credentials:
            raise ValueError("No credentials provided")

        # Render the prompt with optional variables
        logger.debug("Rendering prompt with variables")
        system_prompt, user_prompt, attachment_dicts = prompt.render(**(variables or {}))
        logger.debug(f"Rendered system prompt: {system_prompt}")
        logger.debug(f"Rendered user prompt: {user_prompt}")

        # Log attachments with truncated content
        for att in attachment_dicts:
            content = att.get("content", "")
            if len(content) > 40:
                truncated_content = f"{content[:30]}...{content[-10:]}"
            else:
                truncated_content = content
            logger.debug(
                f"Attachment: type={att.get('media_type')}, name={att.get('file_name')}, content_preview={truncated_content}"
            )

        # Convert dictionary attachments to BaseAttachment objects
        attachments = [AttachmentConverter.from_dict(att) for att in attachment_dicts]
        logger.debug(f"Number of converted attachments: {len(attachments)}")

        # Format messages
        messages = self._format_messages(input_data=user_prompt, system_prompt=system_prompt, attachments=attachments)
        logger.debug(f"Formatted messages: {messages}")

        if not self.state:
            raise ValueError("No state provided, a LLMState must be provided to the LLMInterface")

        # Get model configuration
        model_name = self.state.profile.name
        max_tokens = self.state.runtime_config.max_tokens
        temperature = self.state.runtime_config.temperature

        return messages, model_name, temperature, max_tokens

    async def _create_chat_completion(
        self,
        messages: List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]],
        model: str,
        max_tokens: int,
        temperature: float,
        stream: bool = False,
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """Create a chat completion with the given parameters."""
        return await self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

    def _create_response_metadata(self, is_streaming: bool = False, is_partial: bool = False) -> Dict[str, Any]:
        """Create common metadata for LLMResponse."""
        return {
            "model": self.state.profile.name if self.state else "unknown",
            "is_streaming": is_streaming,
            "is_partial": is_partial,
        }

    async def process(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Process input using OpenAI's API."""
        try:
            logger.debug(f"Processing prompt: {prompt}")
            logger.debug(f"Variables: {variables}")

            # Safe access to metadata and expected_response with proper null checks
            metadata = prompt.metadata if isinstance(prompt, BasePrompt) else None
            expected_response = metadata.expected_response if metadata else None

            logger.debug(f"Prompt metadata: {metadata}")
            logger.debug(f"Expected response format: {expected_response}")

            messages, model_name, temperature, max_tokens = await self._prepare_request(prompt, variables)
            logger.debug(f"Prepared messages: {messages}")
            logger.debug(f"Model name: {model_name}, Temperature: {temperature}, Max tokens: {max_tokens}")
            response = await self._create_chat_completion(
                messages=messages,
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            if not isinstance(response, ChatCompletion):
                raise ValueError("Expected ChatCompletion response for non-streaming request")
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
            messages, model_name, temperature, max_tokens = await self._prepare_request(prompt, variables)
            stream = await self._create_chat_completion(
                messages=messages,
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            if not isinstance(stream, AsyncIterator):
                raise ValueError("Expected AsyncIterator response for streaming request")

            async for chunk in stream:
                try:
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
                            metadata=self._create_response_metadata(is_streaming=True, is_partial=True),
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

    async def upload_file(self, file: Union[bytes, BinaryIO], purpose: FilePurpose = "assistants") -> FileObject:
        """Upload a file to OpenAI.

        Args:
            file: The file content as bytes or a file-like object
            purpose: The purpose of the file ("assistants", "vision", "batch", or "fine-tune")

        Returns:
            FileObject: The uploaded file object from OpenAI
        """
        return await self.client.files.create(file=file, purpose=purpose)

    def _format_messages(
        self, input_data: Any, system_prompt: Optional[str] = None, attachments: Optional[List[BaseAttachment]] = None
    ) -> List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]]:
        """Format input data and optional system prompt into messages."""
        messages: List[Dict[str, Any]] = []

        if system_prompt:
            system_message: Dict[str, Any] = {"role": "system", "content": system_prompt}
            messages.append(system_message)

        if isinstance(input_data, str):
            user_message: Dict[str, Any] = {"role": "user", "content": input_data}
            messages.append(user_message)
        elif isinstance(input_data, dict):
            messages.append(input_data)
        elif isinstance(input_data, list):
            messages.extend(input_data)

        # Add attachments if provided
        if attachments:
            # Get attachments by type
            image_attachments = [att for att in attachments if isinstance(att, ImageAttachment)]
            file_attachments = [att for att in attachments if isinstance(att, FileAttachment)]

            # Handle file attachments
            if file_attachments:
                for file_att in file_attachments:
                    # Create a separate message for each file attachment
                    file_message: Dict[str, Any] = {
                        "role": "assistant",
                        "content": None,
                        "file_ids": [getattr(file_att, "file_id", None)] if hasattr(file_att, "file_id") else [],
                    }
                    messages.append(file_message)

            # Handle image attachments in the last user message
            if image_attachments:
                # Find or create the last user message
                last_user_msg_idx = -1
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i]["role"] == "user":
                        last_user_msg_idx = i
                        break

                if last_user_msg_idx == -1:
                    user_message = {"role": "user", "content": ""}
                    messages.append(user_message)
                    last_user_msg_idx = len(messages) - 1

                # Update the last user message with images
                msg = messages[last_user_msg_idx]
                content = msg["content"]
                if not isinstance(content, str):
                    content = str(content)

                # Create the updated message with images
                updated_message: Dict[str, Any] = {"role": "user", "content": [{"type": "text", "text": content}]}

                # Add image content
                for img in image_attachments:
                    if isinstance(img.content, bytes):
                        # Convert bytes to base64
                        img_content = base64.b64encode(img.content).decode("utf-8")
                    else:
                        img_content = str(img.content)

                    updated_message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img.media_type.value};base64,{img_content}",
                                "detail": "auto",
                            },
                        }
                    )

                messages[last_user_msg_idx] = updated_message

        return cast(List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]], messages)

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
                metadata=self._create_response_metadata(),
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
