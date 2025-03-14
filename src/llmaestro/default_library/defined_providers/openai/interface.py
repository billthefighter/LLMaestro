"""OpenAI interface implementation."""

import logging
from typing import Any, Dict, List, Optional, Union, AsyncIterator, TYPE_CHECKING, cast, overload, BinaryIO, Type
import base64
import json
import asyncio
import httpx

from openai import AsyncOpenAI, BadRequestError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.file_object import FileObject
from openai.types import FilePurpose
from pydantic import BaseModel

from llmaestro.core.models import LLMResponse, TokenUsage, ContextMetrics
from llmaestro.llm.interfaces.base import BaseLLMInterface, ToolInputType, ProcessedToolType
from llmaestro.core.attachments import BaseAttachment, AttachmentConverter, FileAttachment
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.prompts.types import PromptMetadata
from llmaestro.llm.responses import ResponseFormatType, ResponseFormat, StructuredOutputConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class OpenAIInterface(BaseLLMInterface):
    """OpenAI-specific implementation of the LLM interface."""

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize OpenAI client with timeouts
        timeout_config = (
            httpx.Timeout(
                connect=self.socket_timeout,
                read=self.request_timeout,
                write=self.request_timeout,
                pool=self.socket_timeout,
            )
            if self.socket_timeout and self.request_timeout
            else None
        )

        self.client = AsyncOpenAI(api_key=self.credentials.key if self.credentials else None, timeout=timeout_config)
        # Models that require max_completion_tokens instead of max_tokens
        self.models_requiring_completion_tokens = ["o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o3-preview"]

    def _post_super_init(self, **data):
        """Initialize OpenAI-specific components."""
        if not self.state:
            logger.warning("No state provided in post-init")
            return
        logger.debug("OpenAI-specific initialization completed")

    @property
    def model_family(self) -> str:
        """Get the model family."""
        return "openai"

    @property
    def supports_structured_output(self) -> bool:
        """Whether this interface supports native structured output."""
        return self._check_capability("supports_json_mode")

    @property
    def supports_json_schema(self) -> bool:
        """Whether this interface supports JSON schema validation."""
        return self._check_capability("supports_json_mode")

    async def initialize(self) -> None:
        """Initialize async components of the interface."""
        await super().initialize()
        # Additional OpenAI-specific initialization can go here

    @overload
    async def process(
        self,
        prompt: str,
        variables: Optional[Dict[str, Any]] = None,
        tools: None = None,
    ) -> LLMResponse:
        ...

    @overload
    async def process(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
        tools: Optional[List[ToolInputType]] = None,
    ) -> LLMResponse:
        ...

    async def _prepare_request(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Prepare common request parameters for both streaming and non-streaming calls.

        Returns a dictionary with:
        - messages: Formatted messages
        - model: Model name
        - temperature: Temperature (None if not supported)
        - max_tokens: Max tokens
        - processed_tools: Processed tools from the prompt
        - response_format: Response format configuration
        """
        # Convert string prompt to MemoryPrompt if needed
        if isinstance(prompt, str):
            prompt = MemoryPrompt(
                name="direct_prompt",
                description="Direct string prompt",
                system_prompt="",
                user_prompt=prompt,
                metadata=PromptMetadata(
                    type="direct_input",
                    tags=[],
                ),
                expected_response=ResponseFormat(
                    format=ResponseFormatType.TEXT,
                    response_schema=None,
                    retry_config=None,
                ),
            )

        # Validate credentials
        self.validate_credentials()
        if not self.credentials:
            raise ValueError("No credentials provided")

        # Render the prompt with optional variables
        system_prompt, user_prompt, attachment_dicts, prompt_tools = prompt.render(**(variables or {}))

        # Convert dictionary attachments to BaseAttachment objects
        attachments = [AttachmentConverter.from_dict(att) for att in attachment_dicts]

        # Format messages
        messages = await self._format_messages(
            input_data=user_prompt, system_prompt=system_prompt, attachments=attachments
        )

        if not self.state:
            raise ValueError("No state provided, a LLMState must be provided to the LLMInterface")

        # Get model configuration
        model_name = self.state.profile.name
        max_tokens = self.state.runtime_config.max_tokens

        # Get temperature if supported by the model
        temperature = None
        if self._check_capability("supports_temperature"):
            temperature = self.state.runtime_config.temperature

        # Process prompt-level tools
        # Cast the tools from prompt.render() to List[ToolInputType] since we know they're compatible
        prompt_tools_input = cast(Optional[List[ToolInputType]], prompt_tools)
        processed_tools = self._process_tools(prompt_tools_input) if prompt_tools_input else []

        # Configure structured output if specified in prompt
        response_format = None
        if prompt.expected_response:
            response_format = prompt.expected_response.get_structured_output_config()

        return {
            "messages": messages,
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "processed_tools": processed_tools,
            "response_format": response_format,
        }

    async def _prepare_completion_params(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
        tools: Optional[List[ToolInputType]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Prepare all parameters needed for chat completion.

        This method handles all the common logic for process, batch_process, and stream methods:
        1. Prepares the request parameters
        2. Checks model capabilities
        3. Processes and merges tools

        Args:
            prompt: The prompt to process
            variables: Optional variables for prompt rendering
            tools: Optional runtime tools
            stream: Whether this is for streaming (affects capability checks)

        Returns:
            Dictionary with all parameters needed for _create_chat_completion
        """
        # Log initial state
        logger.debug(f"Preparing completion for prompt: {prompt}")
        logger.debug(f"Variables: {variables}")
        logger.debug(f"Tools provided: {tools}")
        if isinstance(prompt, BasePrompt):
            logger.debug(f"Prompt tools: {prompt.tools}")

        # Get prompt-level tools and messages through _prepare_request
        request_params = await self._prepare_request(prompt, variables)

        messages = request_params["messages"]
        model_name = request_params["model"]
        temperature = request_params["temperature"]
        max_tokens = request_params["max_tokens"]
        prompt_tools = request_params["processed_tools"]
        response_format = request_params["response_format"]

        logger.debug(f"Prepared request with model: {model_name}")
        logger.debug(f"Prompt tools after prepare: {prompt_tools}")

        # Check capability requirements in strict mode
        # This will raise exceptions for unsupported capabilities
        capability_kwargs = {"strict": True, "prompt": prompt, "runtime_tools": tools, "temperature": temperature}

        # Add stream parameter only if this is for streaming
        if stream:
            capability_kwargs["stream"] = True

        self._check_model_capabilities(**capability_kwargs)

        # Process and merge tools only if supported
        final_tools = None
        if prompt_tools or tools:
            # The capability check above will have already raised an exception if tools aren't supported
            logger.debug("Processing tools...")
            final_tools = await self._prepare_tools(prompt_tools, tools)
            logger.debug(f"Final processed tools: {[t.name for t in final_tools] if final_tools else None}")

        # Prepare parameters for chat completion
        completion_params = {
            "messages": messages,
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": final_tools if not response_format else None,  # Don't use tools if using response format
            "response_format": response_format,
        }

        # Remove None values
        return {k: v for k, v in completion_params.items() if v is not None}

    async def process(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
        tools: Optional[List[ToolInputType]] = None,
    ) -> LLMResponse:
        """Process input using OpenAI's API with tool support."""
        try:
            # Prepare all parameters for chat completion
            completion_params = await self._prepare_completion_params(
                prompt=prompt, variables=variables, tools=tools, stream=False
            )

            # Create chat completion
            logger.debug("Creating chat completion...")
            response = await self._create_chat_completion(**completion_params)

            if not isinstance(response, ChatCompletion):
                error_msg = "Expected ChatCompletion response for non-streaming request"
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.debug(f"Raw OpenAIResponse: {response}")

            # Get the response content
            llm_response = await self._handle_response(response)
            logger.debug(f"Final LLMResponse content: {llm_response.content[:200]}...")

            # If we have a Pydantic model in the prompt's expected response, validate against it
            if isinstance(prompt, BasePrompt) and prompt.expected_response and prompt.expected_response.pydantic_model:
                try:
                    # Validate the response content against the Pydantic model
                    model = prompt.expected_response.pydantic_model
                    model.model_validate_json(llm_response.content)
                except Exception as e:
                    logger.error(f"Failed to validate response against Pydantic model: {str(e)}")
                    return self._handle_error(e)

            return llm_response
        except Exception as e:
            logger.error(f"Error in process: {str(e)}", exc_info=True)
            return self._handle_error(e)

    async def batch_process(
        self,
        prompts: List[Union[BasePrompt, str]],
        variables: Optional[List[Optional[Dict[str, Any]]]] = None,
        batch_size: Optional[int] = None,
        tools: Optional[List[ToolInputType]] = None,
    ) -> List[LLMResponse]:
        """Process multiple prompts in a batch with tool support."""
        # Check if parallel requests are supported
        self._check_model_capabilities(strict=True, parallel_requests=True)

        # Ensure variables list matches prompts length if provided
        if variables is not None and len(variables) != len(prompts):
            raise ValueError("Number of variable sets must match number of prompts")

        # Process each prompt
        results = []
        for i, prompt in enumerate(prompts):
            try:
                # Use corresponding variables if provided, otherwise None
                prompt_vars = variables[i] if variables is not None else None

                # Prepare all parameters for chat completion
                completion_params = await self._prepare_completion_params(
                    prompt=prompt, variables=prompt_vars, tools=tools, stream=False
                )

                # Create chat completion
                response = await self._create_chat_completion(**completion_params)

                if not isinstance(response, ChatCompletion):
                    raise ValueError("Expected ChatCompletion response for non-streaming request")
                logger.debug(f"Raw OpenAIResponse for prompt {i}: {response}")
                result = await self._handle_response(response)
            except Exception as e:
                result = self._handle_error(e)

            results.append(result)

        return results

    async def stream(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
        tools: Optional[List[ToolInputType]] = None,
    ) -> AsyncIterator[LLMResponse]:
        """Stream responses from OpenAI's API one token at a time with tool support."""
        try:
            # Prepare all parameters for chat completion
            completion_params = await self._prepare_completion_params(
                prompt=prompt, variables=variables, tools=tools, stream=True
            )

            # Add streaming parameter
            completion_params["stream"] = True

            # Make the API call
            stream = await self.client.chat.completions.create(**completion_params)

            async for chunk in stream:
                try:
                    if chunk.choices and chunk.choices[0].delta and hasattr(chunk.choices[0].delta, "content"):
                        delta = chunk.choices[0].delta
                        # Initialize content as empty string
                        content = ""

                        # Process content based on type
                        delta_content = delta.content

                        if delta_content is None:
                            # No content in this chunk
                            continue
                        elif isinstance(delta_content, str):
                            # Simple string content
                            content = delta_content
                        else:
                            # For any other type (including lists), convert to string safely
                            try:
                                # Try to extract text from list content if it's a list
                                if isinstance(delta_content, list):
                                    text_parts = []
                                    # Use a simple for loop with explicit type checking
                                    for i in range(len(delta_content)):
                                        part = delta_content[i]
                                        if isinstance(part, dict) and part.get("type") == "text":
                                            text_parts.append(part.get("text", ""))

                                    # Join text parts if we found any
                                    if text_parts:
                                        content = "".join(text_parts)
                                    else:
                                        content = str(delta_content)
                                else:
                                    # For any other type, convert to string
                                    content = str(delta_content)
                            except (TypeError, AttributeError, IndexError) as e:
                                # Fallback for any errors during content extraction
                                logger.warning(f"Error extracting content from stream chunk: {e}")
                                content = str(delta_content)

                        if content:  # Only yield if there's content
                            yield LLMResponse(
                                content=content,
                                success=True,
                                token_usage=TokenUsage(
                                    completion_tokens=1,
                                    prompt_tokens=0,
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

    async def _format_messages(
        self, input_data: Any, system_prompt: Optional[str] = None, attachments: Optional[List[BaseAttachment]] = None
    ) -> List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]]:
        """Format input data and optional system prompt into messages."""
        # Use the base class implementation which already has capability checking
        messages = await super()._format_messages(input_data, system_prompt, attachments)

        # Cast to the specific OpenAI type
        return cast(List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]], messages)

    async def _handle_attachments(
        self, messages: List[Dict[str, Any]], attachments: List[BaseAttachment]
    ) -> List[Dict[str, Any]]:
        """Handle OpenAI-specific attachment formatting."""
        # Get attachments by type
        image_attachments = [att for att in attachments if att.media_type.is_image()]
        file_attachments = [att for att in attachments if not att.media_type.is_image()]

        # Handle file attachments
        if file_attachments:
            for att in file_attachments:
                # Cast to FileAttachment if possible
                file_att = cast(FileAttachment, att) if isinstance(att, FileAttachment) else None

                # Skip if not a FileAttachment
                if not file_att:
                    logger.warning(f"Skipping non-FileAttachment: {att}")
                    continue

                # If no file_id, try to upload the file
                if not file_att.file_id and isinstance(file_att.content, str):
                    try:
                        # Decode base64 content
                        file_content = base64.b64decode(file_att.content)
                        # Upload the file
                        file_obj = await self.upload_file(file_content)
                        # Update the attachment with the file ID
                        file_att.file_id = file_obj.id
                    except Exception as e:
                        logger.error(f"Failed to upload file: {str(e)}")
                        continue

                # Create a separate message for each file attachment with file_id
                if file_att.file_id:
                    file_message: Dict[str, Any] = {
                        "role": "user",
                        "content": f"Processing file: {file_att.file_name}",
                        "file_ids": [file_att.file_id],
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
            content = msg.get("content", "")

            # Handle different content types
            text_content = ""
            if isinstance(content, list):
                # Extract text from content parts that are in a list
                text_parts = []
                # Check type first, then check if non-empty
                if isinstance(content, list) and content:
                    try:
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                        text_content = "".join(text_parts) if text_parts else str(content)
                    except TypeError:
                        # Fallback if iteration fails
                        text_content = str(content)
                else:
                    text_content = str(content)
            elif content is not None:
                # Convert to string
                text_content = str(content)

            # Create the updated message with images
            updated_message: Dict[str, Any] = {"role": "user", "content": [{"type": "text", "text": text_content}]}

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

        return messages

    async def _handle_tool_call(self, message: ChatCompletionMessage) -> str:
        """Handle OpenAI's function calling response."""
        if not message.tool_calls:
            return message.content or ""

        results = []
        for tool_call in message.tool_calls:
            if not tool_call.function:
                continue

            # Get the tool from our processed tools
            tool_name = tool_call.function.name
            logger.debug(f"Looking for tool: {tool_name}")
            logger.debug(f"Available tools: {list(self.available_tools.keys())}")

            tool = self.get_cached_tool(tool_name)

            if not tool:
                error_msg = f"Error: Tool '{tool_name}' not found"
                logger.error(error_msg)
                results.append(error_msg)
                continue

            try:
                # Parse arguments
                args = json.loads(tool_call.function.arguments)
                logger.debug(f"Executing {tool_name} with args: {args}")

                # Execute and format result
                result = await self._handle_tool_execution(tool, args)
                logger.debug(f"Tool execution result: {result}")
                results.append(result)

            except json.JSONDecodeError:
                error_msg = f"Error: Invalid arguments for tool '{tool_name}'"
                logger.error(error_msg)
                results.append(error_msg)
            except Exception as e:
                error_msg = f"Error executing '{tool_name}': {str(e)}"
                logger.error(error_msg)
                results.append(error_msg)

        # Combine all results
        return "\n\n".join(results)

    async def _handle_response(self, response: ChatCompletion) -> LLMResponse:
        """Handle the response from OpenAI's API."""
        try:
            # Get the first choice message
            if not response.choices:
                return self._handle_error(ValueError("No response choices available"))

            message = response.choices[0].message
            if not message:
                return self._handle_error(ValueError("No message in response choice"))

            # Initialize content as empty string
            content = ""

            # Extract content based on message type
            if hasattr(message, "function_call") and message.function_call:
                # For structured output using function calling
                content = message.function_call.arguments
            elif message.tool_calls:
                # For regular tool calls
                if message.tool_calls and message.tool_calls[0].function:
                    # Ensure the arguments are valid JSON
                    try:
                        # Parse and re-serialize to ensure valid JSON
                        args = json.loads(message.tool_calls[0].function.arguments)
                        content = json.dumps(args)
                    except json.JSONDecodeError:
                        # If parsing fails, use the raw arguments
                        content = message.tool_calls[0].function.arguments
            else:
                # Handle regular message content
                message_content = message.content

                # Process content based on type
                if message_content is None:
                    # Keep content as empty string
                    content = ""
                elif isinstance(message_content, str):
                    # Simple string content
                    content = message_content
                else:
                    # For any other type (including lists), convert to string safely
                    try:
                        # Try to extract text from list content if it's a list
                        if isinstance(message_content, list):
                            text_parts = []
                            # Use a simple for loop with explicit type checking
                            for i in range(len(message_content)):
                                part = message_content[i]
                                if isinstance(part, dict) and part.get("type") == "text":
                                    text_parts.append(part.get("text", ""))

                            # Join text parts if we found any
                            if text_parts:
                                content = "".join(text_parts)
                            else:
                                content = str(message_content)
                        else:
                            # For any other type, convert to string
                            content = str(message_content)
                    except (TypeError, AttributeError, IndexError) as e:
                        # Fallback for any errors during content extraction
                        logger.warning(f"Error extracting content: {e}")
                        content = str(message_content)

            # Create token usage info
            usage = response.usage or TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
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

    async def _handle_timeout(self, coro, timeout, error_message):
        """Handle a coroutine with a timeout."""
        try:
            return await asyncio.wait_for(coro, timeout)
        except asyncio.TimeoutError:
            logger.error(error_message)
            raise

    def _requires_completion_tokens(self, model_id: str) -> bool:
        """Check if a model requires max_completion_tokens instead of max_tokens.

        Args:
            model_id: The model ID to check

        Returns:
            bool: True if the model requires max_completion_tokens, False otherwise
        """
        return any(model_id.startswith(prefix) for prefix in self.models_requiring_completion_tokens)

    def _format_tools_for_provider(self, tools: List[ProcessedToolType]) -> List[Dict[str, Any]]:
        """Convert processed tools into OpenAI's function calling format.

        Args:
            tools: List of processed ToolParams objects

        Returns:
            List of tool definitions in OpenAI's format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        **tool.parameters,
                        "additionalProperties": False,
                    },
                },
            }
            for tool in tools
        ]

    def _has_pattern_validators(self, model: Type[BaseModel]) -> bool:
        """Check if a Pydantic model has pattern validators which are not supported by OpenAI.

        Args:
            model: The Pydantic model to check

        Returns:
            bool: True if the model has pattern validators, False otherwise
        """
        try:
            # Get the model schema
            schema = model.model_json_schema()

            # Check if any properties have pattern validators
            if "properties" in schema:
                for prop_name, prop_schema in schema["properties"].items():
                    if "pattern" in prop_schema:
                        logger.debug(f"Found pattern validator in property '{prop_name}': {prop_schema['pattern']}")
                        return True

            return False
        except Exception as e:
            logger.warning(f"Error checking for pattern validators: {e}")
            return False  # Assume no pattern validators if we can't check

    async def _create_chat_completion(
        self,
        messages: List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]],
        model: str,
        max_tokens: int,
        temperature: Optional[float] = None,
        tools: Optional[List[ProcessedToolType]] = None,
        response_format: Optional[StructuredOutputConfig] = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion with OpenAI.

        This method supports two paths for structured output:
        1. Direct Pydantic Parsing (Experimental):
           - Uses OpenAI's beta.chat.completions.parse endpoint
           - Designed for direct Pydantic model validation
           - Only available for models that support direct Pydantic parsing
           - Requires the model to have supports_direct_pydantic_parse=True

        2. Standard JSON Object (Stable):
           - Uses standard chat completions with type: "json_object"
           - Includes schema validation in system prompt
           - Works consistently across all models
           - Recommended for production use when direct parsing is not available

        Args:
            messages: List of messages to send to the API
            model: Model ID to use
            temperature: Temperature parameter (None if not supported)
            max_tokens: Maximum tokens to generate
            tools: Optional list of tools to include
            response_format: Optional response format configuration
            **kwargs: Additional keyword arguments to pass to the API

        Raises:
            TimeoutError: If the request exceeds the configured timeout
            Exception: For other API errors
        """
        logger.debug(f"Creating chat completion with model: {model}")
        logger.debug(f"Response format configuration: {response_format}")
        logger.debug(f"Number of messages: {len(messages)}")

        # Log first message content without attachments
        if messages:
            first_msg = messages[0]
            content_preview = "<structured_content>"
            msg_content = first_msg.get("content")

            if isinstance(msg_content, str):
                content_preview = msg_content
                if content_preview and len(content_preview) > 200:
                    content_preview = content_preview[:200] + "..."

            logger.debug(f"First message preview: {content_preview}")

        # Set up base kwargs including messages
        base_kwargs = {
            "model": model,
            "messages": messages,
            **kwargs,
        }

        # Only include temperature if it's provided (model supports it)
        if temperature is not None:
            base_kwargs["temperature"] = temperature

        # First try with max_tokens
        try:
            # Add max_tokens parameter
            base_kwargs["max_tokens"] = max_tokens
            logger.debug(f"Trying with max_tokens={max_tokens}")

            # Handle tools if provided
            if tools:
                openai_tools = self._format_tools_for_provider(tools)
                base_kwargs["tools"] = openai_tools
                base_kwargs["tool_choice"] = "auto"  # Let OpenAI decide when to use tools
                logger.debug(f"Added {len(openai_tools)} tools to request")

            # Handle response format configuration
            if response_format:
                pydantic_model = response_format.pydantic_model
                logger.debug(f"Pydantic model from response_format: {pydantic_model}")
                logger.debug(f"Full response_format dict: {response_format}")

                # Check if model supports direct Pydantic parsing
                supports_direct_parsing = self._check_capability("supports_direct_pydantic_parse")
                logger.debug(f"Model supports direct Pydantic parsing: {supports_direct_parsing}")

                if pydantic_model is not None and supports_direct_parsing:
                    # Check for pattern validators which are not supported by OpenAI
                    has_patterns = self._has_pattern_validators(pydantic_model)
                    if has_patterns:
                        logger.warning(
                            "Pydantic model '{}' contains pattern validators "
                            "which are not supported by OpenAI's direct Pydantic integration. "
                            "Falling back to standard JSON object format.".format(pydantic_model.__name__)
                        )
                    else:
                        try:
                            logger.debug(f"Using direct Pydantic parsing with model: {pydantic_model}")
                            logger.info(f"Using Pydantic model directly: {pydantic_model.__name__}")
                            # Remove response_format from base_kwargs to avoid conflicts
                            base_kwargs.pop("response_format", None)
                            # Log final kwargs without the full messages
                            debug_kwargs = {
                                k: v if k != "messages" else f"<{len(v)} messages>" for k, v in base_kwargs.items()
                            }
                            logger.debug(f"Final parse endpoint configuration: {debug_kwargs}")
                            logger.debug(f"Calling parse with response_format={pydantic_model}")
                            final_kwargs = {**base_kwargs, "response_format": pydantic_model}
                            final_debug_kwargs = {
                                k: v if k != "messages" else f"<{len(v)} messages>" for k, v in final_kwargs.items()
                            }
                            logger.debug(f"Final kwargs: {final_debug_kwargs}")

                            # Call parse endpoint with timeout
                            return await self._handle_timeout(
                                self.client.beta.chat.completions.parse(**final_kwargs),
                                self.request_timeout,
                                "OpenAI parse endpoint request timed out",
                            )
                        except TimeoutError:
                            raise
                        except Exception as e:
                            logger.error(f"Direct Pydantic parsing failed: {e}")
                            logger.warning("Falling back to standard JSON object format")
                            # Fall through to standard format

                # Standard JSON object format (stable)
                base_kwargs["response_format"] = {"type": "json_object"}
                if response_format.effective_schema:
                    # Add schema validation in the system prompt
                    schema_str = json.dumps(response_format.effective_schema, indent=2)
                    system_msg = next((m for m in messages if m["role"] == "system"), None)
                    if system_msg:
                        system_msg["content"] = (
                            f"{system_msg['content']}\n\n"
                            f"Validate the response against this JSON schema:\n"
                            f"```json\n{schema_str}\n```"
                        )
                    else:
                        messages.insert(
                            0,
                            {
                                "role": "system",
                                "content": (
                                    f"Validate the response against this JSON schema:\n" f"```json\n{schema_str}\n```"
                                ),
                            },
                        )
                    base_kwargs["messages"] = messages

            # Log final kwargs without the full messages
            debug_kwargs = {k: v if k != "messages" else f"<{len(v)} messages>" for k, v in base_kwargs.items()}
            logger.debug(f"Final request configuration: {debug_kwargs}")

            # Call create endpoint with timeout
            return await self._handle_timeout(
                self.client.chat.completions.create(**base_kwargs),
                self.request_timeout,
                "OpenAI chat completion request timed out",
            )

        except BadRequestError as e:
            # Check if the error is about max_tokens vs max_completion_tokens
            if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                logger.debug(f"Model {model} requires max_completion_tokens, retrying with correct parameter")
                # Replace max_tokens with max_completion_tokens
                base_kwargs.pop("max_tokens")
                base_kwargs["max_completion_tokens"] = max_tokens
                # Add this model to our list for future reference
                if model not in self.models_requiring_completion_tokens:
                    self.models_requiring_completion_tokens.append(model)
                # Retry the request
                return await self._handle_timeout(
                    self.client.chat.completions.create(**base_kwargs),
                    self.request_timeout,
                    "OpenAI chat completion request timed out",
                )
            else:
                # If it's a different error, re-raise it
                raise

    def _create_response_metadata(self, is_streaming: bool = False, is_partial: bool = False) -> Dict[str, Any]:
        """Create common metadata for LLMResponse."""
        return {
            "model": self.state.profile.name if self.state else "unknown",
            "is_streaming": is_streaming,
            "is_partial": is_partial,
        }

    async def parse_structured_response(
        self,
        response: str,
        expected_format: ResponseFormat,
    ) -> Any:
        """Parse a structured response from OpenAI.

        Args:
            response: The raw response string from OpenAI
            expected_format: The expected response format

        Returns:
            Parsed response object (may be a Pydantic model or dict)
        """
        try:
            # Check capabilities
            supports_json_mode = self._check_capability("supports_json_mode")
            supports_direct_parsing = self._check_capability("supports_direct_pydantic_parse")

            if not supports_json_mode:
                logger.warning(
                    f"Model '{self.state.profile.name if self.state else 'unknown'}' does not support JSON mode. "
                    "Falling back to manual parsing."
                )

            # Log the raw response for debugging
            logger.debug(f"Raw response before parsing: {response}")

            # Handle function call responses
            if isinstance(response, dict):
                if "function_call" in response:
                    function_call = cast(Dict[str, Any], response.get("function_call", {}))
                    response = function_call.get("arguments", "")
                elif "arguments" in response:
                    response = cast(Dict[str, Any], response).get("arguments", "")

            # Clean up the response string to ensure valid JSON
            if isinstance(response, str):
                response = response.strip()

                # Remove any markdown or code block markers
                if response.startswith("```"):
                    lines = response.split("\n")
                    # Remove first and last lines (code block markers)
                    response = "\n".join(lines[1:-1])
                    # If first line is a language identifier (e.g. 'json'), remove it
                    if response.startswith("json"):
                        response = "\n".join(response.split("\n")[1:])

                # Clean up the response
                response = response.strip()
                logger.debug(f"Cleaned response before parsing: {response}")

            # Parse the JSON
            try:
                if isinstance(response, str):
                    # If response is already a valid JSON object, parse it directly
                    if response.startswith("{") and response.endswith("}"):
                        parsed_data = json.loads(response)
                    else:
                        # If the response has content but is missing braces, add them
                        if not response.startswith("{"):
                            response = "{"

                        # Clean up any trailing commas and ensure proper JSON formatting
                        response = response.rstrip(",")
                        if not response.endswith("}"):
                            response = response + "}"

                        # Clean up any extra whitespace between key-value pairs
                        response = " ".join(response.split())
                        logger.debug(f"Final response before parsing: {response}")
                        parsed_data = json.loads(response)
                else:
                    parsed_data = response

                # If we have a Pydantic model, validate against it
                if expected_format.pydantic_model and supports_direct_parsing:
                    return expected_format.pydantic_model.model_validate(parsed_data)
                elif expected_format.pydantic_model:
                    # Fall back to JSON validation if direct parsing is not supported
                    logger.debug("Direct Pydantic parsing not supported, using JSON validation")
                    return expected_format.pydantic_model.model_validate(parsed_data)

                return parsed_data

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                logger.error(f"Response content: {response}")
                raise

        except Exception as e:
            logger.error(f"Error parsing structured response: {str(e)}")
            logger.error(f"Raw response: {response}")
            raise
