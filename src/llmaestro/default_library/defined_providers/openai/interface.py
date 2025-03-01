"""OpenAI interface implementation."""

import logging
from typing import Any, Dict, List, Optional, Union, AsyncIterator, TYPE_CHECKING, cast, overload, BinaryIO
import base64
import json
import asyncio
import httpx

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.file_object import FileObject
from openai.types import FilePurpose
from pydantic import BaseModel

from llmaestro.core.models import LLMResponse, TokenUsage, ContextMetrics
from llmaestro.llm.interfaces.base import BaseLLMInterface, ToolInputType, ProcessedToolType
from llmaestro.core.attachments import BaseAttachment, ImageAttachment, AttachmentConverter, FileAttachment
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.prompts.types import PromptMetadata
from llmaestro.llm.responses import ResponseFormatType, ResponseFormat, StructuredOutputConfig
from llmaestro.prompts.tools import ToolParams

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
        return True

    @property
    def supports_json_schema(self) -> bool:
        """Whether this interface supports JSON schema validation."""
        return True

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
    ) -> tuple[
        List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]],
        str,
        float,
        int,
        List[ProcessedToolType],
        Optional[StructuredOutputConfig],
    ]:
        """Prepare common request parameters for both streaming and non-streaming calls.

        Returns a tuple of:
        - Formatted messages
        - Model name
        - Temperature
        - Max tokens
        - Processed tools from the prompt
        - Response format configuration
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
        temperature = self.state.runtime_config.temperature

        # Process prompt-level tools
        # Cast the tools from prompt.render() to List[ToolInputType] since we know they're compatible
        prompt_tools_input = cast(Optional[List[ToolInputType]], prompt_tools)
        processed_tools = self._process_tools(prompt_tools_input) if prompt_tools_input else []

        # Configure structured output if specified in prompt
        response_format = None
        if prompt.expected_response:
            response_format = prompt.expected_response.get_structured_output_config()

        return messages, model_name, temperature, max_tokens, processed_tools, response_format

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
                if expected_format.pydantic_model:
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

    async def _create_chat_completion(
        self,
        messages: List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]],
        model: str,
        temperature: float,
        max_tokens: int,
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
            content_preview = (
                first_msg.get("content", "") if isinstance(first_msg.get("content"), str) else "<structured_content>"
            )
            if len(content_preview) > 200:
                content_preview = content_preview[:200] + "..."
            logger.debug(f"First message preview: {content_preview}")

        # Set up base kwargs including messages
        base_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        # Log kwargs without the full messages
        debug_kwargs = {**base_kwargs}
        debug_kwargs["messages"] = f"<{len(messages)} messages>"
        logger.debug(f"Base configuration: {debug_kwargs}")

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
            supports_direct_parsing = self.state and self.state.profile.capabilities.supports_direct_pydantic_parse
            logger.debug(f"Model supports direct Pydantic parsing: {supports_direct_parsing}")

            if pydantic_model is not None and supports_direct_parsing:
                try:
                    logger.debug(f"Using direct Pydantic parsing with model: {pydantic_model}")
                    # Remove response_format from base_kwargs to avoid conflicts
                    base_kwargs.pop("response_format", None)
                    # Log final kwargs without the full messages
                    debug_kwargs = {k: v if k != "messages" else f"<{len(v)} messages>" for k, v in base_kwargs.items()}
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
                    system_msg[
                        "content"
                    ] = f"{system_msg['content']}\n\nValidate the response against this JSON schema:\n```json\n{schema_str}\n```"
                else:
                    messages.insert(
                        0,
                        {
                            "role": "system",
                            "content": f"Validate the response against this JSON schema:\n```json\n{schema_str}\n```",
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

    def _create_response_metadata(self, is_streaming: bool = False, is_partial: bool = False) -> Dict[str, Any]:
        """Create common metadata for LLMResponse."""
        return {
            "model": self.state.profile.name if self.state else "unknown",
            "is_streaming": is_streaming,
            "is_partial": is_partial,
        }

    def _process_tools(self, tools: Optional[List[ToolInputType]]) -> List[ProcessedToolType]:
        """Process a list of tools into standardized ToolParams objects for OpenAI.

        This method converts various tool input types into ToolParams objects:
        - Functions are converted using ToolParams.from_function
        - Pydantic models are passed directly using OpenAI's pydantic_function_tool

        Args:
            tools: List of tools to process, can be functions, Pydantic models, or ToolParams

        Returns:
            List of processed ToolParams objects ready for use with the LLM
        """
        if not tools:
            return []

        processed_tools = []
        for tool in tools:
            if isinstance(tool, ProcessedToolType):
                # For existing ToolParams, keep as is
                processed_tools.append(tool)
            elif isinstance(tool, type) and issubclass(tool, BaseModel):
                # For Pydantic models, create a ToolParams with the model as the function
                schema = tool.model_json_schema()
                tool_params = ToolParams(
                    name=tool.__name__, description=tool.__doc__ or "", parameters=schema, return_type=tool, source=tool
                )
                processed_tools.append(tool_params)
            elif callable(tool):
                # For callables, create a ToolParams with the function
                tool_params = ToolParams.from_function(tool)
                processed_tools.append(tool_params)
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")

        return processed_tools

    async def process(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
        tools: Optional[List[ToolInputType]] = None,
    ) -> LLMResponse:
        """Process input using OpenAI's API with tool support."""
        try:
            # Get prompt-level tools and messages through _prepare_request
            messages, model_name, temperature, max_tokens, prompt_tools, response_format = await self._prepare_request(
                prompt, variables
            )

            # Process and merge tools
            final_tools = await self._prepare_tools(prompt_tools, tools)

            # Create chat completion
            response = await self._create_chat_completion(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=final_tools if not response_format else None,  # Don't use tools if using response format
                response_format=response_format,
            )

            if not isinstance(response, ChatCompletion):
                raise ValueError("Expected ChatCompletion response for non-streaming request")
            logger.debug(f"Raw OpenAIResponse: {response}")

            # Get the response content
            llm_response = await self._handle_response(response)

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
            return self._handle_error(e)

    async def batch_process(
        self,
        prompts: List[Union[BasePrompt, str]],
        variables: Optional[List[Optional[Dict[str, Any]]]] = None,
        batch_size: Optional[int] = None,
        tools: Optional[List[ToolInputType]] = None,
    ) -> List[LLMResponse]:
        """Process multiple prompts in a batch with tool support."""
        # Ensure variables list matches prompts length if provided
        if variables is not None and len(variables) != len(prompts):
            raise ValueError("Number of variable sets must match number of prompts")

        # Process each prompt
        results = []
        for i, prompt in enumerate(prompts):
            try:
                # Use corresponding variables if provided, otherwise None
                prompt_vars = variables[i] if variables is not None else None

                # Get prompt-level tools and messages through _prepare_request
                (
                    messages,
                    model_name,
                    temperature,
                    max_tokens,
                    prompt_tools,
                    response_format,
                ) = await self._prepare_request(prompt, prompt_vars)

                # Process and merge tools
                final_tools = await self._prepare_tools(prompt_tools, tools)

                # Create chat completion
                response = await self._create_chat_completion(
                    messages=messages,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=final_tools if not response_format else None,  # Don't use tools if using response format
                    response_format=response_format,
                )

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
            # Get prompt-level tools and messages
            messages, model_name, temperature, max_tokens, prompt_tools, response_format = await self._prepare_request(
                prompt, variables
            )

            # Process and merge tools
            final_tools = await self._prepare_tools(prompt_tools, tools)

            stream = await self._create_chat_completion(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=final_tools,
                response_format=response_format,
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
            messages = await self._handle_attachments(messages, attachments)

        return cast(List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]], messages)

    async def _handle_attachments(
        self, messages: List[Dict[str, Any]], attachments: List[BaseAttachment]
    ) -> List[Dict[str, Any]]:
        """Handle OpenAI-specific attachment formatting."""
        # Get attachments by type
        image_attachments = [att for att in attachments if isinstance(att, ImageAttachment)]
        file_attachments = [att for att in attachments if isinstance(att, FileAttachment)]

        # Handle file attachments
        if file_attachments:
            for file_att in file_attachments:
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
            message = response.choices[0].message if response.choices else None
            if not message:
                return self._handle_error(ValueError("No response choices available"))

            # Extract content from function call, tool calls, or regular message
            content = ""
            if hasattr(message, "function_call") and message.function_call:
                # For structured output using function calling
                content = message.function_call.arguments
            elif message.tool_calls:
                # For regular tool calls
                if message.tool_calls[0].function:
                    content = message.tool_calls[0].function.arguments
            else:
                content = message.content or ""

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
