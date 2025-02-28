"""OpenAI interface implementation."""

import logging
from typing import Any, Dict, List, Optional, Union, AsyncIterator, TYPE_CHECKING, cast, overload, BinaryIO
import base64
import json

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
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
from llmaestro.prompts.types import PromptMetadata, ResponseFormat
from llmaestro.llm.responses import ResponseFormatType
from llmaestro.prompts.tools import ToolParams

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
    ]:
        """Prepare common request parameters for both streaming and non-streaming calls.

        Returns a tuple of:
        - Formatted messages
        - Model name
        - Temperature
        - Max tokens
        - Processed tools from the prompt
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

        return messages, model_name, temperature, max_tokens, processed_tools

    @property
    def supports_structured_output(self) -> bool:
        """Whether this interface supports native structured output."""
        return True

    @property
    def supports_json_schema(self) -> bool:
        """Whether this interface supports JSON schema validation."""
        return True

    def configure_structured_output(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure structured output settings for OpenAI.

        Args:
            config: Configuration from ResponseFormat.get_structured_output_config()

        Returns:
            Dict of OpenAI-specific configuration
        """
        output_config: Dict[str, Any] = {}

        # Configure response format for JSON output
        if config.get("format") in ["json", "json_schema"]:
            output_config["response_format"] = {"type": "json_object"}

            # Add schema validation to system message if provided
            if config.get("schema"):
                schema = json.loads(config["schema"]) if isinstance(config["schema"], str) else config["schema"]

                # Configure function calling for schema validation
                output_config["functions"] = [
                    {
                        "name": "output_json",
                        "description": "Output JSON data according to the schema",
                        "parameters": schema,
                    }
                ]
                output_config["function_call"] = {"name": "output_json"}

                # Add schema validation instructions to system message
                if "messages" in config and len(config["messages"]) > 0:
                    system_msg = config["messages"][0]
                    if isinstance(system_msg, dict) and "content" in system_msg:
                        system_msg["content"] = (
                            str(system_msg["content"])
                            + "\nPlease provide your response as a valid JSON object that conforms to the following schema:\n"
                            + json.dumps(schema, indent=2)
                            + "\nEnsure your response is a complete, well-formed JSON object with all required fields."
                            + "\nYour response should be a single JSON object enclosed in curly braces {}."
                        )

                # Add response format instructions
                output_config["response_format"] = {"type": "json_object", "schema": schema}

        return output_config

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
            # Clean up the response string to ensure valid JSON
            response = response.strip()

            # Remove any markdown or code block markers
            if response.startswith("```") and response.endswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])  # Remove first and last lines
                if response.startswith("json"):
                    response = "\n".join(response.split("\n")[1:])  # Remove language identifier

            # Clean up the response
            response = response.strip()

            # First try to parse as a complete JSON object
            try:
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

                    parsed_data = json.loads(response)

                # Handle function call responses
                if isinstance(parsed_data, dict):
                    if "arguments" in parsed_data:
                        parsed_data = json.loads(parsed_data["arguments"])
                    elif "function_call" in parsed_data and "arguments" in parsed_data["function_call"]:
                        parsed_data = json.loads(parsed_data["function_call"]["arguments"])

                # Validate against schema if provided
                if expected_format.response_schema:
                    schema = (
                        json.loads(expected_format.response_schema)
                        if isinstance(expected_format.response_schema, str)
                        else expected_format.response_schema
                    )
                    # TODO: Add schema validation here

                return parsed_data

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
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
        max_tokens: int,
        temperature: float,
        stream: bool = False,
        tools: Optional[List[ProcessedToolType]] = None,
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """Create a chat completion with the given parameters."""
        kwargs = {
            "messages": messages,
            "model": model,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        # Check model capabilities and filter unsupported parameters
        kwargs = self._check_model_capabilities(**kwargs)

        # Configure tools if provided
        if tools:
            # Convert tools to OpenAI's format
            openai_tools = self._format_tools_for_provider(tools)
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"  # Let OpenAI decide when to use tools

        return await self.client.chat.completions.create(**kwargs)

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
            messages, model_name, temperature, max_tokens, prompt_tools = await self._prepare_request(prompt, variables)

            # Process runtime tools if provided
            runtime_tools = self._process_tools(tools) if tools else []

            # Merge tools, giving precedence to runtime tools
            final_tools = prompt_tools + runtime_tools

            # Cache all tools for execution
            for tool in final_tools:
                self.available_tools[tool.name] = tool

            # Create chat completion
            response = await self._create_chat_completion(
                messages=messages,
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                tools=final_tools,
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
        tools: Optional[List[ToolInputType]] = None,
    ) -> List[LLMResponse]:
        """Process multiple prompts in a batch with tool support."""
        # Ensure variables list matches prompts length if provided
        if variables is not None and len(variables) != len(prompts):
            raise ValueError("Number of variable sets must match number of prompts")

        # Process runtime tools once for reuse
        runtime_tools = self._process_tools(tools) if tools else []

        # Process each prompt
        results = []
        for i, prompt in enumerate(prompts):
            # Use corresponding variables if provided, otherwise None
            prompt_vars = variables[i] if variables is not None else None
            result = await self.process(prompt, prompt_vars, tools=tools)
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
            messages, model_name, temperature, max_tokens, prompt_tools = await self._prepare_request(prompt, variables)

            # Process runtime tools
            runtime_tools = self._process_tools(tools) if tools else []

            # Merge tools
            final_tools = prompt_tools + runtime_tools

            # Cache tools
            for tool in final_tools:
                self.available_tools[tool.name] = tool

            stream = await self._create_chat_completion(
                messages=messages,
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                tools=final_tools,
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

        return cast(List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]], messages)

    async def _handle_tool_call(self, message: ChatCompletionMessage) -> str:
        """Handle OpenAI's function calling response.

        Args:
            message: The message containing tool calls

        Returns:
            Formatted response incorporating tool results
        """
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

            # Handle tool calls if present
            if message.tool_calls:
                content = await self._handle_tool_call(message)
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
