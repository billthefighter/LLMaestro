"""Base interfaces for LLM providers."""
from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, AsyncIterator, Type, Callable, TypeVar, Awaitable
import asyncio

from pydantic import BaseModel, Field, ConfigDict

from llmaestro.core.conversations import ConversationContext
from llmaestro.core.models import LLMResponse, TokenUsage
from llmaestro.llm.credentials import APIKey
from llmaestro.prompts.base import BasePrompt
from llmaestro.llm.enums import MediaType
from llmaestro.llm.responses import ResponseFormat
from .tokenizers import BaseTokenizer
from llmaestro.llm.models import LLMState  # Direct import instead of TYPE_CHECKING
from llmaestro.tools.core import ToolParams
from llmaestro.core.models import ContextMetrics
from llmaestro.core.attachments import BaseAttachment

logger = logging.getLogger(__name__)

# Type aliases
ToolInputType = Union[Callable[..., Any], Type[BaseModel], ToolParams]
ProcessedToolType = ToolParams

# Add type variables for better type hinting
T = TypeVar("T")


@dataclass
class ImageInput:
    """Input image data for vision models."""

    content: Union[str, bytes]  # Base64 encoded string or raw bytes
    media_type: MediaType  # Media type of the image
    file_name: Optional[str] = None

    def __init__(
        self,
        content: Union[str, bytes],
        media_type: Union[str, MediaType] = MediaType.JPEG,
        file_name: Optional[str] = None,
    ):
        self.content = content
        self.media_type = media_type if isinstance(media_type, MediaType) else MediaType.from_mime_type(media_type)
        self.file_name = file_name

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "ImageInput":
        """Create an ImageInput from a file path."""
        path = Path(file_path)
        with open(path, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        media_type = MediaType.from_file_extension(str(path))  # Convert Path to str
        return cls(content=content, media_type=media_type, file_name=path.name)

    @classmethod
    def from_bytes(
        cls, data: bytes, media_type: Union[str, MediaType], file_name: Optional[str] = None
    ) -> "ImageInput":
        """Create an ImageInput from bytes."""
        content = base64.b64encode(data).decode()
        media_type_enum = media_type if isinstance(media_type, MediaType) else MediaType.from_mime_type(media_type)
        return cls(content=content, media_type=media_type_enum, file_name=file_name)


class BaseLLMInterface(BaseModel, ABC):
    """Base class for LLM interfaces.

    This class provides the foundation for implementing LLM provider interfaces.
    It handles common functionality like message formatting, token counting,
    and tool management.

    Tool Management:
    --------------
    Tools can be provided at three levels, in order of precedence (highest to lowest):

    1. Runtime Tools (process method argument):
       - Highest precedence
       - Used for request-specific tools
       - Overrides conflicting tools from other levels

    2. Prompt-Level Tools (BasePrompt.tools):
       - Medium precedence
       - Defined as part of the prompt template
       - Overrides interface-level tools

    3. Interface-Level Tools (available_tools cache):
       - Lowest precedence
       - Shared across all requests using this interface
       - Available to all prompts

    Timeout Configuration:
    -------------------
    Timeouts can be configured at three levels:
    1. Request Timeout: Maximum time for a single request (default: 30s)
    2. Total Timeout: Maximum time for all retries (default: 90s)
    3. Socket Timeout: Maximum time for socket operations (default: 10s)

    All timeouts are in seconds and can be disabled by setting to None.

    Tool Processing Flow:
    ------------------
    1. Tools are processed through _process_tools() into standardized ToolParams
    2. Tools are cached in available_tools to avoid reprocessing
    3. Provider-specific formatting is handled by _format_tools_for_provider()
    4. Tool execution is managed by _handle_tool_execution()

    Implementation Guide:
    ------------------
    When implementing a new LLM interface:
    1. Override _format_tools_for_provider() to convert ToolParams to provider format
    2. Use _handle_tool_execution() for consistent tool execution
    3. Implement tool response handling in process() and stream() methods
    4. Cache processed tools using available_tools
    """

    # Default supported media types (can be overridden by specific implementations)
    SUPPORTED_MEDIA_TYPES: Set[MediaType] = {MediaType.JPEG, MediaType.PNG, MediaType.PDF}

    # Pydantic fields
    context: ConversationContext = Field(
        default_factory=ConversationContext, description="Current conversation context"
    )
    tokenizer: Optional[BaseTokenizer] = Field(default=None, description="Tokenizer instance")
    state: LLMState = Field(description="Complete state container for LLM instances")
    ignore_missing_credentials: bool = Field(default=False, description="Whether to ignore missing credentials")
    credentials: Optional[APIKey] = Field(default=None, description="API credentials")
    images: List[ImageInput] = Field(default_factory=list, description="List of images to include in messages")
    available_tools: Dict[str, ProcessedToolType] = Field(
        default_factory=dict, description="Cache of processed tool parameters, keyed by tool name"
    )

    # Timeout configuration
    request_timeout: Optional[float] = Field(
        default=30.0, description="Maximum time in seconds for a single request", ge=0
    )
    total_timeout: Optional[float] = Field(default=90.0, description="Maximum time in seconds for all retries", ge=0)
    socket_timeout: Optional[float] = Field(
        default=10.0, description="Maximum time in seconds for socket operations", ge=0
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        from_attributes=True,
        extra="allow",
        validate_default=False,
        frozen=False,
        populate_by_name=True,
    )

    def __init__(self, **data):
        """Initialize the interface with logging and error handling."""
        logger.debug(f"Initializing {self.__class__.__name__}")
        logger.debug(f"Init data: {data}")
        try:
            super().__init__(**data)
            self._post_super_init(**data)  # Template method for subclasses
            logger.debug(f"{self.__class__.__name__} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {self.__class__.__name__}: {str(e)}", exc_info=True)
            raise

    def _post_super_init(self, **data):
        """Template method for subclass-specific initialization."""
        pass

    def _create_response_metadata(
        self, is_streaming: bool = False, is_partial: bool = False, **additional_metadata
    ) -> Dict[str, Any]:
        """Create standardized response metadata with extension point."""
        base_metadata = {
            "model": self.state.profile.name if self.state else "unknown",
            "is_streaming": is_streaming,
            "is_partial": is_partial,
        }
        return {**base_metadata, **additional_metadata}

    def _calculate_context_metrics(self) -> ContextMetrics:
        """Calculate standardized context window metrics."""
        if not self.state:
            return ContextMetrics(
                max_context_tokens=0, current_context_tokens=0, available_tokens=0, context_utilization=0.0
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

    def _handle_error(self, e: Exception, context: str = "") -> LLMResponse:
        """Standardized error handling for LLM operations."""
        error_message = f"Error {context}: {str(e)}"
        logger.error(error_message, exc_info=True)
        return LLMResponse(
            content="",
            success=False,
            error=str(e),
            metadata={"error_type": type(e).__name__},
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    def _process_tools(self, tools: Optional[List[Union[ToolInputType, ProcessedToolType]]]) -> List[ProcessedToolType]:
        """Process tools into standardized format and update cache."""
        if not tools:
            return []

        processed_tools = []
        for tool in tools:
            if isinstance(tool, ProcessedToolType):
                # For existing ToolParams, keep as is
                processed_tools.append(tool)
            elif isinstance(tool, type) and issubclass(tool, BaseModel):
                # For Pydantic models, use the from_pydantic method
                tool_params = ToolParams.from_pydantic(tool)
                processed_tools.append(tool_params)
            elif callable(tool):
                # For callables, create a ToolParams with the function
                tool_params = ToolParams.from_function(tool)
                processed_tools.append(tool_params)
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")

        return processed_tools

    async def _prepare_tools(
        self,
        prompt_tools: Optional[List[ProcessedToolType]] = None,
        runtime_tools: Optional[List[ToolInputType]] = None,
    ) -> List[ProcessedToolType]:
        """Prepare and merge tools from different sources.

        Args:
            prompt_tools: Optional list of already processed tools from the prompt
            runtime_tools: Optional list of unprocessed tools from runtime

        Returns:
            List of processed tools, with runtime tools taking precedence
        """
        # Process prompt-level tools (already processed, just need to be merged)
        processed_prompt_tools = prompt_tools or []

        # Process runtime tools
        processed_runtime_tools = self._process_tools(runtime_tools) if runtime_tools else []

        # Merge tools, giving precedence to runtime tools
        final_tools = processed_prompt_tools + processed_runtime_tools

        # Cache all tools
        for tool in final_tools:
            self.available_tools[tool.name] = tool

        return final_tools

    async def _format_messages(
        self, input_data: Any, system_prompt: Optional[str] = None, attachments: Optional[List[BaseAttachment]] = None
    ) -> List[Dict[str, Any]]:
        """Base message formatting with extension points."""
        messages: List[Dict[str, Any]] = []

        # Handle system prompt
        if system_prompt:
            # Check if system prompts are supported
            supports_system_prompt = self._check_capability("supports_system_prompt")
            if supports_system_prompt:
                system_message = self._create_system_message(system_prompt)
                messages.append(system_message)
            else:
                # Convert system prompt to user message if system prompts not supported
                logger.warning(
                    f"System prompts are not supported by model "
                    f"'{self.state.profile.name if self.state else 'unknown'}'. "
                    "Converting system prompt to user message."
                )
                user_message = self._create_user_message(f"System instruction: {system_prompt}")
                messages.append(user_message)

        # Handle input data
        if isinstance(input_data, str):
            user_message = self._create_user_message(input_data)
            messages.append(user_message)
        elif isinstance(input_data, dict):
            messages.append(input_data)
        elif isinstance(input_data, list):
            messages.extend(input_data)

        # Handle attachments through template method
        if attachments:
            # Check if vision is supported when attachments are present
            if any(attachment.media_type.is_image() for attachment in attachments) and not self._check_capability(
                "supports_vision"
            ):
                logger.warning(
                    f"Vision capabilities are not supported by model "
                    f"'{self.state.profile.name if self.state else 'unknown'}'. "
                    "Image attachments will be ignored."
                )
                # Filter out image attachments
                attachments = [attachment for attachment in attachments if not attachment.media_type.is_image()]

            if attachments:  # Only process if we still have attachments after filtering
                messages = await self._handle_attachments(messages, attachments)

        return messages

    def _create_system_message(self, content: str) -> Dict[str, Any]:
        """Create a system message."""
        return {"role": "system", "content": content}

    def _create_user_message(self, content: str) -> Dict[str, Any]:
        """Create a user message."""
        return {"role": "user", "content": content}

    async def _handle_attachments(
        self, messages: List[Dict[str, Any]], attachments: List[BaseAttachment]
    ) -> List[Dict[str, Any]]:
        """Template method for handling attachments."""
        return messages  # Base implementation does nothing

    def _check_capability(self, capability_name: str, raise_error: bool = False) -> bool:
        """Check if a specific capability is supported by the model.

        Args:
            capability_name: Name of the capability to check (e.g., 'supports_streaming')
            raise_error: Whether to raise an error if the capability is not supported

        Returns:
            True if the capability is supported, False otherwise

        Raises:
            ValueError: If raise_error is True and the capability is not supported
        """
        if not self.state or not getattr(self.state.profile, "capabilities", None):
            # If we don't have state or capabilities, assume not supported
            if raise_error:
                raise ValueError(f"Capability '{capability_name}' is not supported by this model")
            return False

        capabilities = self.state.profile.capabilities
        is_supported = getattr(capabilities, capability_name, False)

        if not is_supported and raise_error:
            model_name = self.state.profile.name if self.state else "unknown"
            raise ValueError(f"Capability '{capability_name}' is not supported by model '{model_name}'")

        return is_supported

    def _check_model_capabilities(self, **kwargs: Any) -> Dict[str, Any]:
        """Check if the model supports specific capabilities and filter parameters accordingly.

        This method examines the provided kwargs against the model's capabilities and removes
        parameters that are not supported by the model.

        Args:
            **kwargs: Parameters to check against model capabilities

        Returns:
            Filtered parameters with unsupported options removed
        """
        if not self.state or not getattr(self.state.profile, "capabilities", None):
            return kwargs

        filtered_kwargs = kwargs.copy()
        capabilities = self.state.profile.capabilities

        # Map parameters to capability flags
        capability_param_map = {
            "temperature": "supports_temperature",
            "stream": "supports_streaming",
            "tools": "supports_tools",
            "functions": "supports_function_calling",
            "function_call": "supports_function_calling",
            "response_format": "supports_json_mode",
            "frequency_penalty": "supports_frequency_penalty",
            "presence_penalty": "supports_presence_penalty",
            "stop": "supports_stop_sequences",
        }

        # Check each parameter against capabilities
        for param, capability in capability_param_map.items():
            if param in filtered_kwargs and not getattr(capabilities, capability, False):
                logger.warning(
                    f"Parameter '{param}' is not supported by model '{self.state.profile.name}' "
                    f"(missing capability: {capability}). Parameter will be ignored."
                )
                filtered_kwargs.pop(param, None)

        # Special handling for system messages
        if not getattr(capabilities, "supports_system_prompt", True) and "messages" in filtered_kwargs:
            messages = filtered_kwargs["messages"]
            if isinstance(messages, list) and any(msg.get("role") == "system" for msg in messages):
                logger.warning(
                    f"System messages are not supported by model '{self.state.profile.name}'. "
                    "System messages will be converted to user messages."
                )
                # Convert system messages to user messages
                new_messages = []
                for msg in messages:
                    if msg.get("role") == "system":
                        new_messages.append(
                            {"role": "user", "content": f"System instruction: {msg.get('content', '')}"}
                        )
                    else:
                        new_messages.append(msg)
                filtered_kwargs["messages"] = new_messages

        return filtered_kwargs

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Basic token counting."""
        if not self.tokenizer:
            return 0
        return self.tokenizer.count_messages(messages)

    def validate_credentials(self) -> None:
        """Validate the credentials for the LLM provider."""
        if not self.credentials and not self.ignore_missing_credentials:
            raise ValueError(f"API key is required for provider {self.state.model_family if self.state else 'unknown'}")

    @property
    @abstractmethod
    def model_family(self) -> str:
        """Get the model family."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize async components of the interface."""
        pass

    def clear_tool_cache(self) -> None:
        """Clear the cached processed tools."""
        self.available_tools.clear()

    def get_cached_tool(self, name: str) -> Optional[ProcessedToolType]:
        """Get a processed tool from the cache by name.

        Args:
            name: Name of the tool to retrieve

        Returns:
            The processed ToolParams if found, None otherwise
        """
        return self.available_tools.get(name)

    @abstractmethod
    async def process(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
        tools: Optional[List[ToolInputType]] = None,
    ) -> LLMResponse:
        """Process input using the LLM with optional tool support.

        Implementation Requirements:
        1. Check capabilities before using features:
           - Use self._check_capability('supports_tools') before processing tools
           - Use self._check_capability('supports_streaming') before enabling streaming
           - Use self._check_capability('supports_json_mode') before using structured output
        2. Process tools in order of precedence:
           - Runtime tools (from method argument)
           - Prompt-level tools (from BasePrompt.tools)
           - Interface-level tools (from available_tools)
        3. Use _process_tools() to standardize tool format
        4. Use _format_tools_for_provider() for API-specific formatting
        5. Use _handle_tool_execution() for consistent tool execution
        6. Handle both sync and async tool execution
        7. Validate tool inputs against parameter schemas

        Example Implementation:
            ```python
            async def process(self, prompt, variables=None, tools=None):
                # Check capabilities
                supports_tools = self._check_capability('supports_tools')

                # Process tools if supported
                final_tools = []
                if supports_tools:
                    # 1. Interface-level tools (lowest precedence)
                    final_tools.extend(self.available_tools.values())

                    # 2. Prompt-level tools (medium precedence)
                    if isinstance(prompt, BasePrompt) and prompt.tools:
                        prompt_tools = self._process_tools(prompt.tools)
                        final_tools.extend(prompt_tools)

                    # 3. Runtime tools (highest precedence)
                    if tools:
                        runtime_tools = self._process_tools(tools)
                        final_tools.extend(runtime_tools)

                    # Format for provider and make API call
                    api_tools = self._format_tools_for_provider(final_tools)
                    response = await self._make_api_call(prompt, api_tools)

                    # Handle tool calls in response
                    if response.tool_calls:
                        for call in response.tool_calls:
                            result = await self._handle_tool_execution(
                                self.available_tools[call.name],
                                call.arguments
                            )
                            # Process tool result...
                else:
                    # Make API call without tools
                    response = await self._make_api_call(prompt)

                return LLMResponse(...)
            ```
        """
        pass

    @abstractmethod
    async def batch_process(
        self,
        prompts: List[Union[BasePrompt, str]],
        variables: Optional[List[Optional[Dict[str, Any]]]] = None,
        batch_size: Optional[int] = None,
        tools: Optional[List[Union[Callable, Type[BaseModel], ToolParams]]] = None,
    ) -> List[LLMResponse]:
        """Process multiple prompts in a batch.

        Args:
            prompts: List of prompts to process
            variables: Optional list of variable dictionaries for each prompt
            batch_size: Optional batch size for processing
            tools: Optional list of tools/functions that can be called by the LLM.
                  Can be functions, Pydantic models, or ToolParams objects.

        Implementation Requirements:
            - Check capabilities before using features:
              - Use self._check_capability('supports_tools') before processing tools
              - Use self._check_capability('supports_parallel_requests') before batching
            - Must use _process_tools() to convert tools into standardized format
            - Must handle tool invocation responses from the LLM
            - Must validate tool inputs against parameter schemas
            - Must handle both sync and async tool execution
            - Tools should be processed once and reused across batch

        Returns:
            List of LLMResponse objects
        """
        pass

    @abstractmethod
    async def stream(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Union[Callable, Type[BaseModel], ToolParams]]] = None,
    ) -> AsyncIterator[LLMResponse]:
        """Stream responses from the LLM one token at a time.

        Args:
            prompt: The prompt to process, either as a BasePrompt or string
            variables: Optional variables to use in prompt rendering
            tools: Optional list of tools/functions that can be called by the LLM.
                  Can be functions, Pydantic models, or ToolParams objects.

        Implementation Requirements:
            - Check capabilities before using features:
              - Use self._check_capability('supports_streaming', raise_error=True) to verify streaming support
              - Use self._check_capability('supports_tools') before processing tools
            - Must use _process_tools() to convert tools into standardized format
            - Must handle tool invocation responses from the LLM
            - Must validate tool inputs against parameter schemas
            - Must handle both sync and async tool execution
            - Must properly handle streaming while tools are being executed

        Yields:
            LLMResponse objects containing partial responses
        """
        pass

    async def shutdown(self) -> None:
        """Shutdown the interface."""
        self.clear_tool_cache()  # Clear tool cache on shutdown
        pass

    @property
    @abstractmethod
    def supports_structured_output(self) -> bool:
        """Whether this interface supports native structured output.

        Implementation should check capabilities:
        - Check if the model supports JSON mode: self._check_capability('supports_json_mode')
        - Check if the model supports direct Pydantic parsing: self._check_capability('supports_direct_pydantic_parse')

        Example implementation:
        ```python
        @property
        def supports_structured_output(self) -> bool:
            return self._check_capability('supports_json_mode')
        ```
        """
        pass

    @property
    @abstractmethod
    def supports_json_schema(self) -> bool:
        """Whether this interface supports JSON schema validation.

        Implementation should check capabilities:
        - Check if the model supports JSON schema: self._check_capability('supports_json_mode')

        Example implementation:
        ```python
        @property
        def supports_json_schema(self) -> bool:
            return self._check_capability('supports_json_mode')
        ```
        """
        pass

    @abstractmethod
    async def parse_structured_response(
        self,
        response: str,
        expected_format: ResponseFormat,
    ) -> Any:
        """Parse a structured response from the LLM.

        Args:
            response: The raw response string from the LLM
            expected_format: The expected response format

        Implementation Requirements:
            - Check capabilities before using features:
              - Use self._check_capability('supports_json_mode') before using JSON mode
              - Use self._check_capability('supports_direct_pydantic_parse') before using direct Pydantic parsing
            - Fall back to manual parsing if native capabilities are not available
            - Validate the parsed response against the expected format

        Returns:
            Parsed response object (may be a Pydantic model or dict)
        """
        pass

    @abstractmethod
    def _format_tools_for_provider(self, tools: List[ProcessedToolType]) -> Any:
        """Convert processed tools into provider-specific format.

        This method should be implemented by provider interfaces to convert
        standardized ToolParams into the format expected by their API.

        Args:
            tools: List of processed ToolParams objects

        Returns:
            Provider-specific tool format (type varies by provider)

        Example Implementation:
            ```python
            def _format_tools_for_provider(self, tools):
                return [{
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                } for tool in tools]
            ```
        """
        pass

    async def _handle_tool_execution(self, tool: ProcessedToolType, args: Dict[str, Any]) -> str:
        """Execute a tool and format its result for LLM consumption.

        This method provides consistent tool execution and result formatting
        across all interfaces. Override only if provider needs custom handling.

        Args:
            tool: The processed tool to execute
            args: Arguments to pass to the tool

        Returns:
            Formatted string response suitable for LLM consumption

        Example Response:
            ```
            Tool 'calculator' executed successfully.
            Result: 42
            Type: int
            ```
        """
        try:
            result = await tool.execute(**args)
            return f"Tool '{tool.name}' executed successfully.\n" f"Result: {result}\n" f"Type: {type(result).__name__}"
        except Exception as e:
            return f"Tool '{tool.name}' execution failed.\nError: {str(e)}"

    async def _handle_timeout(self, coro: Awaitable[T], timeout: Optional[float], error_msg: str) -> T:
        """Handle timeouts for async operations.

        Args:
            coro: Coroutine to execute with timeout
            timeout: Timeout in seconds (None for no timeout)
            error_msg: Error message to use if timeout occurs

        Returns:
            Result of the coroutine

        Raises:
            TimeoutError: If the operation times out
        """
        if timeout is None:
            return await coro

        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"Operation timed out after {timeout}s: {error_msg}")
            raise TimeoutError(error_msg) from None
