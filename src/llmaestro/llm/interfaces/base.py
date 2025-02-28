"""Base interfaces for LLM providers."""
from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, AsyncIterator, Type, Callable, TypeVar

from pydantic import BaseModel, Field, ConfigDict

from llmaestro.core.conversations import ConversationContext
from llmaestro.core.models import LLMResponse
from llmaestro.llm.credentials import APIKey
from llmaestro.prompts.base import BasePrompt
from llmaestro.llm.enums import MediaType
from llmaestro.prompts.types import ResponseFormat
from .tokenizers import BaseTokenizer
from llmaestro.llm.models import LLMState  # Direct import instead of TYPE_CHECKING
from llmaestro.prompts.tools import ToolParams

logger = logging.getLogger(__name__)

# Add type variables for better type hinting
ToolInputType = Union[Callable[..., Any], Type[BaseModel], ToolParams]
ProcessedToolType = ToolParams
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
        logger.debug("Initializing BaseLLMInterface")
        logger.debug(f"Base init data: {data}")
        try:
            super().__init__(**data)
            logger.debug("BaseLLMInterface initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BaseLLMInterface: {str(e)}", exc_info=True)
            raise

    def model_post_init(self, __context: Any) -> None:
        """Initialize the interface after Pydantic model validation."""
        logger.debug("Running BaseLLMInterface post-init")
        if not self.state:
            logger.warning("No state provided in base post-init")
            return
        logger.debug("BaseLLMInterface post-init completed")

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Basic token counting."""
        if not self.tokenizer:
            return 0
        return self.tokenizer.count_messages(messages)

    def validate_credentials(self) -> None:
        """Validate the credentials for the LLM provider."""
        if not self.credentials and not self.ignore_missing_credentials:
            raise ValueError(f"API key is required for provider {self.state.model_family if self.state else 'unknown'}")

    def _format_messages(self, input_data: Any) -> List[Dict[str, Any]]:
        """Format input data into messages."""
        messages: List[Dict[str, Any]] = []

        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, dict):
            messages.append(input_data)
        elif isinstance(input_data, list):
            messages.extend(input_data)

        # Add images if provided
        if self.images:
            # Find or create user message
            last_user_msg: Optional[Dict[str, Any]] = None
            for msg in reversed(messages):
                if msg["role"] == "user":
                    last_user_msg = msg
                    break
            if not last_user_msg:
                last_user_msg = {"role": "user", "content": ""}
                messages.append(last_user_msg)

            # Add images to user message
            image_content: List[Dict[str, Any]] = []
            if isinstance(last_user_msg["content"], str):
                image_content.append({"type": "text", "text": last_user_msg["content"]})

            # Add image attachments
            for img in self.images:
                if isinstance(img.content, bytes):
                    img_content = base64.b64encode(img.content).decode("utf-8")
                else:
                    img_content = str(img.content)

                image_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{img.media_type.value};base64,{img_content}",
                            "detail": "auto",
                        },
                    }
                )

            last_user_msg["content"] = image_content

        return messages

    @property
    @abstractmethod
    def model_family(self) -> str:
        """Get the model family."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize async components of the interface."""
        pass

    def _process_tools(self, tools: Optional[List[ToolInputType]]) -> List[ProcessedToolType]:
        """Process tools into standardized format and update cache.

        This method handles the conversion of various tool types into ToolParams
        and manages the tool cache. Tools are processed once and cached for reuse.

        Args:
            tools: List of tools to process. Can be:
                - Functions (converted using ToolParams.from_function)
                - Pydantic models (converted using ToolParams.from_pydantic)
                - ToolParams objects (used as-is)

        Returns:
            List of processed ToolParams objects

        Example:
            ```python
            # Process different tool types
            tools = [
                my_function,  # Function
                MyPydanticModel,  # Pydantic model
                existing_tool_params  # ToolParams
            ]
            processed = self._process_tools(tools)
            ```
        """
        if not tools:
            return []

        processed_tools = []
        for tool in tools:
            # Get tool name for caching
            tool_name = None
            if isinstance(tool, ToolParams):
                tool_name = tool.name
            elif isinstance(tool, type) and issubclass(tool, BaseModel):
                tool_name = tool.__name__
            elif callable(tool):
                tool_name = tool.__name__
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")

            # Process and cache new tool
            processed_tool = None
            if isinstance(tool, ToolParams):
                processed_tool = self._use_tool(tool)
            elif isinstance(tool, type) and issubclass(tool, BaseModel):
                processed_tool = self._use_tool(ToolParams.from_pydantic(tool))
            elif callable(tool):
                processed_tool = self._use_tool(ToolParams.from_function(tool))

            if processed_tool:
                # Update cache and add to result list
                self.available_tools[tool_name] = processed_tool
                processed_tools.append(processed_tool)

        return processed_tools

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
        1. Process tools in order of precedence:
           - Runtime tools (from method argument)
           - Prompt-level tools (from BasePrompt.tools)
           - Interface-level tools (from available_tools)
        2. Use _process_tools() to standardize tool format
        3. Use _format_tools_for_provider() for API-specific formatting
        4. Use _handle_tool_execution() for consistent tool execution
        5. Handle both sync and async tool execution
        6. Validate tool inputs against parameter schemas

        Example Implementation:
            ```python
            async def process(self, prompt, variables=None, tools=None):
                # Process tools in order of precedence
                final_tools = []

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
        """Whether this interface supports native structured output."""
        pass

    @property
    @abstractmethod
    def supports_json_schema(self) -> bool:
        """Whether this interface supports JSON schema validation."""
        pass

    @abstractmethod
    def configure_structured_output(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure structured output settings for this interface.

        Args:
            config: Configuration from ResponseFormat.get_structured_output_config()

        Returns:
            Dict of interface-specific configuration to be used in API calls
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

        Returns:
            Parsed response object (may be a Pydantic model or dict)
        """
        pass

    def _use_tool(self, tool_params: ToolParams) -> ToolParams:
        """Validate and normalize tool parameters.

        This base implementation ensures tools meet minimum requirements.
        Providers typically won't need to override this unless they have
        specific validation needs.

        Args:
            tool_params: The tool parameters to process

        Returns:
            Validated and normalized tool parameters

        Raises:
            ValueError: If tool parameters are invalid
        """
        if not tool_params.name:
            raise ValueError("Tool must have a name")
        if not tool_params.parameters:
            raise ValueError("Tool must have parameters schema")
        return tool_params

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

    def _check_model_capabilities(self, **kwargs: Any) -> Dict[str, Any]:
        """Check if the model supports specific capabilities and filter parameters accordingly.

        This method checks the model's capabilities (stored in state.profile.capabilities)
        and removes any unsupported parameters from the kwargs.

        Args:
            **kwargs: Parameters to check against model capabilities

        Returns:
            Dict of filtered parameters that are supported by the model

        Example:
            ```python
            kwargs = {
                "temperature": 0.7,
                "max_tokens": 100,
                "stream": True
            }
            filtered = self._check_model_capabilities(**kwargs)
            # If model doesn't support temperature, it will be removed
            ```
        """
        if not self.state or not getattr(self.state.profile, "capabilities", None):
            return kwargs

        filtered_kwargs = kwargs.copy()
        capabilities = self.state.profile.capabilities

        # Check temperature support
        if not getattr(capabilities, "supports_temperature", False):
            filtered_kwargs.pop("temperature", None)

        # Add more capability checks here as needed
        # Example:
        # if not getattr(capabilities, 'supports_streaming', False):
        #     filtered_kwargs.pop('stream', None)

        return filtered_kwargs
