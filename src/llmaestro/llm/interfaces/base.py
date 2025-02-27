"""Base interfaces for LLM providers."""
from __future__ import annotations

import base64
import logging
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, AsyncIterator, Type, Callable

from pydantic import BaseModel, Field, ConfigDict

from llmaestro.core.conversations import ConversationContext
from llmaestro.core.models import LLMResponse
from llmaestro.llm.credentials import APIKey
from llmaestro.prompts.base import BasePrompt
from llmaestro.llm.enums import MediaType
from llmaestro.prompts.types import ResponseFormat
from .tokenizers import BaseTokenizer
from llmaestro.llm.models import LLMState  # Direct import instead of TYPE_CHECKING

logger = logging.getLogger(__name__)


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


class ToolParams(BaseModel):
    """Parameters for a tool/function that can be used by an LLM."""

    name: str = Field(description="The function's name")
    description: str = Field(description="Details on when and how to use the function")
    parameters: Dict[str, Any] = Field(description="JSON schema defining the function's input arguments")
    return_type: Optional[Any] = Field(default=None, description="The return type of the function if available")
    is_async: bool = Field(default=False, description="Whether the function is async")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @staticmethod
    def _get_parameter_schema(param: inspect.Parameter) -> Dict[str, Any]:
        """Get JSON Schema for a parameter using Pydantic's type system."""
        from pydantic import TypeAdapter

        # Get type annotation or default to str
        annotation = param.annotation if param.annotation != inspect.Parameter.empty else str

        # Get JSON schema for the type
        schema = TypeAdapter(annotation).json_schema()

        # Add default value if present
        if param.default != inspect.Parameter.empty:
            schema["default"] = param.default

        return schema

    @classmethod
    def from_function(cls, func: Callable) -> ToolParams:
        """Generate tool parameters from a function."""
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        parameters = {}
        required = []

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            # Get parameter schema using Pydantic
            param_info = cls._get_parameter_schema(param)

            # Handle required fields
            if param.default == inspect.Parameter.empty:
                required.append(name)

            parameters[name] = param_info

        param_schema = {"type": "object", "properties": parameters, "required": required}

        # Check if async
        is_async = inspect.iscoroutinefunction(func)

        # Get return type if available
        return_type = None
        if sig.return_annotation != inspect.Signature.empty:
            return_type = sig.return_annotation

        return cls(
            name=func.__name__, description=doc, parameters=param_schema, return_type=return_type, is_async=is_async
        )

    @classmethod
    def from_pydantic(cls, model: Type[BaseModel]) -> ToolParams:
        """Generate tool parameters from a Pydantic model."""
        schema = model.model_json_schema()

        tool_params = cls(name=model.__name__, description=model.__doc__ or "", parameters=schema, return_type=model)

        return tool_params


class BaseLLMInterface(BaseModel, ABC):
    """Base class for LLM interfaces."""

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
    available_tools: Dict[str, ToolParams] = Field(
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

    def _process_tools(self, tools: Optional[List[Union[Callable, Type[BaseModel], ToolParams]]]) -> List[ToolParams]:
        """Process a list of tools into standardized ToolParams objects.

        This helper method converts various tool input types into ToolParams objects:
        - Functions are converted using ToolParams.from_function
        - Pydantic models are converted using ToolParams.from_pydantic
        - ToolParams objects are processed using _use_tool

        The processed tools are cached in self.processed_tools to avoid reprocessing.

        Args:
            tools: List of tools to process, can be functions, Pydantic models, or ToolParams

        Returns:
            List of processed ToolParams objects ready for use with the LLM
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

            # Check cache first
            if tool_name in self.available_tools:
                processed_tools.append(self.available_tools[tool_name])
                continue

            # Process and cache new tool
            processed_tool = None
            if isinstance(tool, ToolParams):
                processed_tool = self._use_tool(tool)
            elif isinstance(tool, type) and issubclass(tool, BaseModel):
                processed_tool = self._use_tool(ToolParams.from_pydantic(tool))
            elif callable(tool):
                processed_tool = self._use_tool(ToolParams.from_function(tool))

            if processed_tool:
                self.available_tools[tool_name] = processed_tool
                processed_tools.append(processed_tool)

        return processed_tools

    def clear_tool_cache(self) -> None:
        """Clear the cached processed tools."""
        self.available_tools.clear()

    def get_cached_tool(self, name: str) -> Optional[ToolParams]:
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
        tools: Optional[List[Union[Callable, Type[BaseModel], ToolParams]]] = None,
    ) -> LLMResponse:
        """Process input using the LLM.

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

        Returns:
            LLMResponse containing the model's response and metadata
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

    @abstractmethod
    def _use_tool(self, tool_params: ToolParams) -> ToolParams:
        """Process tool parameters for specific LLM implementation.

        This method should be implemented by subclasses to convert the
        ToolParams into a format suitable for their specific LLM API.

        Args:
            tool_params: The tool parameters to process

        Returns:
            Processed tool parameters
        """
        pass
