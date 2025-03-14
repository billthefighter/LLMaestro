"""Base classes for prompts."""
import re
from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from llmaestro.llm.enums import MediaType
from llmaestro.prompts.mixins import VersionMixin
from llmaestro.prompts.types import PromptMetadata
from llmaestro.tools.core import ToolParams
from pydantic import BaseModel, Field, create_model
from llmaestro.llm.schema_utils import schema_to_json
from llmaestro.core.persistence import PersistentModel

from llmaestro.core.attachments import BaseAttachment, FileAttachment, ImageAttachment, AttachmentConverter
from llmaestro.llm.responses import ResponseFormat
import jsonschema


# Type definitions for prompt variables
class SerializableType(str, Enum):
    """Enumeration of serializable types for prompt variables."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    SCHEMA = "schema"


# Mapping of SerializableType to Python types
TYPE_MAPPING = {
    SerializableType.STRING: str,
    SerializableType.INTEGER: int,
    SerializableType.FLOAT: float,
    SerializableType.BOOLEAN: bool,
    SerializableType.LIST: list,
    SerializableType.DICT: dict,
    SerializableType.SCHEMA: Union[dict, str],
}


class PromptVariable(PersistentModel):
    """Represents a variable in a prompt template."""

    name: str
    description: Optional[str] = None
    expected_input_type: SerializableType = SerializableType.STRING
    string_conversion_template: Optional[Union[str, Callable]] = None

    def convert_value(self, value: Any) -> str:
        """Convert the input value to a string using the template or default str() method."""
        if value is None:
            raise ValueError(f"Value for variable {self.name} cannot be None")

        # Special handling for SCHEMA type
        if self.expected_input_type == SerializableType.SCHEMA:
            if not isinstance(value, (dict, str)):
                raise ValueError(f"Schema variable {self.name} expects a dict or JSON string, got {type(value)}")

            try:
                return schema_to_json(value)
            except ValueError as e:
                raise ValueError(f"Schema variable {self.name} contains invalid JSON: {e}") from e

        if self.string_conversion_template is None:
            return str(value)

        if isinstance(self.string_conversion_template, str):
            return self.string_conversion_template.format(value=value)

        return self.string_conversion_template(value)


class BasePrompt(BaseModel):
    """Base class for all prompts.

    This class handles the core prompt functionality without version control.
    For versioning support, use VersionedPrompt instead.
    """

    name: str
    description: str
    system_prompt: str = Field(description="System prompt template")
    user_prompt: str = Field(description="User prompt template")
    metadata: Optional[PromptMetadata] = Field(default=None, description="Optional metadata about the prompt")
    examples: Optional[List[Dict[str, Any]]] = None
    attachments: List[BaseAttachment] = Field(default_factory=list, description="List of file attachments")
    variables: List[PromptVariable] = Field(default_factory=list, description="List of variable definitions")
    expected_response: Optional[ResponseFormat] = Field(default=None, description="Expected response format")
    tools: List[ToolParams] = Field(default_factory=list, description="List of tools available to the prompt")
    template_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional schema for template validation"
    )

    # Variables model for validation
    _variables_model: Optional[Type[BaseModel]] = None

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}

    def __init__(self, **data):
        """Initialize the prompt."""
        super().__init__(**data)
        self._create_variables_model()
        self._validate_template()

    def __str__(self) -> str:
        """Return a string representation of the prompt."""
        return (
            f"{self.name}: {self.description}\n"
            f"System Prompt: {self.system_prompt}\n"
            f"User Prompt: {self.user_prompt}"
        )

    def _create_variables_model(self) -> None:
        """Create a Pydantic model for the prompt variables."""
        if not self.variables:
            return

        fields = {}
        for var in self.variables:
            python_type = TYPE_MAPPING[var.expected_input_type]
            fields[var.name] = (python_type, Field(description=var.description or ""))

        # Create a new model with the fields
        model_name = f"{self.name.title()}Variables"
        self._variables_model = create_model(model_name, __base__=BaseModel, **fields)

    def get_variables_model(self) -> Optional[Type[BaseModel]]:
        """Get the Pydantic model for the prompt variables.

        Returns:
            A Pydantic model class with fields matching the prompt variables,
            or None if no variables are defined.

        Example:
            ```python
            prompt = BasePrompt(
                name="example",
                variables=[
                    PromptVariable(name="user_name", expected_input_type=SerializableType.STRING),
                    PromptVariable(name="count", expected_input_type=SerializableType.INTEGER)
                ],
                ...
            )

            # Get the variables model
            VariablesModel = prompt.get_variables_model()

            # Create and validate variables
            vars = VariablesModel(user_name="Alice", count=5)

            # Use in render
            prompt.render(variables=vars.model_dump())
            ```
        """
        return self._variables_model

    def get_variable_types(self) -> Dict[str, SerializableType]:
        """Get a dictionary mapping variable names to their expected types."""
        return {var.name: var.expected_input_type for var in self.variables}

    def get_required_variables(self) -> Set[str]:
        """Get the set of required variable names used in the prompt templates.

        Returns:
            Set of variable names that must be provided when rendering.
        """
        return self._extract_template_vars()

    def add_attachment(
        self,
        content: Union[str, bytes],
        media_type: Union[str, MediaType],
        file_name: str,
        description: Optional[str] = None,
    ) -> None:
        """Add a file attachment to the prompt.

        Args:
            content: The file content as string or bytes
            media_type: The media type (MIME type) of the file
            file_name: Name of the file
            description: Optional description of the attachment
        """
        if isinstance(media_type, str):
            media_type = MediaType.from_mime_type(media_type)

        if media_type.is_image():
            attachment = ImageAttachment(
                content=content, media_type=media_type, file_name=file_name, description=description
            )
        else:
            attachment = FileAttachment(
                content=content, media_type=media_type, file_name=file_name, description=description
            )
        self.attachments.append(attachment)

    def clear_attachments(self) -> None:
        """Remove all attachments from the prompt."""
        self.attachments.clear()

    def render(self, **variable_values: Any) -> Tuple[str, str, List[Dict[str, Any]], List[ToolParams]]:
        """Render the prompt template with the given variable values.

        Args:
            **variable_values: Keyword arguments matching the defined variables.
                             Values must match the types defined in the prompt's variables.

        Returns:
            Tuple of (system_prompt, user_prompt, attachments, tools)
            where:
            - attachments is a list of dicts compatible with LLM interface
            - tools is a list of ToolParams that can be formatted by the specific LLM interface

        Raises:
            ValueError: If required variables are missing or if variable values don't match expected types.
        """
        self._validate_template()

        # Validate variables against the model
        if self._variables_model is not None:
            try:
                validated_vars = self._variables_model(**variable_values)
                var_dict = validated_vars.model_dump()
            except Exception as e:
                raise ValueError(f"Invalid variable values: {e}") from e
        else:
            var_dict = variable_values

        try:
            # Convert variables using their defined conversion methods
            converted_kwargs: Dict[str, str] = {}
            for var in self.variables:
                if var.name in var_dict:
                    converted_kwargs[var.name] = var.convert_value(var_dict[var.name])
                elif var.name in self._extract_template_vars():
                    raise ValueError(f"Missing required variable: {var.name}")

            # Format the prompts
            formatted_system_prompt = self.system_prompt.format(**converted_kwargs)
            formatted_user_prompt = self.user_prompt.format(**converted_kwargs)

            # Add response format information to system prompt if available
            if self.expected_response:
                formatted_system_prompt = self._add_response_format(formatted_system_prompt)

            # Format attachments for LLM interface
            formatted_attachments = [AttachmentConverter.to_interface_format(att) for att in self.attachments]

            return formatted_system_prompt, formatted_user_prompt, formatted_attachments, self.tools
        except KeyError as e:
            required_vars = self._extract_template_vars()
            missing_vars = [var for var in required_vars if var not in var_dict]
            raise ValueError(f"Missing required variables: {missing_vars}. Error: {e}") from e

    def _add_response_format(self, system_prompt: str) -> str:
        """Add response format information to the system prompt."""
        if not self.expected_response:
            return system_prompt

        response_format = self.expected_response
        if response_format.format and response_format.response_schema:
            return (
                f"{system_prompt}\n"
                f"Please provide your response in {response_format.format.value} format "
                f"using the following schema:\n{response_format.response_schema}"
            )
        elif response_format.format:
            return f"{system_prompt}\nPlease provide your response in {response_format.format.value} format."
        return system_prompt

    def _validate_template(self) -> None:
        """Validate the prompt templates."""
        try:
            # Check for required variables
            required_vars = self._extract_template_vars()
            defined_vars = {var.name for var in self.variables}
            missing_vars = required_vars - defined_vars
            if missing_vars:
                raise ValueError(f"Template contains undefined variables: {missing_vars}")
        except Exception as err:
            raise ValueError(f"Template validation failed: {str(err)}") from err

    def _extract_template_vars(self) -> Set[str]:
        """Extract required variables from the prompt template."""
        pattern = r"\{([^}]+)\}"
        system_vars = set(re.findall(pattern, self.system_prompt))
        user_vars = set(re.findall(pattern, self.user_prompt))
        return system_vars.union(user_vars)

    def _validate_variables(self, variables: Dict[str, Any]) -> None:
        """Validate variables against the template schema."""
        try:
            if hasattr(self, "template_schema") and self.template_schema:
                jsonschema.validate(variables, self.template_schema)
        except jsonschema.ValidationError as err:
            raise ValueError(f"Variable validation failed: {str(err)}") from err

    @abstractmethod
    async def save(self) -> bool:
        """Save the prompt to its storage backend.

        Returns:
            bool: True if save was successful, False otherwise.

        Raises:
            NotImplementedError: If the storage backend is not implemented.
        """
        raise NotImplementedError("Storage backend not implemented")

    @classmethod
    @abstractmethod
    async def load(cls, identifier: str) -> Optional["BasePrompt"]:
        """Load a prompt from its storage backend.

        Args:
            identifier: String identifier for the prompt (e.g., file path, URL, etc.)

        Returns:
            Optional[BasePrompt]: The loaded prompt, or None if not found.

        Raises:
            NotImplementedError: If the storage backend is not implemented.
        """
        raise NotImplementedError("Storage backend not implemented")


class VersionedPrompt(BasePrompt, VersionMixin):
    """A prompt with version control support."""

    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()},
        "exclude": {"version_history": {"exclude": True}},  # Allow excluding version history in dumps
    }

    @abstractmethod
    async def save(self) -> bool:
        """Save the prompt with version information to its storage backend.

        This method should handle saving both the prompt content and its version history.

        Returns:
            bool: True if save was successful, False otherwise.
        """
        pass

    @classmethod
    @abstractmethod
    async def load(cls, identifier: str) -> Optional["VersionedPrompt"]:
        """Load a versioned prompt from its storage backend.

        Args:
            identifier: String identifier for the prompt (e.g., file path, URL, etc.)

        Returns:
            Optional[VersionedPrompt]: The loaded prompt with version history, or None if not found.
        """
        pass
