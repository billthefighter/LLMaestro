"""Base classes for prompts."""
import json
import re
from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from llmaestro.llm.enums import MediaType
from llmaestro.prompts.mixins import VersionMixin
from llmaestro.prompts.types import PromptMetadata
from pydantic import BaseModel, Field, create_model

from llmaestro.core.attachments import BaseAttachment, FileAttachment, ImageAttachment, AttachmentConverter


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


class PromptVariable(BaseModel):
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

            # If it's already a string, validate it's valid JSON
            if isinstance(value, str):
                try:
                    json.loads(value)
                    return value
                except json.JSONDecodeError as e:
                    raise ValueError(f"Schema variable {self.name} contains invalid JSON: {e}")

            # If it's a dict, convert to JSON string
            try:
                return json.dumps(value)
            except TypeError as e:
                raise ValueError(f"Schema variable {self.name} contains non-serializable values: {e}")

        if self.string_conversion_template is None:
            return str(value)

        if isinstance(self.string_conversion_template, str):
            return self.string_conversion_template.format(value=value)

        if callable(self.string_conversion_template):
            try:
                return self.string_conversion_template(value)
            except Exception as e:
                raise ValueError(
                    f"Function passed to string_conversion_template for variable {self.name} raised an error: {e}"
                ) from e

        raise ValueError(f"Invalid string_conversion_template for variable {self.name}")


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
            f"{self.name}: {self.description} \nSystem Prompt: {self.system_prompt} \nUser Prompt: {self.user_prompt}"
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

    def render(
        self, variables: Optional[Union[Dict[str, Any], BaseModel]] = None, **additional_kwargs: Any
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Render the prompt template with the given variables.

        Args:
            variables: Dictionary or Pydantic model containing variable values.
                      If using the variables model from get_variables_model(),
                      values will be validated against expected types.
            additional_kwargs: Additional keyword arguments for backward compatibility.

        Returns:
            Tuple of (system_prompt, user_prompt, attachments)
            where attachments is a list of dicts compatible with LLM interface

        Raises:
            ValueError: If required variables are missing or if variable values don't match expected types.
        """
        self._validate_template()

        # Convert variables to dict if it's a model
        var_dict: Dict[str, Any] = {}
        if isinstance(variables, BaseModel):
            var_dict = variables.model_dump()
        elif variables is not None:
            # If we have a variables model, validate the dict
            if self._variables_model is not None:
                var_dict = self._variables_model(**variables).model_dump()
            else:
                var_dict = variables

        # Add additional kwargs
        all_kwargs = {**var_dict, **additional_kwargs}

        try:
            # Convert variables using their defined conversion methods
            converted_kwargs: Dict[str, str] = {}
            for var in self.variables:
                if var.name in all_kwargs:
                    converted_kwargs[var.name] = var.convert_value(all_kwargs[var.name])
                elif var.name in self._extract_template_vars():
                    raise ValueError(f"Missing required variable: {var.name}")

            # Add any remaining kwargs that don't have PromptVariable definitions
            for key, value in all_kwargs.items():
                if key not in converted_kwargs:
                    converted_kwargs[key] = str(value)

            formatted_system_prompt = self.system_prompt.format(**converted_kwargs)
            formatted_user_prompt = self.user_prompt.format(**converted_kwargs)

            # Add response format information to system prompt if available
            if self.metadata and self.metadata.expected_response:
                response_format = self.metadata.expected_response
                if response_format.format and response_format.response_schema:
                    formatted_system_prompt = f"{formatted_system_prompt}\nPlease provide your response in {response_format.format.value} format using the following schema:\n{response_format.response_schema}"
                elif response_format.format:
                    formatted_system_prompt = f"{formatted_system_prompt}\nPlease provide your response in {response_format.format.value} format."

            # Format attachments for LLM interface using AttachmentConverter
            formatted_attachments = [AttachmentConverter.to_interface_format(att) for att in self.attachments]

            return formatted_system_prompt, formatted_user_prompt, formatted_attachments
        except KeyError as e:
            required_vars = self._extract_template_vars()
            missing_vars = [var for var in required_vars if var not in all_kwargs]
            raise ValueError(f"Missing required variables: {missing_vars}. Error: {e}") from e

    def _validate_template(self) -> None:
        """Validate the template format and variables."""
        # Check for balanced braces
        for template in [self.system_prompt, self.user_prompt]:
            open_count = template.count("{")
            close_count = template.count("}")
            if open_count != close_count:
                raise ValueError(f"Unbalanced braces in template: {open_count} open, {close_count} close")

        # Check for valid variable names
        pattern = r"\{([^}]+)\}"
        for match in re.finditer(pattern, self.system_prompt + self.user_prompt):
            var_name = match.group(1)
            if not var_name.isidentifier() and not any(c in var_name for c in ":.[]"):
                raise ValueError(f"Invalid variable name in template: {var_name}")

    def _extract_template_vars(self) -> Set[str]:
        """Extract required variables from the prompt template."""
        pattern = r"\{([^}]+)\}"
        system_vars = set(re.findall(pattern, self.system_prompt))
        user_vars = set(re.findall(pattern, self.user_prompt))
        return system_vars.union(user_vars)

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
