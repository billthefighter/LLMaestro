"""Response types and format handling for LLM interactions."""

import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union
import yaml
from pydantic import BaseModel, ConfigDict, Field
import json

from llmaestro.llm.models import TokenUsage, ContextMetrics
from llmaestro.llm.schema_utils import convert_to_schema, schema_to_json, validate_json


class ResponseFormatType(str, Enum):
    """Enumeration of supported response format types."""

    TEXT = "text"
    JSON = "json"
    JSON_SCHEMA = "json_schema"  # For structured JSON output with schema validation
    MARKDOWN = "markdown"
    CODE = "code"
    YAML = "yaml"
    XML = "xml"
    PYDANTIC = "pydantic"  # Direct Pydantic model validation


class StructuredOutputConfig(BaseModel):
    """Configuration for structured output between ResponseFormat and LLM interfaces.

    This class serves as a data transfer object (DTO) that encapsulates the minimal
    configuration needed by LLM providers to set up structured output handling.
    It is typically created by ResponseFormat.get_structured_output_config() and
    consumed by LLMInterface.configure_structured_output().

    The class maintains a clear separation between the user-facing configuration in
    ResponseFormat and the provider-specific implementation details in LLM interfaces.
    """

    format: ResponseFormatType = Field(..., description="The format type requested for the response")
    schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON schema for validation, if provided directly. Mutually exclusive with pydantic_model.",
    )
    pydantic_model: Optional[Type[BaseModel]] = Field(
        default=None,
        description="Pydantic model class for validation and schema generation. Mutually exclusive with schema.",
    )
    model_name: Optional[str] = Field(
        default=None, description="Name of the Pydantic model, if using model-based validation"
    )

    @property
    def has_schema(self) -> bool:
        """Whether this config has a schema available (either direct or via model)."""
        return bool(self.schema or self.pydantic_model)

    def model_post_init(self, __context: Any) -> None:
        """Validate mutual exclusivity of schema sources."""
        if self.schema is not None and self.pydantic_model is not None:
            raise ValueError("Cannot specify both schema and pydantic_model - they are mutually exclusive")

        if self.model_name and not self.pydantic_model:
            raise ValueError("model_name can only be specified when using pydantic_model")

    @property
    def effective_schema(self) -> Optional[Dict[str, Any]]:
        """Get the effective schema, whether from direct schema or Pydantic model."""
        if self.pydantic_model:
            return self.pydantic_model.model_json_schema()
        return self.schema

    class Config:
        arbitrary_types_allowed = True


class ValidationResult(BaseModel):
    """Result of response validation."""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    retry_count: int = 0
    original_response: Any
    formatted_response: Optional[Any] = None


class RetryConfig(BaseModel):
    """Configuration for validation retries."""

    max_retries: int = 3
    error_prompt_template: str = (
        "The previous response was invalid. Errors: {errors}\nPlease try again with a valid response."
    )


class ResponseFormat(BaseModel):
    """Response format specification with validation capabilities.

    The preferred way to create a ResponseFormat is by using the from_pydantic_model() class method:
        ResponseFormat.from_pydantic_model(MyModel)

    Direct JSON schema usage should only be used as a fallback when a Pydantic model
    is not available or practical.
    """

    format: ResponseFormatType
    response_schema: Optional[str] = Field(
        default=None,
        description="Schema for structured response formats. DEPRECATED: Prefer using pydantic_model instead.",
    )
    retry_config: Optional[RetryConfig] = Field(default_factory=RetryConfig)
    convert_to_json_schema: bool = Field(default=False, description="Whether to convert Pydantic models to JSON schema")
    pydantic_model: Optional[Type[BaseModel]] = Field(
        default=None,
        description="Pydantic model for validation. This is the preferred way to specify the response format.",
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate mutual exclusivity of schema sources and warn about deprecated usage."""
        if self.response_schema is not None and self.pydantic_model is not None:
            raise ValueError("Cannot specify both response_schema and pydantic_model - they are mutually exclusive")

        if self.response_schema is not None and not self.convert_to_json_schema:
            import warnings

            warnings.warn(
                "Direct JSON schema usage is deprecated. Prefer using from_pydantic_model() instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    @classmethod
    def from_pydantic_model(
        cls,
        model: Type[BaseModel],
        convert_to_json_schema: bool = False,
        format_type: ResponseFormatType = ResponseFormatType.JSON_SCHEMA,
    ) -> "ResponseFormat":
        """Create a ResponseFormat from a Pydantic model (preferred method).

        Args:
            model: The Pydantic model to use for validation
            convert_to_json_schema: Whether to convert the model to JSON schema (not recommended)
            format_type: The format type to use

        Returns:
            A ResponseFormat instance configured to use the Pydantic model
        """
        schema = None
        if convert_to_json_schema:
            schema = schema_to_json(model)

        return cls(
            format=format_type,
            response_schema=schema if convert_to_json_schema else None,
            convert_to_json_schema=convert_to_json_schema,
            pydantic_model=None if convert_to_json_schema else model,
        )

    @classmethod
    def from_json_schema(
        cls,
        schema: Union[str, Dict[str, Any]],
        format_type: ResponseFormatType = ResponseFormatType.JSON_SCHEMA,
    ) -> "ResponseFormat":
        """Create a ResponseFormat from a JSON schema (fallback method).

        NOTE: This method should only be used when a Pydantic model is not available.
        Consider creating a Pydantic model instead of using raw JSON schema.

        Args:
            schema: The JSON schema as a string or dict
            format_type: The format type to use

        Returns:
            A ResponseFormat instance configured to use the JSON schema
        """
        if isinstance(schema, dict):
            schema = json.dumps(schema)

        return cls(format=format_type, response_schema=schema, convert_to_json_schema=True)

    def get_structured_output_config(self) -> StructuredOutputConfig:
        """Get configuration for structured output based on format type and schema."""
        config = StructuredOutputConfig(
            format=self.format,
        )

        if self.pydantic_model and not self.convert_to_json_schema:
            # Pass the Pydantic model directly if not converting to JSON schema
            config.pydantic_model = self.pydantic_model
            config.model_name = self.pydantic_model.__name__
        elif self.response_schema:
            # Use the JSON schema if provided
            config.schema = convert_to_schema(self.response_schema)

        # Add format-specific configuration
        if self.format == ResponseFormatType.JSON_SCHEMA:
            config.response_format_override = {"type": "json_object"}

        return config

    def get_required_fields(self) -> List[str]:
        """Get the list of required fields from the response schema."""
        if not self.response_schema:
            return []

        try:
            schema = convert_to_schema(self.response_schema)
            return schema.get("required", [])
        except ValueError:
            return []

    @property
    def requires_schema(self) -> bool:
        """Whether this format type typically requires a schema."""
        return self.format in {
            ResponseFormatType.JSON,
            ResponseFormatType.JSON_SCHEMA,
            ResponseFormatType.XML,
            ResponseFormatType.YAML,
            ResponseFormatType.PYDANTIC,
        }

    @property
    def is_structured(self) -> bool:
        """Whether this format represents structured data."""
        return self.format in {
            ResponseFormatType.JSON,
            ResponseFormatType.JSON_SCHEMA,
            ResponseFormatType.XML,
            ResponseFormatType.YAML,
            ResponseFormatType.PYDANTIC,
        }

    def validate_response(self, response: str) -> ValidationResult:
        """Validate a response against the format and schema requirements."""
        result = ValidationResult(is_valid=False, original_response=response)

        try:
            formatted = self._parse_format(response)
            result.formatted_response = formatted

            if self.format == ResponseFormatType.PYDANTIC and self.pydantic_model:
                try:
                    # For PYDANTIC format, validate using the model directly
                    self.pydantic_model.model_validate_json(response)
                    result.is_valid = True
                except Exception as e:
                    result.errors.append(str(e))
            elif self.requires_schema:
                # Use get_structured_output_config to leverage effective_schema
                config = self.get_structured_output_config()
                if config.effective_schema:
                    try:
                        validate_json(formatted, json.dumps(config.effective_schema))
                        result.is_valid = True
                    except ValueError as e:
                        result.errors.append(str(e))
                else:
                    result.is_valid = True
            else:
                result.is_valid = True

        except Exception as e:
            result.errors.append(str(e))

        return result

    def _parse_format(self, response: str) -> Any:
        """Parse the response according to its format type."""
        try:
            if self.format in {ResponseFormatType.JSON, ResponseFormatType.JSON_SCHEMA, ResponseFormatType.PYDANTIC}:
                return validate_json(response)
            elif self.format == ResponseFormatType.YAML:
                return yaml.safe_load(response)
            elif self.format == ResponseFormatType.XML:
                return ET.fromstring(response)
            else:
                return response
        except Exception as e:
            raise ValueError(f"Failed to parse {self.format} response: {str(e)}")

    def generate_retry_prompt(self, validation_result: ValidationResult) -> Optional[str]:
        """Generate a prompt for retrying with invalid response."""
        if validation_result.is_valid or not self.retry_config:
            return None

        if validation_result.retry_count >= self.retry_config.max_retries:
            return None

        error_str = "\n".join(validation_result.errors)
        return self.retry_config.error_prompt_template.format(errors=error_str)


class BaseResponse(BaseModel):
    """Base class for all response types."""

    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = Field(description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")
    execution_time: Optional[float] = Field(default=None, description="Time taken to generate response in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the response")
    format: ResponseFormatType = Field(default=ResponseFormatType.TEXT, description="Format of the response content")

    model_config = ConfigDict(validate_assignment=True)


class LLMResponse(BaseResponse):
    """Response from an LLM model."""

    content: str = Field(..., description="The content of the response")
    token_usage: TokenUsage = Field(..., description="Token usage statistics")
    context_metrics: Optional[ContextMetrics] = Field(default=None, description="Context window metrics")
    validation_result: Optional[ValidationResult] = None

    def is_json(self) -> bool:
        """Check if response is in JSON format."""
        return self.format == ResponseFormatType.JSON

    def is_structured(self) -> bool:
        """Check if response is in a structured format."""
        return self.format in {ResponseFormatType.JSON, ResponseFormatType.YAML, ResponseFormatType.XML}

    def requires_validation(self) -> bool:
        """Check if response requires format validation."""
        return self.is_structured()
