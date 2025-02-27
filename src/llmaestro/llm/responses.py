"""Response types and format handling for LLM interactions."""

import json
import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, cast, Type

import jsonschema
import yaml
from pydantic import BaseModel, ConfigDict, Field

from llmaestro.llm.models import TokenUsage, ContextMetrics


class ResponseFormatType(str, Enum):
    """Enumeration of supported response formats from LLMs."""

    JSON = "json"
    JSON_SCHEMA = "json_schema"  # For structured JSON output with schema validation
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    YAML = "yaml"
    XML = "xml"


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
    """Response format specification with validation capabilities."""

    format: ResponseFormatType
    response_schema: Optional[str] = None  # Schema for structured response formats
    retry_config: Optional[RetryConfig] = Field(default_factory=RetryConfig)
    convert_to_json_schema: bool = Field(default=True, description="Whether to convert Pydantic models to JSON schema")
    pydantic_model: Optional[Type[BaseModel]] = None

    @classmethod
    def from_pydantic_model(
        cls,
        model: Type[BaseModel],
        convert_to_json_schema: bool = True,
        format_type: ResponseFormatType = ResponseFormatType.JSON_SCHEMA,
    ) -> "ResponseFormat":
        """Create a ResponseFormat from a Pydantic model.

        Args:
            model: The Pydantic model class to use for response validation
            convert_to_json_schema: Whether to convert the model to JSON schema
            format_type: The response format type to use
        """
        schema = None
        if convert_to_json_schema:
            schema = json.dumps(model.model_json_schema())

        return cls(
            format=format_type,
            response_schema=schema,
            convert_to_json_schema=convert_to_json_schema,
            pydantic_model=model,
        )

    def get_structured_output_config(self) -> Dict[str, Any]:
        """Get configuration for structured output based on format type and schema.

        This method is used by LLM interfaces to configure their structured output
        settings based on the response format requirements.

        Returns:
            Dict containing configuration for structured output, which may include:
            - schema: The JSON schema if using schema validation
            - response_format: Format-specific configuration for the LLM
            - model_info: Information about the Pydantic model if using direct model parsing
        """
        config: Dict[str, Any] = {
            "format": self.format.value,
            "requires_schema": self.requires_schema,
        }

        if self.pydantic_model:
            config["model_info"] = {
                "model_name": self.pydantic_model.__name__,
                "model_class": self.pydantic_model,
            }

        if self.response_schema:
            config["schema"] = json.loads(self.response_schema)

        # Add format-specific configuration
        if self.format == ResponseFormatType.JSON_SCHEMA:
            config["response_format"] = {"type": "json_object"}

        return config

    def get_required_fields(self) -> List[str]:
        """Get the list of required fields from the response schema.

        Returns:
            List of field names that are required according to the schema.
        """
        if not self.response_schema:
            return []

        try:
            schema = json.loads(self.response_schema)
            return schema.get("required", [])
        except json.JSONDecodeError:
            return []

    @property
    def requires_schema(self) -> bool:
        """Whether this format type typically requires a schema."""
        return self.format in {
            ResponseFormatType.JSON,
            ResponseFormatType.JSON_SCHEMA,
            ResponseFormatType.XML,
            ResponseFormatType.YAML,
        }

    @property
    def is_structured(self) -> bool:
        """Whether this format represents structured data."""
        return self.format in {
            ResponseFormatType.JSON,
            ResponseFormatType.JSON_SCHEMA,
            ResponseFormatType.XML,
            ResponseFormatType.YAML,
        }

    def validate_response(self, response: str) -> ValidationResult:
        """Validate a response against the format and schema requirements."""
        result = ValidationResult(is_valid=False, original_response=response)

        try:
            formatted = self._parse_format(response)
            result.formatted_response = formatted

            if self.requires_schema and self.response_schema:
                self._validate_against_schema(formatted, result)
            else:
                result.is_valid = True

        except Exception as e:
            result.errors.append(str(e))

        return result

    def _parse_format(self, response: str) -> Any:
        """Parse the response according to its format type."""
        try:
            if self.format == ResponseFormatType.JSON:
                return json.loads(response)
            elif self.format == ResponseFormatType.YAML:
                return yaml.safe_load(response)
            elif self.format == ResponseFormatType.XML:
                return ET.fromstring(response)
            else:
                return response
        except Exception as e:
            raise ValueError(f"Failed to parse {self.format} response: {str(e)}")

    def _validate_against_schema(self, parsed_response: Any, result: ValidationResult) -> None:
        """Validate parsed response against schema."""
        try:
            if self.format == ResponseFormatType.JSON and self.response_schema:
                schema = json.loads(cast(str, self.response_schema))
                jsonschema.validate(parsed_response, schema)
            elif self.format == ResponseFormatType.XML:
                # XML schema validation would go here
                pass
            elif self.format == ResponseFormatType.YAML and self.response_schema:
                if isinstance(parsed_response, (dict, list)):
                    schema = json.loads(cast(str, self.response_schema))
                    jsonschema.validate(parsed_response, schema)

            result.is_valid = True

        except jsonschema.ValidationError as e:
            result.errors.append(f"Schema validation error: {str(e)}")
        except Exception as e:
            result.errors.append(f"Validation error: {str(e)}")

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
