"""Response types and format handling for LLM interactions."""

import json
import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, cast

import jsonschema
import yaml
from pydantic import BaseModel, ConfigDict, Field

from llmaestro.llm.models import TokenUsage, ContextMetrics


class ResponseFormatType(str, Enum):
    """Enumeration of supported response formats from LLMs."""

    JSON = "json"
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
    schema: Optional[str] = None  # Schema for structured response formats
    retry_config: Optional[RetryConfig] = Field(default_factory=RetryConfig)

    @property
    def requires_schema(self) -> bool:
        """Whether this format type typically requires a schema."""
        return self.format in {ResponseFormatType.JSON, ResponseFormatType.XML, ResponseFormatType.YAML}

    @property
    def is_structured(self) -> bool:
        """Whether this format represents structured data."""
        return self.format in {ResponseFormatType.JSON, ResponseFormatType.XML, ResponseFormatType.YAML}

    def validate_response(self, response: str) -> ValidationResult:
        """Validate a response against the format and schema requirements."""
        result = ValidationResult(is_valid=False, original_response=response)

        try:
            formatted = self._parse_format(response)
            result.formatted_response = formatted

            if self.requires_schema and self.schema:
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
            if self.format == ResponseFormatType.JSON and self.schema:
                schema = json.loads(cast(str, self.schema))
                jsonschema.validate(parsed_response, schema)
            elif self.format == ResponseFormatType.XML:
                # XML schema validation would go here
                pass
            elif self.format == ResponseFormatType.YAML and self.schema:
                if isinstance(parsed_response, (dict, list)):
                    schema = json.loads(cast(str, self.schema))
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
