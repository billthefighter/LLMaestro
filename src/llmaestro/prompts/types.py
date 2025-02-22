"""Shared type definitions for prompts."""
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, cast

import jsonschema
import yaml
from jsonschema import ValidationError
from pydantic import BaseModel, Field, validator


class VersionInfo(BaseModel):
    """Version information for a prompt."""

    number: str
    timestamp: datetime
    author: str
    description: str
    change_type: str
    git_commit: Optional[str] = None

    def model_dump_json(self, **kwargs) -> str:
        data = self.model_dump(**kwargs)
        if "timestamp" in data:
            data["timestamp"] = data["timestamp"].isoformat()
        return json.dumps(data, **kwargs)


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


class ResponseFormatType(str, Enum):
    """Enumeration of supported response formats from LLMs."""

    JSON = "json"
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    YAML = "yaml"
    XML = "xml"


class ResponseFormat(BaseModel):
    """Response format specification."""

    format: ResponseFormatType
    response_schema: Optional[str] = Field(None, alias="schema")  # Renamed to avoid shadowing
    retry_config: Optional[RetryConfig] = Field(default_factory=RetryConfig)

    @validator("response_schema")
    def validate_schema(cls, v: Optional[str], values: Dict) -> Optional[str]:
        """Validate that schema is present for formats that require it."""
        if not v and "format" in values:
            format_type = values["format"]
            if format_type in {ResponseFormatType.JSON, ResponseFormatType.XML, ResponseFormatType.YAML}:
                raise ValueError(f"Schema is required for format type {format_type}")
        return v

    @property
    def requires_schema(self) -> bool:
        """Whether this format type typically requires a schema."""
        return self.format in {ResponseFormatType.JSON, ResponseFormatType.XML, ResponseFormatType.YAML}

    @property
    def is_structured(self) -> bool:
        """Whether this format represents structured data."""
        return self.format in {ResponseFormatType.JSON, ResponseFormatType.XML, ResponseFormatType.YAML}

    def validate_response(self, response: str) -> ValidationResult:
        """Validate a response against the format and schema requirements.

        Args:
            response: The raw response string from the LLM

        Returns:
            ValidationResult containing validation status and any errors
        """
        result = ValidationResult(is_valid=False, original_response=response)

        try:
            # First validate the basic format
            formatted = self._parse_format(response)
            result.formatted_response = formatted

            # Then validate against schema if required
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
                return response  # For TEXT, MARKDOWN, CODE
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
                # Would need to implement XMLSchema validation
                pass
            elif self.format == ResponseFormatType.YAML and self.response_schema:
                # Convert YAML to JSON and validate against JSON schema
                if isinstance(parsed_response, (dict, list)):
                    schema = json.loads(cast(str, self.response_schema))
                    jsonschema.validate(parsed_response, schema)

            result.is_valid = True

        except ValidationError as e:
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


class PromptMetadata(BaseModel):
    """Enhanced metadata for prompts."""

    type: str
    expected_response: ResponseFormat
    model_requirements: Optional[Dict] = None
    decomposition: Optional[Dict] = None
    tags: List[str] = Field(default_factory=list)
    is_active: bool = True
