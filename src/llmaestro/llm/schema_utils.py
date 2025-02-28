"""Utilities for handling JSON schemas and validation."""

import json
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel


def convert_to_schema(schema_input: Union[Dict[str, Any], Type[BaseModel], str]) -> Dict[str, Any]:
    """Convert various schema inputs to a dictionary representation.

    Args:
        schema_input: Either a dictionary schema, Pydantic model class, or JSON string

    Returns:
        Dict representation of the schema

    Raises:
        ValueError: If the input is invalid or cannot be converted
    """
    if isinstance(schema_input, dict):
        return schema_input
    elif isinstance(schema_input, str):
        try:
            return json.loads(schema_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON schema string: {e}")
    elif isinstance(schema_input, type) and issubclass(schema_input, BaseModel):
        return schema_input.model_json_schema()
    else:
        raise ValueError(f"Unsupported schema input type: {type(schema_input)}")


def schema_to_json(schema: Union[Dict[str, Any], Type[BaseModel], str]) -> str:
    """Convert a schema to its JSON string representation.

    Args:
        schema: Either a dictionary schema, Pydantic model class, or JSON string

    Returns:
        JSON string representation of the schema
    """
    if isinstance(schema, str):
        # Validate it's proper JSON by parsing and re-stringifying
        try:
            return json.dumps(json.loads(schema))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON schema string: {e}")

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        # For Pydantic models, use model_json_schema to get complete schema including nested models
        return json.dumps(schema.model_json_schema())

    if isinstance(schema, dict):
        # Process dictionary schema, converting any nested Pydantic models
        processed_schema = {}
        for key, value in schema.items():
            if isinstance(value, type) and issubclass(value, BaseModel):
                processed_schema[key] = value.model_json_schema()
            elif isinstance(value, dict):
                # Recursively process nested dictionaries
                processed_schema[key] = json.loads(schema_to_json(value))
            else:
                processed_schema[key] = value
        return json.dumps(processed_schema)

    raise ValueError(f"Unsupported schema type: {type(schema)}")


def validate_json(
    data: Union[str, Dict[str, Any]], schema: Optional[Union[Dict[str, Any], Type[BaseModel], str]] = None
) -> Dict[str, Any]:
    """Validate and parse JSON data, optionally against a schema.

    Args:
        data: JSON string or dictionary to validate
        schema: Optional schema to validate against

    Returns:
        Parsed and validated dictionary

    Raises:
        ValueError: If the data is invalid JSON or fails schema validation
    """
    # Parse JSON if needed
    if isinstance(data, str):
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
    else:
        parsed_data = data

    # Validate against schema if provided
    if schema:
        schema_dict = convert_to_schema(schema)
        if isinstance(schema_dict, dict) and schema_dict.get("type") == "object":
            # If schema is a Pydantic model, use its validation
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate(parsed_data).model_dump()
            # Otherwise use jsonschema validation
            import jsonschema

            try:
                jsonschema.validate(parsed_data, schema_dict)
            except jsonschema.ValidationError as e:
                raise ValueError(f"Schema validation failed: {e}")

    return parsed_data
