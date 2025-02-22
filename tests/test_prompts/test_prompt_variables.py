"""Tests for prompt variable functionality."""
import pytest
import json
from typing import Any, Callable, Type, Union

from llmaestro.prompts.base import PromptVariable, SerializableType


@pytest.fixture
def string_variable() -> PromptVariable:
    """Basic string variable for testing."""
    return PromptVariable(
        name="test_var",
        description="Test variable",
        expected_input_type=SerializableType.STRING
    )


@pytest.fixture
def list_variable() -> PromptVariable:
    """List variable with custom conversion template."""
    return PromptVariable(
        name="items",
        description="List of items",
        expected_input_type=SerializableType.LIST,
        string_conversion_template=lambda x: "\n".join(f"- {item}" for item in x)
    )


@pytest.fixture
def schema_variable() -> PromptVariable:
    """Schema variable for testing JSON conversion."""
    return PromptVariable(
        name="schema",
        description="JSON schema",
        expected_input_type=SerializableType.SCHEMA
    )


def test_basic_string_conversion(string_variable: PromptVariable):
    """Test basic string conversion with default template."""
    # Act
    result = string_variable.convert_value("test")

    # Assert
    assert result == "test"
    assert isinstance(result, str)


def test_number_to_string_conversion(string_variable: PromptVariable):
    """Test converting numbers to strings."""
    # Act & Assert
    assert string_variable.convert_value(123) == "123"
    assert string_variable.convert_value(3.14) == "3.14"


def test_list_with_custom_template(list_variable: PromptVariable):
    """Test list conversion with custom template."""
    # Arrange
    items = ["apple", "banana", "orange"]

    # Act
    result = list_variable.convert_value(items)

    # Assert
    assert result == "- apple\n- banana\n- orange"
    assert isinstance(result, str)


@pytest.mark.parametrize("template,value,expected", [
    ("Value: {value}", "test", "Value: test"),
    ("Count: {value}", 42, "Count: 42"),
    (lambda x: x.upper(), "hello", "HELLO"),
    (lambda x: ", ".join(x), ["a", "b", "c"], "a, b, c"),
])
def test_custom_conversion_templates(
    template: Any,
    value: Any,
    expected: str
):
    """Test different types of conversion templates."""
    # Arrange
    var = PromptVariable(
        name="test",
        string_conversion_template=template
    )

    # Act
    result = var.convert_value(value)

    # Assert
    assert result == expected


def test_schema_conversion_dict(schema_variable: PromptVariable):
    """Test converting dict to JSON schema string."""
    # Arrange
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }

    # Act
    result = schema_variable.convert_value(schema)

    # Assert
    assert isinstance(result, str)
    # Verify it's valid JSON
    parsed = json.loads(result)
    assert parsed == schema


def test_schema_conversion_json_string(schema_variable: PromptVariable):
    """Test accepting valid JSON string for schema."""
    # Arrange
    json_str = '{"type": "object", "properties": {"name": {"type": "string"}}}'

    # Act
    result = schema_variable.convert_value(json_str)

    # Assert
    assert result == json_str
    # Verify it's valid JSON
    json.loads(result)


@pytest.mark.parametrize("invalid_value,expected_error", [
    (None, ValueError),  # None value
    (lambda x: x, ValueError),  # Non-serializable value for schema
    ('{"invalid": json}', ValueError),  # Invalid JSON string for schema
])
def test_invalid_conversions(
    schema_variable: PromptVariable,
    invalid_value: Any,
    expected_error: Type[Exception]
):
    """Test error cases for value conversion."""
    with pytest.raises(expected_error):
        schema_variable.convert_value(invalid_value)


def test_invalid_template_function(string_variable: PromptVariable):
    """Test error handling with failing template function."""
    # Arrange
    def failing_template(x: Any) -> str:
        raise RuntimeError("Template error")

    string_variable.string_conversion_template = failing_template

    # Act & Assert
    with pytest.raises(ValueError, match="Function passed to string_conversion_template.*"):
        string_variable.convert_value("test")


def test_invalid_template_type(string_variable: PromptVariable):
    """Test error handling with invalid template type."""
    # Arrange - using an integer which is an invalid template type
    string_variable.string_conversion_template = 123  # type: ignore

    # Act & Assert
    with pytest.raises(ValueError, match="Invalid string_conversion_template for variable test_var"):
        string_variable.convert_value("test")
