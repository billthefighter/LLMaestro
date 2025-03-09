"""Tests for tool parameter generation functionality."""
import inspect
from typing import List, Optional, Callable, Any
from pydantic import BaseModel, Field
import os
import pytest
import json

from llmaestro.tools.core import ToolParams, FunctionGuard, BasicFunctionGuard

# Test Fixtures
@pytest.fixture
def sample_sync_function():
    """Sample synchronous function for testing."""
    def test_function(name: str, count: int = 5, tags: Optional[List[str]] = None) -> str:
        """Test function docstring."""
        return f"Hello {name} {count} times"
    return test_function


@pytest.fixture
def sample_async_function():
    """Sample asynchronous function for testing."""
    async def test_async_function(query: str, limit: int = 10) -> List[str]:
        """Test async function docstring."""
        return [query] * limit
    return test_async_function


@pytest.fixture
def sample_pydantic_model():
    """Sample Pydantic model for testing."""
    class TestModel(BaseModel):
        """Test model docstring."""
        name: str = Field(description="The name field")
        age: int = Field(default=0, description="The age field")
        tags: Optional[List[str]] = Field(default=None, description="Optional tags")

    return TestModel


# New fixtures for FunctionGuard testing
@pytest.fixture
def risky_file_function():
    """Sample function that attempts file operations."""
    def read_file(path: str) -> str:
        """Read contents of a file."""
        with open(path) as f:
            return f.read()
    return read_file


class FileSystemGuard(BasicFunctionGuard):
    """Custom guard for file system operations."""

    def __init__(self, func: Callable, allowed_paths: List[str]):
        super().__init__(func)
        self.allowed_paths = allowed_paths

    def is_safe_to_run(self, **kwargs: Any) -> bool:
        if not super().is_safe_to_run(**kwargs):
            return False

        path = kwargs.get('path')
        if not path:
            return False

        return any(str(path).startswith(allowed) for allowed in self.allowed_paths)


# Test Cases for FunctionGuard
def test_basic_function_guard(sample_sync_function):
    """Test BasicFunctionGuard with valid and invalid arguments."""
    guard = BasicFunctionGuard(sample_sync_function)

    # Test valid arguments
    assert guard.is_safe_to_run(name="test", count=3)
    result = guard(name="test", count=3)
    assert result == "Hello test 3 times"

    # Test invalid arguments
    assert not guard.is_safe_to_run(invalid_arg="test")
    with pytest.raises(ValueError):
        guard(invalid_arg="test")


def test_custom_function_guard(risky_file_function, tmp_path):
    """Test custom FunctionGuard implementation with file system safety checks."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    # Create guard with allowed path
    guard = FileSystemGuard(risky_file_function, allowed_paths=[str(tmp_path)])

    # Test with allowed path
    assert guard.is_safe_to_run(path=str(test_file))
    result = guard(path=str(test_file))
    assert result == "test content"

    # Test with disallowed path
    disallowed_path = "/etc/passwd"
    assert not guard.is_safe_to_run(path=disallowed_path)
    with pytest.raises(ValueError):
        guard(path=disallowed_path)


def test_pydantic_model_guard(sample_pydantic_model):
    """Test FunctionGuard with Pydantic model."""
    guard = BasicFunctionGuard(sample_pydantic_model)

    # Test valid model instantiation
    assert guard.is_safe_to_run(name="test")
    result = guard(name="test")
    assert isinstance(result, sample_pydantic_model)
    assert result.name == "test"
    assert result.age == 0  # default value

    # Test invalid model instantiation
    assert not guard.is_safe_to_run(invalid_field="test")
    with pytest.raises(ValueError):
        guard(invalid_field="test")


# Test Cases for from_function
def test_sync_function_conversion(sample_sync_function):
    """Test converting a synchronous function to ToolParams."""
    tool_params = ToolParams.from_function(sample_sync_function)

    assert tool_params.name == "test_function"
    assert "Test function docstring" in tool_params.description
    assert not tool_params.is_async
    assert tool_params.return_type == str

    # Check parameters schema
    assert tool_params.parameters["type"] == "object"
    properties = tool_params.parameters["properties"]
    assert "name" in properties
    assert "count" in properties
    assert "tags" in properties
    assert properties["name"]["type"] == "string"
    assert properties["count"]["type"] == "integer"  # JSON Schema type
    assert "default" in properties["count"]
    assert properties["count"]["default"] == 5
    assert "name" in tool_params.parameters["required"]


@pytest.mark.asyncio
async def test_async_function_conversion(sample_async_function):
    """Test converting an asynchronous function to ToolParams."""
    tool_params = ToolParams.from_function(sample_async_function)

    assert tool_params.name == "test_async_function"
    assert "Test async function docstring" in tool_params.description
    assert tool_params.is_async
    assert tool_params.return_type == List[str]

    # Check parameters schema
    assert tool_params.parameters["type"] == "object"
    properties = tool_params.parameters["properties"]
    assert "query" in properties
    assert "limit" in properties
    assert properties["query"]["type"] == "string"
    assert properties["limit"]["type"] == "integer"  # JSON Schema type
    assert "default" in properties["limit"]
    assert properties["limit"]["default"] == 10
    assert "query" in tool_params.parameters["required"]


def test_function_without_annotations(sample_sync_function):
    """Test converting a function without type annotations."""
    def untyped_function(name, count=5):
        """Untyped function."""
        return f"{name} {count}"

    tool_params = ToolParams.from_function(untyped_function)

    assert tool_params.name == "untyped_function"
    assert tool_params.parameters["type"] == "object"
    properties = tool_params.parameters["properties"]
    assert properties["name"]["type"] == "string"  # Default type
    assert properties["count"]["type"] == "string"  # Default type
    assert "default" in properties["count"]
    assert properties["count"]["default"] == 5


# Test Cases for from_pydantic
def test_pydantic_model_conversion(sample_pydantic_model):
    """Test converting a Pydantic model to ToolParams."""
    tool_params = ToolParams.from_pydantic(sample_pydantic_model)

    assert tool_params.name == "TestModel"
    assert "Test model docstring" in tool_params.description
    assert tool_params.return_type == sample_pydantic_model

    # Check schema
    schema = tool_params.parameters
    assert "properties" in schema
    properties = schema["properties"]

    # Check field definitions
    assert "name" in properties
    assert properties["name"]["description"] == "The name field"
    assert properties["name"]["type"] == "string"

    assert "age" in properties
    assert properties["age"]["description"] == "The age field"
    assert properties["age"]["type"] == "integer"
    assert properties["age"]["default"] == 0

    assert "tags" in properties
    assert properties["tags"]["description"] == "Optional tags"
    # For Optional[List[str]], we expect an anyOf schema
    assert "anyOf" in properties["tags"]
    array_type = next(t for t in properties["tags"]["anyOf"] if t["type"] == "array")
    assert array_type["items"]["type"] == "string"


def test_pydantic_model_required_fields(sample_pydantic_model):
    """Test that required fields are correctly identified in Pydantic model conversion."""
    tool_params = ToolParams.from_pydantic(sample_pydantic_model)

    required_fields = tool_params.parameters.get("required", [])
    assert "name" in required_fields  # name is required
    assert "age" not in required_fields  # age has default value
    assert "tags" not in required_fields  # tags is optional


@pytest.mark.parametrize("input_type,expected_type", [
    (str, "string"),
    (int, "integer"),
    (float, "number"),
    (bool, "boolean"),
    (List[str], "array"),
])
def test_type_conversions(input_type, expected_type):
    """Test various type conversions in parameter schemas."""
    def test_func(param: input_type):
        pass

    tool_params = ToolParams.from_function(test_func)
    param_type = tool_params.parameters["properties"]["param"]["type"]
    assert param_type == expected_type  # Use JSON Schema types
