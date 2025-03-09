"""Test fixtures for tools module."""
import pytest
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from llmaestro.tools.core import BasicFunctionGuard, ToolParams
from llmaestro.tools.registry import ToolRegistry


class TestToolInput(BaseModel):
    """Test input model for tool testing."""

    message: str = Field(description="A test message")
    count: int = Field(default=1, description="Number of times to repeat the message")


def test_tool_function(message: str, count: int = 1) -> str:
    """A simple test tool function that repeats a message.

    Args:
        message: The message to repeat
        count: Number of times to repeat the message

    Returns:
        The repeated message
    """
    return message * count


@pytest.fixture
def test_function_guard() -> BasicFunctionGuard:
    """Create a basic function guard for the test tool function."""
    return BasicFunctionGuard(test_tool_function)


@pytest.fixture
def test_tool_params() -> ToolParams:
    """Create a ToolParams instance for testing."""
    return ToolParams.from_function(test_tool_function)


@pytest.fixture
def test_model_params() -> ToolParams:
    """Create a ToolParams instance from a Pydantic model for testing."""
    return ToolParams.from_pydantic(TestToolInput)


@pytest.fixture
def empty_registry() -> ToolRegistry:
    """Create an empty tool registry for testing."""
    # Create a new instance to avoid affecting other tests
    registry = ToolRegistry()
    # Clear any existing tools to ensure a clean state
    registry._tools = {}
    registry._categories = {}
    return registry


@pytest.fixture
def populated_registry(empty_registry) -> ToolRegistry:
    """Create a tool registry with pre-registered tools for testing."""
    # Register test tools
    empty_registry.register_tool("test_function", test_tool_function, category="test")
    empty_registry.register_tool("test_model", TestToolInput, category="test")

    return empty_registry
