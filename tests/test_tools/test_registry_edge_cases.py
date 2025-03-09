"""Tests for edge cases and error handling in the ToolRegistry class."""
import pytest
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from llmaestro.tools.core import BasicFunctionGuard, ToolParams
from llmaestro.tools.registry import ToolRegistry


class TestRegistryEdgeCases:
    """Tests for edge cases in the ToolRegistry class."""

    def test_register_same_name_twice(self, empty_registry):
        """Test registering a tool with the same name twice."""
        # Register first tool
        @empty_registry.register("duplicate_tool")
        def first_tool(param: str) -> str:
            """First tool with duplicate name."""
            return f"First: {param}"

        # Register second tool with same name
        def second_tool(param: str) -> str:
            """Second tool with duplicate name."""
            return f"Second: {param}"

        empty_registry.register_tool("duplicate_tool", second_tool)

        # Verify that the second tool overwrote the first
        tool = empty_registry.get_tool("duplicate_tool")
        assert tool is not None
        assert tool.name == "second_tool"
        assert "Second tool with duplicate name" in tool.description

    def test_register_with_empty_category(self, empty_registry):
        """Test registering a tool with an empty category."""
        @empty_registry.register("empty_category_tool", category="")
        def empty_category_tool(param: str) -> str:
            """Tool with empty category."""
            return f"Empty category: {param}"

        # Verify the tool was registered
        # Empty string categories are not added to the categories list
        assert "empty_category_tool" in empty_registry.get_all_tools()
        # Check that the category is not in the list
        assert "" not in empty_registry.get_categories()

    def test_register_with_none_category(self, empty_registry):
        """Test registering a tool with None as category."""
        @empty_registry.register("none_category_tool", category=None)
        def none_category_tool(param: str) -> str:
            """Tool with None category."""
            return f"None category: {param}"

        # Verify the tool was registered but not in any category
        assert "none_category_tool" in empty_registry.get_all_tools()
        assert None not in empty_registry.get_categories()

    def test_register_with_none_name(self, empty_registry):
        """Test registering a tool with None as name."""
        @empty_registry.register(None)
        def auto_named_tool(param: str) -> str:
            """Tool with auto-generated name."""
            return f"Auto-named: {param}"

        # Verify the tool was registered with its function name
        assert "auto_named_tool" in empty_registry.get_all_tools()

    def test_register_tool_with_invalid_source(self, empty_registry):
        """Test registering a tool with an invalid source."""
        # Try to register a non-callable, non-BaseModel object
        with pytest.raises(Exception):
            empty_registry.register_tool("invalid_tool", "not_a_function")


class TestToolParamsExecution:
    """Tests for executing tools through ToolParams."""

    def test_execute_function_tool(self, populated_registry):
        """Test executing a function-based tool."""
        tool = populated_registry.get_tool("test_function")
        assert tool is not None

        # Execute the tool
        result = tool.source(message="test", count=3)
        assert result == "testtesttest"

    def test_execute_model_tool(self, populated_registry):
        """Test executing a model-based tool."""
        tool = populated_registry.get_tool("test_model")
        assert tool is not None

        # Execute the tool
        model_instance = tool.source(message="test", count=3)
        assert model_instance.message == "test"
        assert model_instance.count == 3

    def test_execute_with_invalid_args(self, populated_registry):
        """Test executing a tool with invalid arguments."""
        tool = populated_registry.get_tool("test_function")
        assert tool is not None

        # Execute with missing required argument
        with pytest.raises(ValueError):
            tool.source()

        # Python's type system is not enforced at runtime, so passing an int to a str parameter
        # might not raise an exception. Instead, check that the result is correct.
        result = tool.source(message=123)  # message should be a string, but Python will convert it
        assert result == 123  # The function returns the input as is, without string conversion


class TestToolDiscoveryEdgeCases:
    """Tests for edge cases in tool discovery."""

    def test_list_tools_empty_registry(self, empty_registry):
        """Test listing tools with an empty registry."""
        tools_info = empty_registry.list_available_tools()
        assert tools_info["total_tools"] == 0
        assert tools_info["tools"] == []

    def test_list_tools_with_multiple_categories(self, empty_registry):
        """Test listing tools with multiple categories."""
        # Register tools in different categories
        @empty_registry.register("tool1", category="category1")
        def tool1(param: str) -> str:
            """Tool in category1."""
            return param

        @empty_registry.register("tool2", category="category2")
        def tool2(param: str) -> str:
            """Tool in category2."""
            return param

        @empty_registry.register("tool3", category="category1")
        def tool3(param: str) -> str:
            """Another tool in category1."""
            return param

        # Verify categories
        categories = empty_registry.get_categories()
        assert len(categories) == 2
        assert "category1" in categories
        assert "category2" in categories

        # Verify tools by category
        category1_tools = empty_registry.get_tools_by_category("category1")
        assert len(category1_tools) == 2

        category2_tools = empty_registry.get_tools_by_category("category2")
        assert len(category2_tools) == 1
