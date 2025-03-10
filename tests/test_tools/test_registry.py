"""Tests for the ToolRegistry class."""
import pytest
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from llmaestro.tools.core import BasicFunctionGuard, ToolParams
from llmaestro.tools.registry import ToolRegistry, get_registry, create_tool_discovery_tool


class TestRegistryInstantiation:
    """Tests for the instantiation behavior of ToolRegistry."""

    def test_instance_creation(self):
        """Test that ToolRegistry creates separate instances."""
        registry1 = ToolRegistry()
        registry2 = ToolRegistry()
        # Each instance should be unique
        assert registry1 is not registry2

        # The create class method should also return a new instance
        registry3 = ToolRegistry.create()
        assert registry1 is not registry3
        assert registry2 is not registry3

    def test_get_registry_function(self):
        """Test that get_registry() returns a new instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        # Each call should return a new instance
        assert registry1 is not registry2


class TestRegistryRegistration:
    """Tests for tool registration functionality."""

    def test_register_decorator(self, empty_registry):
        """Test registering a tool using the decorator."""

        @empty_registry.register("decorated_tool")
        def test_decorated_tool(param: str) -> str:
            """A test tool registered with a decorator."""
            return f"Decorated: {param}"

        # Verify the tool was registered
        tools = empty_registry.get_all_tools()
        assert "decorated_tool" in tools
        # The name in the ToolParams is the function name, not the registered name
        assert tools["decorated_tool"].name == "test_decorated_tool"

    def test_register_with_category(self, empty_registry):
        """Test registering a tool with a category."""

        @empty_registry.register("categorized_tool", category="test_category")
        def test_categorized_tool(param: str) -> str:
            """A test tool with a category."""
            return f"Categorized: {param}"

        # Verify the tool was registered with the category
        assert "test_category" in empty_registry.get_categories()
        category_tools = empty_registry.get_tools_by_category("test_category")
        assert len(category_tools) == 1
        # The name in the ToolParams is the function name, not the registered name
        assert category_tools[0].name == "test_categorized_tool"

    def test_register_tool_function(self, empty_registry):
        """Test registering a tool using register_tool with a function."""

        def direct_tool(param: str) -> str:
            """A test tool registered directly."""
            return f"Direct: {param}"

        empty_registry.register_tool("direct_tool", direct_tool)

        # Verify the tool was registered
        tools = empty_registry.get_all_tools()
        assert "direct_tool" in tools
        # The name in the ToolParams is the function name, not the registered name
        assert tools["direct_tool"].name == "direct_tool"

    def test_register_tool_model(self, empty_registry):
        """Test registering a tool using register_tool with a Pydantic model."""

        class TestModel(BaseModel):
            """A test model for registration."""
            field: str = Field(description="A test field")

        empty_registry.register_tool("model_tool", TestModel)

        # Verify the tool was registered
        tools = empty_registry.get_all_tools()
        assert "model_tool" in tools
        # The name in the ToolParams is the class name, not the registered name
        assert tools["model_tool"].name == "TestModel"
        assert "field" in tools["model_tool"].parameters["properties"]

    def test_register_tool_params(self, empty_registry, test_tool_params):
        """Test registering a tool using register_tool with ToolParams."""
        empty_registry.register_tool("params_tool", test_tool_params)

        # Verify the tool was registered
        tools = empty_registry.get_all_tools()
        assert "params_tool" in tools
        # The name in the ToolParams is preserved from the original ToolParams
        assert tools["params_tool"].name == "test_tool_function"


class TestRegistryRetrieval:
    """Tests for tool retrieval functionality."""

    def test_get_tool(self, populated_registry):
        """Test retrieving a tool by name."""
        tool = populated_registry.get_tool("test_function")
        assert tool is not None
        # The name in the ToolParams is the function name
        assert tool.name == "test_tool_function"

        # Test non-existent tool
        assert populated_registry.get_tool("non_existent") is None

    def test_get_all_tools(self, populated_registry):
        """Test retrieving all tools."""
        tools = populated_registry.get_all_tools()
        assert len(tools) == 2
        assert "test_function" in tools
        assert "test_model" in tools

    def test_get_tools_by_category(self, populated_registry):
        """Test retrieving tools by category."""
        tools = populated_registry.get_tools_by_category("test")
        assert len(tools) == 2

        # Test non-existent category
        assert populated_registry.get_tools_by_category("non_existent") == []

    def test_get_categories(self, populated_registry):
        """Test retrieving all categories."""
        categories = populated_registry.get_categories()
        assert len(categories) == 1
        assert "test" in categories


class TestToolDiscovery:
    """Tests for the tool discovery functionality."""

    def test_list_available_tools(self, populated_registry):
        """Test listing available tools."""
        tools_info = populated_registry.list_available_tools()
        assert "tools" in tools_info
        assert len(tools_info["tools"]) == 2

        # Test with category filter
        tools_info = populated_registry.list_available_tools(category="test")
        assert len(tools_info["tools"]) == 2

        # Test with non-existent category
        tools_info = populated_registry.list_available_tools(category="non_existent")
        assert len(tools_info["tools"]) == 0

    def test_create_tool_discovery_tool(self, populated_registry):
        """Test creating a tool discovery tool."""
        discovery_tool = create_tool_discovery_tool(populated_registry)
        assert isinstance(discovery_tool, ToolParams)
        # The name is explicitly set in create_tool_discovery_tool
        assert discovery_tool.name == "list_available_tools"

        # Test executing the discovery tool
        result = discovery_tool.source()
        assert "tools" in result
        assert len(result["tools"]) == 2

        # Test with category
        result = discovery_tool.source(category="test")
        assert len(result["tools"]) == 2

        # Test with non-existent category
        result = discovery_tool.source(category="non_existent")
        assert len(result["tools"]) == 0
