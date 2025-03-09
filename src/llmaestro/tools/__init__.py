"""LLMaestro Tools Package.

This package provides tools that can be used by LLMs to interact with various systems and services.

The main entry points are:
- all_tools: Central registry of all available tools
- registry: Tool registry for registering and discovering tools
- core: Core tool components like FunctionGuard and ToolParams
"""

# Import core components
from llmaestro.tools.core import FunctionGuard, BasicFunctionGuard, ToolParams

# Import registry
from llmaestro.tools.registry import ToolRegistry, get_registry, create_tool_discovery_tool

# Import all tools
from llmaestro.tools.all_tools import get_all_tools, get_tools_by_category, get_tool, get_categories, register_all_tools

# Import SQL tools
from llmaestro.tools.sql_tools import (
    create_sql_read_only_tool,
    create_sql_read_write_tool,
    SQLQueryParams,
    SQLReadOnlyGuard,
    SQLReadWriteGuard,
)

# Define what's available for import with "from llmaestro.tools import *"
__all__ = [
    # Core components
    "FunctionGuard",
    "BasicFunctionGuard",
    "ToolParams",
    # Registry
    "ToolRegistry",
    "get_registry",
    "create_tool_discovery_tool",
    # All tools
    "get_all_tools",
    "get_tools_by_category",
    "get_tool",
    "get_categories",
    "register_all_tools",
    # SQL tools
    "create_sql_read_only_tool",
    "create_sql_read_write_tool",
    "SQLQueryParams",
    "SQLReadOnlyGuard",
    "SQLReadWriteGuard",
]
