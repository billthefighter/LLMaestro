"""All Tools Registry for LLMaestro.

This module provides a central registry of all tools available in LLMaestro.
It imports and registers tools from all tool modules in the directory.

Usage:
    ```python
    from llmaestro.tools.all_tools import get_all_tools, get_tools_by_category
    
    # Get all available tools
    all_tools = get_all_tools()
    
    # Get tools by category
    sql_tools = get_tools_by_category("database")
    file_tools = get_tools_by_category("file_system")
    ```
"""

from typing import Dict, List, Optional, Any, Union, Callable
import importlib
import inspect
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, Engine, Connection

from llmaestro.tools.registry import get_registry, ToolRegistry
from llmaestro.tools.core import ToolParams

# Import tool modules
from llmaestro.tools import sql_tools
from llmaestro.tools import file_system_tools
from llmaestro.tools import api_integration_tools
from llmaestro.tools import vector_database_tools
from llmaestro.tools import data_processing_tools
from llmaestro.tools import web_scraping_tools
from llmaestro.tools import nlp_tools
from llmaestro.tools import image_processing_tools
from llmaestro.tools import caching_tools
from llmaestro.tools import auth_tools
from llmaestro.tools import workflow_tools


# Get the registry instance
registry = get_registry()


def register_sql_tools(engine_or_connection: Optional[Union[Engine, Connection, Callable[[], Union[Engine, Connection]]]] = None) -> None:
    """Register SQL tools with the registry.
    
    Args:
        engine_or_connection: SQLAlchemy engine, connection, or callable that returns either.
            If not provided, an in-memory SQLite database will be used.
    """
    if engine_or_connection is None:
        # Create an in-memory SQLite database as a default
        engine_or_connection = create_engine("sqlite:///:memory:")
    
    # Create and register SQL tools
    read_only_tool = sql_tools.create_sql_read_only_tool(
        engine_or_connection,
        name="execute_read_only_sql",
        description="Execute a read-only SQL query (SELECT only) against the database."
    )
    
    read_write_tool = sql_tools.create_sql_read_write_tool(
        engine_or_connection,
        name="execute_sql",
        description="Execute an SQL query against the database. Can perform both read and write operations."
    )
    
    # Register the tools with the registry
    registry.register_tool("execute_read_only_sql", read_only_tool, category="database")
    registry.register_tool("execute_sql", read_write_tool, category="database")


def register_placeholder_tools() -> None:
    """Register placeholder tools for modules that are not yet implemented.
    
    This function registers placeholder tools for modules that have been defined
    but not yet fully implemented. These tools will be replaced with actual
    implementations as they are developed.
    """
    # Define a mapping of module names to categories and tool names
    placeholder_tools = {
        "file_system_tools": {
            "category": "file_system",
            "tools": [
                ("read_file", "Read a file from the file system."),
                ("write_file", "Write content to a file."),
                ("list_directory", "List the contents of a directory."),
                ("get_file_metadata", "Get metadata about a file.")
            ]
        },
        "api_integration_tools": {
            "category": "api",
            "tools": [
                ("rest_api_call", "Make a REST API call."),
                ("graphql_query", "Execute a GraphQL query.")
            ]
        },
        "vector_database_tools": {
            "category": "vector_database",
            "tools": [
                ("similarity_search", "Perform a similarity search in a vector database."),
                ("store_embedding", "Store an embedding in a vector database.")
            ]
        },
        "data_processing_tools": {
            "category": "data_processing",
            "tools": [
                ("parse_csv", "Parse a CSV file."),
                ("parse_json", "Parse a JSON string."),
                ("transform_data", "Transform data according to a schema.")
            ]
        },
        "web_scraping_tools": {
            "category": "web_scraping",
            "tools": [
                ("scrape_webpage", "Scrape content from a webpage."),
                ("extract_structured_data", "Extract structured data from HTML.")
            ]
        },
        "nlp_tools": {
            "category": "nlp",
            "tools": [
                ("extract_entities", "Extract entities from text."),
                ("sentiment_analysis", "Analyze the sentiment of text.")
            ]
        },
        "image_processing_tools": {
            "category": "image_processing",
            "tools": [
                ("ocr_image", "Extract text from an image using OCR."),
                ("describe_image", "Generate a description of an image.")
            ]
        },
        "caching_tools": {
            "category": "caching",
            "tools": [
                ("cache_result", "Cache a result for later retrieval."),
                ("get_cached_result", "Retrieve a cached result.")
            ]
        },
        "auth_tools": {
            "category": "auth",
            "tools": [
                ("authenticate_user", "Authenticate a user."),
                ("check_permissions", "Check if a user has permission to perform an action.")
            ]
        },
        "workflow_tools": {
            "category": "workflow",
            "tools": [
                ("execute_workflow", "Execute a workflow."),
                ("schedule_task", "Schedule a task for later execution.")
            ]
        }
    }
    
    # Register placeholder tools
    for module_name, module_info in placeholder_tools.items():
        category = module_info["category"]
        for tool_name, description in module_info["tools"]:
            # Create a placeholder function
            def placeholder_function(*args, **kwargs):
                module_path = f"llmaestro.tools.{module_name}"
                raise NotImplementedError(
                    f"The tool '{tool_name}' in module '{module_path}' is not yet implemented. "
                    f"It is registered as a placeholder for future implementation."
                )
            
            # Set the function name and docstring
            placeholder_function.__name__ = tool_name
            placeholder_function.__doc__ = description
            
            # Register the placeholder function as a tool
            registry.register_tool(tool_name, placeholder_function, category=category)


def register_all_tools(sql_engine: Optional[Union[Engine, Connection, Callable[[], Union[Engine, Connection]]]] = None) -> None:
    """Register all available tools with the registry.
    
    Args:
        sql_engine: SQLAlchemy engine, connection, or callable for SQL tools.
            If not provided, an in-memory SQLite database will be used.
    """
    # Register SQL tools
    register_sql_tools(sql_engine)
    
    # Register placeholder tools for modules that are not yet implemented
    register_placeholder_tools()


def get_all_tools() -> Dict[str, ToolParams]:
    """Get all registered tools.
    
    Returns:
        A dictionary mapping tool names to tool parameters.
    """
    return registry.get_all_tools()


def get_tools_by_category(category: str) -> List[ToolParams]:
    """Get all tools in a category.
    
    Args:
        category: The category to get tools for.
        
    Returns:
        A list of tool parameters for tools in the category.
    """
    return registry.get_tools_by_category(category)


def get_tool(name: str) -> Optional[ToolParams]:
    """Get a tool by name.
    
    Args:
        name: The name of the tool.
        
    Returns:
        The tool parameters, or None if the tool is not found.
    """
    return registry.get_tool(name)


def get_categories() -> List[str]:
    """Get all categories.
    
    Returns:
        A list of all categories.
    """
    return registry.get_categories()


# Register all tools when the module is imported
register_all_tools() 