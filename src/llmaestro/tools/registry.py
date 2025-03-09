"""Tool Registry for LLMaestro.

This module provides a registry for tools in LLMaestro. The registry itself is also a tool
that can be used to retrieve all registered tools.

Key components:
- ToolRegistry: A singleton class that serves as a registry for tools
- register_tool: A decorator for registering tools with the registry
"""

from typing import Dict, List, Optional, Type, Union, Callable, Any
from functools import wraps
import inspect

from pydantic import BaseModel, Field

from llmaestro.tools.core import ToolParams, BasicFunctionGuard, FunctionGuard


class ToolRegistry:
    """A registry for tools in LLMaestro.
    
    This class serves as a singleton registry for tools in LLMaestro. It allows tools
    to be registered and retrieved. The registry itself is also a tool that can be used
    to retrieve all registered tools.
    
    Usage:
        ```python
        # Get the registry instance
        registry = ToolRegistry.get_instance()
        
        # Register a tool
        @registry.register("my_tool")
        def my_tool():
            pass
            
        # Or register a tool directly
        registry.register_tool("my_other_tool", my_other_tool)
        
        # Get a tool by name
        tool = registry.get_tool("my_tool")
        
        # Get all tools
        all_tools = registry.get_all_tools()
        
        # Use the registry as a tool
        tools_info = registry.list_available_tools()
        ```
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Initialize instance variables only once
        if not hasattr(self, '_tools'):
            self._tools: Dict[str, ToolParams] = {}
            self._categories: Dict[str, List[str]] = {}
    
    @classmethod
    def get_instance(cls) -> 'ToolRegistry':
        """Get the singleton instance of the registry."""
        return cls()
    
    def register(self, name: Optional[str] = None, category: Optional[str] = None):
        """Decorator for registering a tool with the registry.
        
        Args:
            name: The name of the tool. If not provided, the function name will be used.
            category: The category of the tool. Used for organizing tools.
            
        Returns:
            A decorator function that registers the decorated function as a tool.
        """
        def decorator(func_or_model: Union[Callable, Type[BaseModel]]):
            nonlocal name
            if name is None:
                name = func_or_model.__name__
                
            self.register_tool(name, func_or_model, category)
            
            @wraps(func_or_model)
            def wrapper(*args, **kwargs):
                return func_or_model(*args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def register_tool(self, name: str, func_or_model: Union[Callable, Type[BaseModel], ToolParams], 
                     category: Optional[str] = None) -> None:
        """Register a tool with the registry.
        
        Args:
            name: The name of the tool.
            func_or_model: The function, Pydantic model, or ToolParams to register.
            category: The category of the tool. Used for organizing tools.
        """
        if isinstance(func_or_model, ToolParams):
            tool_params = func_or_model
        elif inspect.isclass(func_or_model) and issubclass(func_or_model, BaseModel):
            tool_params = ToolParams.from_pydantic(func_or_model)
        else:
            guard = BasicFunctionGuard(func_or_model)
            tool_params = ToolParams.from_function(func_or_model)
            tool_params.source = guard
            
        self._tools[name] = tool_params
        
        if category:
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(name)
    
    def get_tool(self, name: str) -> Optional[ToolParams]:
        """Get a tool by name.
        
        Args:
            name: The name of the tool.
            
        Returns:
            The tool parameters, or None if the tool is not found.
        """
        return self._tools.get(name)
    
    def get_all_tools(self) -> Dict[str, ToolParams]:
        """Get all registered tools.
        
        Returns:
            A dictionary mapping tool names to tool parameters.
        """
        return self._tools.copy()
    
    def get_tools_by_category(self, category: str) -> List[ToolParams]:
        """Get all tools in a category.
        
        Args:
            category: The category to get tools for.
            
        Returns:
            A list of tool parameters for tools in the category.
        """
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def get_categories(self) -> List[str]:
        """Get all categories.
        
        Returns:
            A list of all categories.
        """
        return list(self._categories.keys())
    
    def list_available_tools(self, category: Optional[str] = None) -> Dict[str, Any]:
        """List all available tools, optionally filtered by category.
        
        This method is designed to be used as a tool itself, allowing an LLM to
        discover what tools are available.
        
        Args:
            category: Optional category to filter tools by.
            
        Returns:
            A dictionary containing information about available tools.
        """
        result = {
            "total_tools": len(self._tools),
            "categories": list(self._categories.keys()),
        }
        
        if category:
            tools_in_category = self.get_tools_by_category(category)
            result["category"] = category
            result["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
                for tool in tools_in_category
            ]
        else:
            result["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
                for tool in self._tools.values()
            ]
            
        return result


# Create a tool from the registry's list_available_tools method
def create_tool_discovery_tool(registry: Optional[ToolRegistry] = None) -> ToolParams:
    """Create a tool for discovering available tools.
    
    Args:
        registry: The tool registry to use. If not provided, the singleton instance will be used.
        
    Returns:
        A ToolParams object representing the tool discovery tool.
    """
    if registry is None:
        registry = ToolRegistry.get_instance()
        
    # Create a function that calls the registry's list_available_tools method
    def list_tools(category: Optional[str] = None) -> Dict[str, Any]:
        """List all available tools, optionally filtered by category.
        
        Args:
            category: Optional category to filter tools by.
            
        Returns:
            A dictionary containing information about available tools.
        """
        return registry.list_available_tools(category)
    
    # Create a guard for the function
    guard = BasicFunctionGuard(list_tools)
    
    # Create the tool parameters
    tool = ToolParams.from_function(list_tools)
    tool.name = "list_available_tools"
    tool.description = "List all available tools, optionally filtered by category."
    tool.source = guard
    
    return tool


# Register the tool discovery tool with the registry
registry = ToolRegistry.get_instance()
registry.register_tool("list_available_tools", create_tool_discovery_tool())


# Convenience function to get the registry instance
def get_registry() -> ToolRegistry:
    """Get the singleton instance of the tool registry.
    
    Returns:
        The singleton instance of the tool registry.
    """
    return ToolRegistry.get_instance()
