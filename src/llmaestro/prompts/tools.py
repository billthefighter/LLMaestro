from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional, Callable, Type, Union
import inspect
from abc import ABC, abstractmethod


class FunctionGuard(ABC):
    """Abstract base class for guarding function execution with safety checks.

    FunctionGuard provides a framework for safely executing functions by validating
    their inputs before execution. It acts as a protective layer between the LLM
    and the actual function execution, ensuring that only safe operations are performed.

    Key Features:
        - Input validation before function execution
        - Support for both regular functions and Pydantic models
        - Extensible safety checks through subclassing
        - Consistent interface for function execution

    Usage:
        To implement a custom guard:
        ```python
        class CustomGuard(FunctionGuard):
            def __init__(self, func, **guard_params):
                self._func = func
                self._guard_params = guard_params

            @property
            def function(self):
                return self._func

            def is_safe_to_run(self, **kwargs):
                # Implement custom safety checks
                return True  # or False based on checks
        ```

        To use a guard:
        ```python
        def risky_function(path: str):
            return open(path).read()

        guard = CustomGuard(risky_function, allowed_paths=['/safe/path'])
        result = guard(path='/safe/path/file.txt')  # Executes if safe
        ```

    Safety Considerations:
        When implementing a guard, consider:
        - Input validation (types, ranges, formats)
        - Resource access (files, network, system)
        - Resource limits (memory, time, CPU)
        - Security implications (command injection, path traversal)
        - Business logic constraints

    See Also:
        - BasicFunctionGuard: Default implementation with basic safety checks
        - ToolParams: Uses FunctionGuard for LLM tool execution
    """

    @property
    @abstractmethod
    def function(self) -> Union[Callable, Type[BaseModel]]:
        """The callable function or Pydantic model to execute."""
        pass

    @abstractmethod
    def is_safe_to_run(self, **kwargs: Any) -> bool:
        """Validate that the function is safe to run with the given arguments.

        This method should be implemented by subclasses to provide safety checks
        specific to their use case. Some example checks could include:
        - Validating argument types and ranges
        - Checking for dangerous system commands
        - Validating file paths are in allowed directories
        - Checking resource usage limits
        - Validating network access permissions

        Args:
            **kwargs: The arguments that will be passed to the function

        Returns:
            bool: True if safe to run, False otherwise
        """
        pass

    def __call__(self, **kwargs: Any) -> Any:
        """Execute the function if it passes safety checks.

        Args:
            **kwargs: Arguments to pass to the function

        Returns:
            The result of executing the function

        Raises:
            ValueError: If the function is not safe to run with given arguments
        """
        if not self.is_safe_to_run(**kwargs):
            raise ValueError(f"Function {self.function.__name__} is not safe to run with arguments: {kwargs}")

        if isinstance(self.function, type) and issubclass(self.function, BaseModel):
            return self.function(**kwargs)
        else:
            return self.function(**kwargs)


class BasicFunctionGuard(FunctionGuard):
    """Default implementation of FunctionGuard with basic safety checks."""

    def __init__(self, func: Union[Callable, Type[BaseModel]]):
        self._func = func

    @property
    def function(self) -> Union[Callable, Type[BaseModel]]:
        return self._func

    def is_safe_to_run(self, **kwargs: Any) -> bool:
        """Basic safety checks for function execution.

        Currently implements:
        - Type validation for Pydantic models
        - Parameter validation for functions

        Subclass and override this method to add more specific safety checks.
        """
        if isinstance(self._func, type) and issubclass(self._func, BaseModel):
            try:
                # Validate kwargs against Pydantic model
                self._func(**kwargs)
                return True
            except Exception:
                return False
        else:
            # For regular functions, validate against signature
            try:
                sig = inspect.signature(self._func)
                sig.bind(**kwargs)
                return True
            except Exception:
                return False


class ToolParams(BaseModel):
    """Parameters for a tool/function that can be used by an LLM."""

    name: str = Field(description="The function's name")
    description: str = Field(description="Details on when and how to use the function")
    parameters: Dict[str, Any] = Field(description="JSON schema defining the function's input arguments")
    return_type: Optional[Any] = Field(default=None, description="The return type of the function if available")
    is_async: bool = Field(default=False, description="Whether the function is async")
    source: BasicFunctionGuard = Field(description="The guarded function or model that this tool executes")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments.

        Args:
            **kwargs: Arguments to pass to the function.

        Returns:
            The raw result of executing the function.

        Raises:
            ValueError: If no source function is available.
            TypeError: If arguments don't match the function signature.
        """
        try:
            result = self.source(**kwargs)

            if self.is_async:
                result = await result

            return result

        except Exception as e:
            raise TypeError(f"Failed to execute {self.name}: {str(e)}") from e

    @staticmethod
    def _get_parameter_schema(param: inspect.Parameter) -> Dict[str, Any]:
        """Get JSON Schema for a parameter using Pydantic's type system."""
        from pydantic import TypeAdapter

        # Get type annotation or default to str
        annotation = param.annotation if param.annotation != inspect.Parameter.empty else str

        # Get JSON schema for the type
        schema = TypeAdapter(annotation).json_schema()

        # Add default value if present
        if param.default != inspect.Parameter.empty:
            schema["default"] = param.default

        return schema

    @classmethod
    def from_function(cls, func: Callable) -> "ToolParams":
        """Generate tool parameters from a function."""
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        parameters = {}
        required = []

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            # Get parameter schema using Pydantic
            param_info = cls._get_parameter_schema(param)

            # Handle required fields
            if param.default == inspect.Parameter.empty:
                required.append(name)

            parameters[name] = param_info

        param_schema = {"type": "object", "properties": parameters, "required": required}

        # Check if async
        is_async = inspect.iscoroutinefunction(func)

        # Get return type if available
        return_type = None
        if sig.return_annotation != inspect.Signature.empty:
            return_type = sig.return_annotation

        return cls(
            name=func.__name__,
            description=doc,
            parameters=param_schema,
            return_type=return_type,
            is_async=is_async,
            source=BasicFunctionGuard(func),
        )

    @classmethod
    def from_pydantic(cls, model: Type[BaseModel]) -> "ToolParams":
        """Generate tool parameters from a Pydantic model."""
        schema = model.model_json_schema()

        tool_params = cls(
            name=model.__name__,
            description=model.__doc__ or "",
            parameters=schema,
            return_type=model,
            source=BasicFunctionGuard(model),
        )

        return tool_params

    def to_openai_schema(self, strict: bool = True) -> Dict[str, Any]:
        """Convert tool parameters to OpenAI function schema format.

        Args:
            strict: Whether to enforce strict parameter validation (default: True)

        Returns:
            Dict containing OpenAI-compatible function schema
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    **self.parameters,
                    # Ensure additionalProperties is set to False for strict mode
                    "additionalProperties": False if strict else True,
                },
                "strict": strict,
            },
        }
