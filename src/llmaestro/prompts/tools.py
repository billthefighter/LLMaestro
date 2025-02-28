from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional, Callable, Type, Union
import inspect


class ToolParams(BaseModel):
    """Parameters for a tool/function that can be used by an LLM."""

    name: str = Field(description="The function's name")
    description: str = Field(description="Details on when and how to use the function")
    parameters: Dict[str, Any] = Field(description="JSON schema defining the function's input arguments")
    return_type: Optional[Any] = Field(default=None, description="The return type of the function if available")
    is_async: bool = Field(default=False, description="Whether the function is async")
    source: Union[Callable, Type[BaseModel]] = Field(description="The source function or model that this tool wraps")

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
            if isinstance(self.source, type) and issubclass(self.source, BaseModel):
                # For Pydantic models, instantiate with kwargs
                result = self.source(**kwargs)
            elif callable(self.source):
                # For functions, execute with kwargs
                if self.is_async:
                    result = await self.source(**kwargs)
                else:
                    result = self.source(**kwargs)
            else:
                raise TypeError(f"Invalid source type for tool {self.name}: {type(self.source)}")

            return result

        except Exception as e:
            raise TypeError(f"Failed to execute {self.name}: {str(e)}")

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
            source=func,
        )

    @classmethod
    def from_pydantic(cls, model: Type[BaseModel]) -> "ToolParams":
        """Generate tool parameters from a Pydantic model."""
        schema = model.model_json_schema()

        tool_params = cls(
            name=model.__name__, description=model.__doc__ or "", parameters=schema, return_type=model, source=model
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
