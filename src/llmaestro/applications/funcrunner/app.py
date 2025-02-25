import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, get_type_hints

from llmaestro.config.agent import AgentTypeConfig
from llmaestro.llm.factory import LLMFactory
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.provider_state import ProviderStateManager
from llmaestro.llm.credential_manager import CredentialManager
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.loader import PromptLoader
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, str]
    return_type: str


class FunctionRegistry:
    def __init__(self):
        self._functions: Dict[str, Callable] = {}
        self._definitions: Dict[str, FunctionDefinition] = {}

    def register(self, func: Callable, description: str):
        """Register a function with its metadata"""
        hints = get_type_hints(func)
        params = {name: str(hints.get(name, Any).__name__) for name in inspect.signature(func).parameters}

        self._functions[func.__name__] = func
        self._definitions[func.__name__] = FunctionDefinition(
            name=func.__name__,
            description=description,
            parameters=params,
            return_type=str(hints.get("return", Any).__name__),
        )

    def get_function(self, name: str) -> Optional[Callable]:
        return self._functions.get(name)

    def get_definitions(self) -> List[FunctionDefinition]:
        return list(self._definitions.values())


class FunctionCallRequest(BaseModel):
    function_name: str
    arguments: Dict[str, Any]


class FunctionCallResponse(BaseModel):
    result: Any
    error: Optional[str] = None


class FunctionRunner:
    def __init__(
        self,
        api_key: Optional[str] = None,
        config_path: Optional[Path] = None,
        model_name: str = "claude-3-sonnet-20240229",
    ):
        self.registry = FunctionRegistry()
        self._init_config(config_path, api_key, model_name)
        self._init_llm()
        self._init_prompts()

    def _init_config(self, config_path: Optional[Path], api_key: Optional[str], model_name: str):
        """Initialize configuration from file or environment."""
        if config_path and config_path.exists():
            import yaml

            with open(config_path) as f:
                config_data = yaml.safe_load(f)
                api_key = api_key or config_data.get("llm", {}).get("api_key")

        self.config = AgentTypeConfig(
            provider="anthropic",  # Default to Anthropic for function calling
            model_name=model_name,
            api_key=api_key,
            max_tokens=1024,
            temperature=0.7,
        )

    def _init_llm(self):
        """Initialize LLM interface using factory pattern."""
        # Initialize managers
        self.llm_registry = LLMRegistry()
        self.provider_manager = ProviderStateManager()
        self.credential_manager = CredentialManager()
        
        # Initialize provider
        self.provider_manager.initialize_provider(
            family=self.config.provider,
            api_key=self.config.api_key
        )
        
        # Create LLM instance
        factory = LLMFactory(
            registry=self.llm_registry,
            provider_manager=self.provider_manager,
            credential_manager=self.credential_manager
        )
        self.llm = factory.create_llm(
            model_name=self.config.model,
            runtime_config=self.config.runtime
        )

    def _init_prompts(self):
        """Load prompt templates for function calling."""
        self.prompt_loader = PromptLoader({"file": BasePrompt})
        # Load function calling prompts - will be implemented in prompt.yaml
        self.prompts = {}

    def register_function(self, func: Callable, description: str):
        """Register a new function that can be called by the LLM"""
        self.registry.register(func, description)

    async def execute_function(self, request: FunctionCallRequest) -> FunctionCallResponse:
        """Execute a registered function with the provided arguments"""
        try:
            func = self.registry.get_function(request.function_name)
            if not func:
                raise ValueError(f"Function {request.function_name} not found")

            result = func(**request.arguments)
            return FunctionCallResponse(result=result)
        except Exception as e:
            logger.error(f"Function execution failed: {e}")
            return FunctionCallResponse(result=None, error=str(e))

    async def process_llm_request(self, user_input: str) -> Any:
        """Process a natural language request to call functions"""
        # Get available functions
        function_definitions = self.registry.get_definitions()

        # Load and render prompt
        if "function_calling" not in self.prompts:
            prompt = await self.prompt_loader.load_prompt("file", "src/applications/funcrunner/prompt.yaml")
            if prompt is None:
                raise ValueError("Failed to load function calling prompt")
            self.prompts["function_calling"] = prompt

        # Render prompt with function definitions and user input
        prompt = self.prompts["function_calling"]
        system_prompt, user_prompt = prompt.render(
            functions="\n".join(str(f) for f in function_definitions), user_input=user_input
        )

        # Format messages for LLM
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        # Process with LLM
        response = await self.llm.process(messages)

        try:
            # Parse response JSON
            function_call = response.json()

            # Validate response format
            if not isinstance(function_call, dict):
                raise ValueError("Invalid response format from LLM: not a dictionary")

            function_name = function_call.get("function_name")
            arguments = function_call.get("arguments")

            if not function_name or not isinstance(arguments, dict):
                raise ValueError("Invalid response format: missing function_name or arguments")

            # Execute function if confidence is high enough
            if function_call.get("confidence", 0) >= 0.7:
                request = FunctionCallRequest(function_name=function_name, arguments=arguments)
                result = await self.execute_function(request)
                return result
            else:
                return FunctionCallResponse(
                    result=None, error=f"Low confidence in function selection: {function_call.get('confidence', 0)}"
                )

        except Exception as e:
            logger.error(f"Failed to process LLM response: {e}")
            return FunctionCallResponse(result=None, error=str(e))
