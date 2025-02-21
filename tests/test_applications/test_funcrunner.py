"""Tests for the FunctionRunner application."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llmaestro.applications.funcrunner.app import (
    FunctionRunner,
    FunctionRegistry,
    FunctionDefinition,
    FunctionCallRequest,
    FunctionCallResponse
)
from llmaestro.llm.interfaces.base import LLMResponse
from llmaestro.prompts.base import BasePrompt


class MockPrompt:
    """Mock prompt for testing."""
    def __init__(self):
        self.name = "function_calling"
        self.system_prompt = "You are a function calling assistant"
        self.user_prompt = "User request: {{user_input}}\nAvailable functions:\n{{functions}}"

    def render(self, **kwargs) -> Tuple[str, str]:
        return (
            self.system_prompt,
            self.user_prompt.replace("{{user_input}}", kwargs.get("user_input", ""))
                           .replace("{{functions}}", kwargs.get("functions", ""))
        )


# Test Functions
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def greet(name: str, greeting: str = "Hello") -> str:
    """Greet someone with a custom greeting."""
    return f"{greeting}, {name}!"

def process_data(data: Dict[str, Any], filter_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Process a dictionary of data, optionally filtering keys."""
    if filter_keys:
        return {k: v for k, v in data.items() if k in filter_keys}
    return data


# Mock Data
MOCK_ADD_RESPONSE = {
    "function_name": "add",
    "arguments": {"a": 5, "b": 3},
    "confidence": 0.95
}

MOCK_GREET_RESPONSE = {
    "function_name": "greet",
    "arguments": {"name": "World", "greeting": "Hi"},
    "confidence": 0.9
}


# Mock Classes
class MockLLMResponse:
    """Mock LLM response for testing."""
    def __init__(self, content: str):
        self.content = content

    def json(self) -> Dict[str, Any]:
        return json.loads(self.content)


# Fixtures
@pytest.fixture
def mock_prompt():
    """Mock prompt for testing."""
    return MockPrompt()

@pytest.fixture
def mock_prompt_loader(mock_prompt):
    """Mock prompt loader for testing."""
    mock = AsyncMock()
    mock.load_prompt.return_value = mock_prompt
    return mock

@pytest.fixture
def mock_llm_registry():
    """Mock model registry for testing."""
    mock = MagicMock()
    mock.get_model.return_value = MagicMock(
        capabilities=MagicMock(
            supports_function_calling=True,
            max_context_window=100000
        )
    )
    return mock

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = AsyncMock()
    mock.process.return_value = MockLLMResponse(json.dumps(MOCK_ADD_RESPONSE))
    return mock

@pytest.fixture
def test_config_path(tmp_path):
    """Create a test config file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
llm:
  provider: anthropic
  model: claude-3-sonnet-20240229
  api_key: test-key
    """)
    return config_path

@pytest.fixture
def function_runner(mock_llm, mock_prompt_loader, mock_llm_registry):
    """Create a FunctionRunner instance for testing."""
    with patch("src.llmaestro.applications.funcrunner.app.create_llm_interface") as mock_create_llm, \
         patch("src.llmaestro.applications.funcrunner.app.PromptLoader") as MockPromptLoader, \
         patch("src.llmaestro.applications.funcrunner.app.LLMRegistry") as MockLLMRegistry:

        mock_create_llm.return_value = mock_llm
        MockPromptLoader.return_value = mock_prompt_loader
        MockLLMRegistry.from_yaml.return_value = mock_llm_registry

        runner = FunctionRunner(api_key="test-key")
        # Register test functions
        runner.register_function(add, "Add two numbers together")
        runner.register_function(greet, "Greet someone with a custom greeting")
        return runner


# Unit Tests
def test_function_runner_init(function_runner):
    """Test FunctionRunner initialization."""
    assert isinstance(function_runner.registry, FunctionRegistry)
    assert function_runner.llm is not None
    assert function_runner.config.provider == "anthropic"
    assert function_runner.config.model_name == "claude-3-sonnet-20240229"


def test_function_registry():
    """Test FunctionRegistry functionality."""
    registry = FunctionRegistry()

    # Test function registration
    registry.register(add, "Add two numbers")

    # Test function retrieval
    func = registry.get_function("add")
    assert func is add

    # Test function definitions
    definitions = registry.get_definitions()
    assert len(definitions) == 1
    assert isinstance(definitions[0], FunctionDefinition)
    assert definitions[0].name == "add"
    assert definitions[0].parameters == {"a": "int", "b": "int"}
    assert definitions[0].return_type == "int"


def test_function_registration(function_runner):
    """Test function registration in FunctionRunner."""
    definitions = function_runner.registry.get_definitions()
    assert len(definitions) == 2

    # Check add function
    add_def = next(d for d in definitions if d.name == "add")
    assert add_def.parameters == {"a": "int", "b": "int"}
    assert add_def.return_type == "int"

    # Check greet function
    greet_def = next(d for d in definitions if d.name == "greet")
    assert greet_def.parameters == {"name": "str", "greeting": "str"}
    assert greet_def.return_type == "str"


@pytest.mark.asyncio
async def test_execute_function(function_runner):
    """Test function execution."""
    # Test successful execution
    request = FunctionCallRequest(
        function_name="add",
        arguments={"a": 5, "b": 3}
    )
    response = await function_runner.execute_function(request)
    assert isinstance(response, FunctionCallResponse)
    assert response.result == 8
    assert response.error is None

    # Test function not found
    request = FunctionCallRequest(
        function_name="nonexistent",
        arguments={}
    )
    response = await function_runner.execute_function(request)
    assert response.result is None
    assert "not found" in response.error

    # Test invalid arguments
    request = FunctionCallRequest(
        function_name="add",
        arguments={"a": "not_a_number", "b": 3}
    )
    response = await function_runner.execute_function(request)
    assert response.result is None
    assert response.error is not None


@pytest.mark.asyncio
async def test_process_llm_request_success(function_runner, mock_llm):
    """Test successful LLM request processing."""
    mock_llm.process.return_value = MockLLMResponse(json.dumps(MOCK_ADD_RESPONSE))

    result = await function_runner.process_llm_request("Add 5 and 3")
    assert isinstance(result, FunctionCallResponse)
    assert result.result == 8
    assert result.error is None

    # Verify LLM was called correctly
    mock_llm.process.assert_called_once()
    messages = mock_llm.process.call_args[0][0]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_process_llm_request_low_confidence(function_runner, mock_llm):
    """Test LLM request with low confidence."""
    mock_response = MOCK_ADD_RESPONSE.copy()
    mock_response["confidence"] = 0.5
    mock_llm.process.return_value = MockLLMResponse(json.dumps(mock_response))

    result = await function_runner.process_llm_request("Add some numbers maybe?")
    assert isinstance(result, FunctionCallResponse)
    assert result.result is None
    assert result.error is not None
    assert "Low confidence" in str(result.error)


@pytest.mark.asyncio
async def test_process_llm_request_invalid_response(function_runner, mock_llm):
    """Test handling of invalid LLM responses."""
    # Test invalid JSON
    mock_llm.process.return_value = MockLLMResponse("not json")
    result = await function_runner.process_llm_request("Add 5 and 3")
    assert result.error is not None

    # Test missing required fields
    mock_llm.process.return_value = MockLLMResponse(json.dumps({}))
    result = await function_runner.process_llm_request("Add 5 and 3")
    assert result.error is not None


# Integration Tests
@pytest.mark.integration
@pytest.mark.asyncio
async def test_function_runner_integration(test_config_path, mock_llm_registry, mock_llm, mock_prompt_loader):
    """Integration test with config file and multiple functions."""
    # Initialize runner with real config
    with patch("llmaestro.applications.funcrunner.app.LLMRegistry") as MockLLMRegistry, \
         patch("llmaestro.applications.funcrunner.app.create_llm_interface") as mock_create_llm, \
         patch("llmaestro.applications.funcrunner.app.PromptLoader") as MockPromptLoader:

        MockLLMRegistry.from_yaml.return_value = mock_llm_registry
        mock_create_llm.return_value = mock_llm
        MockPromptLoader.return_value = mock_prompt_loader

        runner = FunctionRunner(config_path=test_config_path)

        # Register test functions
        runner.register_function(add, "Add two numbers together")
        runner.register_function(greet, "Greet someone with a custom greeting")
        runner.register_function(process_data, "Process a dictionary of data")

        # Test add function
        mock_llm.process.return_value = MockLLMResponse(json.dumps(MOCK_ADD_RESPONSE))
        result = await runner.process_llm_request("Can you add 5 and 3?")
        assert isinstance(result, FunctionCallResponse)
        assert result.result == 8

        # Test greet function
        mock_llm.process.return_value = MockLLMResponse(json.dumps(MOCK_GREET_RESPONSE))
        result = await runner.process_llm_request("Say hi to World")
        assert isinstance(result, FunctionCallResponse)
        assert result.result == "Hi, World!"


@pytest.mark.asyncio
async def test_function_runner_with_real_config(test_config_path, mock_llm_registry, mock_llm):
    """Test FunctionRunner with a real config file."""
    with patch("llmaestro.applications.funcrunner.app.LLMRegistry") as MockLLMRegistry, \
         patch("llmaestro.applications.funcrunner.app.create_llm_interface") as mock_create_llm:

        MockLLMRegistry.from_yaml.return_value = mock_llm_registry
        mock_create_llm.return_value = mock_llm

        runner = FunctionRunner(config_path=test_config_path)
        assert runner.config.provider == "anthropic"
        assert runner.config.model_name == "claude-3-sonnet-20240229"
        assert runner.config.api_key == "test-key"
