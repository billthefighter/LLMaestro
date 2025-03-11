"""Tests for LLM integration with tools."""
import json
import pytest
import logging
from typing import List, Optional, cast, Union

from pydantic import BaseModel, Field

from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.core.models import LLMResponse
from llmaestro.prompts.base import BasePrompt, PromptVariable, SerializableType
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.tools.core import ToolParams, BasicFunctionGuard
from llmaestro.llm.responses import ResponseFormat, ResponseFormatType
from llmaestro.default_library.defined_providers.openai.interface import OpenAIInterface
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam

logger = logging.getLogger(__name__)


@pytest.fixture
def get_weather():
    """Get current temperature for a given location."""
    def _get_weather(location: str) -> str:
        return f"The weather in {location} is always sunny."
    return _get_weather


@pytest.fixture
async def openai_prompt(get_weather) -> BasePrompt:
    """Create a prompt for testing OpenAI tool usage."""
    return MemoryPrompt(
        name="weather_query",
        description="Query weather information using tools",
        system_prompt="You are a weather assistant.",
        user_prompt="What is the weather like in {location} today?",
        variables=[
            PromptVariable(name="location", expected_input_type=SerializableType.STRING)
        ],
        tools=[
            ToolParams(
                name="get_weather",
                description="Get current temperature for a given location.",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia"
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": False
                },
                is_async=False,
                source=BasicFunctionGuard(get_weather)
            )
        ]
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_openai_tool_integration(test_settings, llm_registry: LLMRegistry, openai_prompt: BasePrompt):
    """Test OpenAI interface with tool integration."""
    if not test_settings.use_real_tokens:
        pytest.skip("Skipping test that requires LLM API tokens")

    # Arrange
    logger.info("Testing OpenAI tool integration")
    logger.info(f"llm_registry.get_registered_models(): {llm_registry.get_registered_models()}")
    required_capabilities = {"supports_function_calling", "supports_tools"}
    model_name = llm_registry.find_cheapest_model_with_capabilities(required_capabilities)
    llm_instance = await llm_registry.create_instance(model_name)

    # Act - first response should contain tool calls
    logger.info(f"openai_prompt: {openai_prompt}")
    response = await llm_instance.interface.process(openai_prompt, variables={"location": "Paris"})

    # Assert first response
    assert isinstance(response, LLMResponse)
    assert response.success is True

    # The response should be a JSON string containing the tool call arguments
    tool_args = json.loads(response.content)
    assert "location" in tool_args
    assert tool_args["location"] in ["Paris", "Paris, France"]

    # Execute the tool directly
    tool = openai_prompt.tools[0]  # get_weather tool
    result = await tool.execute(**tool_args)

    # Verify tool execution result
    assert "The weather in Paris" in result
    assert "sunny" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_call_response_handling(test_settings, llm_registry: LLMRegistry):
    """Test handling of tool call responses."""
    if not test_settings.use_real_tokens:
        pytest.skip("Skipping test that requires LLM API tokens")

    # Create a test tool
    def calculator(a: int, b: int, operation: str = "add") -> int:
        """Simple calculator function."""
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        raise ValueError(f"Unsupported operation: {operation}")

    # Create a prompt with the calculator tool
    prompt = MemoryPrompt(
        name="calculator",
        description="Test calculator tool",
        system_prompt="You are a calculator assistant.",
        user_prompt="What is {a} {operation} {b}?",
        variables=[
            PromptVariable(name="a", expected_input_type=SerializableType.INTEGER),
            PromptVariable(name="b", expected_input_type=SerializableType.INTEGER),
            PromptVariable(name="operation", expected_input_type=SerializableType.STRING),
        ],
        tools=[
            ToolParams(
                name="calculator",
                description="Perform basic arithmetic operations.",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                        "operation": {
                            "type": "string",
                            "enum": ["add", "multiply"]
                        }
                    },
                    "required": ["a", "b", "operation"],
                    "additionalProperties": False
                },
                is_async=False,
                source=BasicFunctionGuard(calculator)
            )
        ]
    )

    # Get LLM instance
    required_capabilities = {"supports_function_calling", "supports_tools"}
    model_name = llm_registry.find_cheapest_model_with_capabilities(required_capabilities)
    llm_instance = await llm_registry.create_instance(model_name)

    # Test addition
    response = await llm_instance.interface.process(
        prompt,
        variables={"a": 5, "b": 3, "operation": "add"}
    )

    assert isinstance(response, LLMResponse)
    assert response.success is True

    # Parse tool call arguments
    tool_args = json.loads(response.content)
    assert tool_args["a"] == 5
    assert tool_args["b"] == 3
    assert tool_args["operation"] == "add"

    # Execute tool
    tool = prompt.tools[0]
    result = await tool.execute(**tool_args)
    assert result == 8

    # Test multiplication
    response = await llm_instance.interface.process(
        prompt,
        variables={"a": 4, "b": 6, "operation": "multiply"}
    )

    assert isinstance(response, LLMResponse)
    assert response.success is True

    # Parse tool call arguments
    tool_args = json.loads(response.content)
    assert tool_args["a"] == 4
    assert tool_args["b"] == 6
    assert tool_args["operation"] == "multiply"

    # Execute tool
    result = await tool.execute(**tool_args)
    assert result == 24


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_token_parameter_failover(test_settings, llm_registry: LLMRegistry):
    """Test that OpenAI interface correctly handles max_tokens vs max_completion_tokens errors
    and respects model capabilities for parameter filtering."""
    if not test_settings.use_real_tokens:
        pytest.skip("Skipping test that requires LLM API tokens")

    # Create a prompt with parameters that might not be supported
    prompt = MemoryPrompt(
        name="token_test",
        description="Test token parameter handling",
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'hello' in exactly 5 words.",
        variables=[],
        expected_response=ResponseFormat(
            format=ResponseFormatType.TEXT,
            response_schema=None,
            retry_config=None,
        ),
    )

    # Get LLM instance for a model that requires max_completion_tokens
    required_capabilities = {"supports_json_mode", "supports_tools"}
    model_name = llm_registry.find_cheapest_model_with_capabilities(required_capabilities)
    llm_instance = await llm_registry.create_instance(model_name)
    interface = cast(OpenAIInterface, llm_instance.interface)

    # Modify the model's capabilities to test parameter filtering
    if interface.state and interface.state.profile.capabilities:
        interface.state.profile.capabilities.supports_frequency_penalty = False
        interface.state.profile.capabilities.supports_presence_penalty = False

    # Process the prompt with parameters that should be filtered
    response = await interface.process(
        prompt,
        variables={},
    )

    # Assert the response was successful
    assert isinstance(response, LLMResponse)
    assert response.success is True
    assert response.content is not None and len(response.content) > 0

    # Verify the model was added to the list of models requiring max_completion_tokens
    assert model_name in interface.models_requiring_completion_tokens

    # Try to create a chat completion directly with unsupported parameters
    # This should be filtered by _check_model_capabilities
    messages = [ChatCompletionUserMessageParam(role="user", content="Hello")]
    test_messages = cast(
        List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]],
        messages
    )
    completion = await interface._create_chat_completion(
        messages=test_messages,
        model=model_name,
        temperature=0.7,
        max_tokens=10,
        frequency_penalty=0.5,  # This should be filtered out
        presence_penalty=0.5,   # This should be filtered out
    )

    assert completion is not None
    assert completion.choices[0].message is not None

    # Process another prompt to verify it uses max_completion_tokens directly
    second_response = await interface.process(prompt)
    assert isinstance(second_response, LLMResponse)
    assert second_response.success is True
    assert second_response.content is not None and len(second_response.content) > 0
