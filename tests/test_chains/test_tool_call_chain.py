"""Tests for the ToolCallChain from tool_call_chain_example.py.

This test file exercises all conditionals and edges in the ToolCallChain,
including tool call handling, validation, and retry logic.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List
from uuid import uuid4
import json

from llmaestro.chains.tool_call_chain_example import ToolCallChain, create_tool_call_example
from llmaestro.core.orchestrator import Orchestrator
from llmaestro.agents.agent_pool import AgentPool
from llmaestro.prompts.base import BasePrompt
from llmaestro.core.models import LLMResponse
from llmaestro.llm.responses import ResponseFormat, ResponseFormatType, ValidationResult


@pytest.fixture
def orchestrator():
    """Mock orchestrator for testing."""
    return MagicMock(spec=Orchestrator)


@pytest.fixture
def agent_pool():
    """Mock agent pool for testing."""
    pool = MagicMock(spec=AgentPool)
    pool.execute_prompt = AsyncMock()
    return pool


@pytest.fixture
def initial_prompt():
    """Initial prompt that may trigger tool calls."""
    return BasePrompt(
        name="test_tool_call_prompt",
        description="Test prompt that may trigger tool calls",
        system_prompt="You are a helpful assistant that can use tools to answer questions.",
        user_prompt="What's the weather like in New York today?",
    )


@pytest.fixture
def expected_response_format():
    """Expected response format for validation."""
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["answer", "sources"],
    }
    return ResponseFormat.from_json_schema(
        schema=schema,
        format_type=ResponseFormatType.JSON_SCHEMA
    )


@pytest.fixture
def tool_call_response():
    """Mock LLM response with tool calls."""
    response = MagicMock(spec=LLMResponse)
    response.content = "I'll check the weather for you."
    response.metadata = {
        "tool_calls": [
            {
                "id": str(uuid4()),
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "New York"}
                }
            }
        ]
    }
    return response


@pytest.fixture
def no_tool_call_response():
    """Mock LLM response without tool calls."""
    response = MagicMock(spec=LLMResponse)
    response.content = "The weather in New York today is sunny with a high of 75°F."
    response.metadata = {"tool_calls": []}
    return response


@pytest.fixture
def valid_final_response():
    """Mock LLM response that passes validation."""
    response = MagicMock(spec=LLMResponse)
    response.content = """
    {
        "answer": "The weather in New York today is sunny with a high of 75°F.",
        "sources": ["weather.gov", "accuweather.com"]
    }
    """
    response.metadata = {"tool_calls": []}
    return response


@pytest.fixture
def invalid_final_response():
    """Mock LLM response that fails validation."""
    response = MagicMock(spec=LLMResponse)
    response.content = """
    {
        "answer": "The weather in New York today is sunny with a high of 75°F."
    }
    """
    response.metadata = {"tool_calls": []}
    return response


@pytest.fixture
def tool_execution_result():
    """Mock tool execution result."""
    return {
        "temperature": 75,
        "condition": "sunny",
        "humidity": 45,
        "wind_speed": 10,
        "source": "weather.gov"
    }


@pytest.mark.asyncio
async def test_tool_call_chain_with_tool_calls(
    orchestrator, agent_pool, initial_prompt, expected_response_format,
    tool_call_response, tool_execution_result, valid_final_response
):
    """Test ToolCallChain with a path that includes tool calls."""
    # Create the chain
    chain = ToolCallChain(orchestrator)
    chain.agent_pool = agent_pool

    # Configure the chain
    await chain.create_tool_call_chain(
        name="Tool Call Test",
        initial_prompt=initial_prompt,
        expected_response_format=expected_response_format,
    )

    # Mock the agent_pool.execute_prompt to return different responses in sequence
    # First call: return a response with tool calls
    # Second call: return a valid final response after tool execution
    agent_pool.execute_prompt.side_effect = [
        tool_call_response,  # Initial response with tool calls
        valid_final_response  # Response after tool execution
    ]

    # Mock the tool execution
    with patch('llmaestro.chains.chains.execute_tool_call', new_callable=AsyncMock) as mock_execute_tool:
        mock_execute_tool.return_value = tool_execution_result

        # Execute the chain
        results = await chain.execute_with_tool_handling()

        # Verify the results
        assert results is not None
        assert "results" in results
        assert "validation_status" in results
        assert "conversation_id" in results
        assert results["validation_status"] == "success"

        # Verify that the tool was called
        mock_execute_tool.assert_called_once()

        # Verify that the agent_pool was called twice
        assert agent_pool.execute_prompt.call_count == 2


@pytest.mark.asyncio
async def test_tool_call_chain_without_tool_calls(
    orchestrator, agent_pool, initial_prompt, expected_response_format,
    no_tool_call_response, valid_final_response
):
    """Test ToolCallChain with a path that doesn't include tool calls."""
    # Create the chain
    chain = ToolCallChain(orchestrator)
    chain.agent_pool = agent_pool

    # Configure the chain
    await chain.create_tool_call_chain(
        name="No Tool Call Test",
        initial_prompt=initial_prompt,
        expected_response_format=expected_response_format,
    )

    # Mock the agent_pool.execute_prompt to return a response without tool calls
    agent_pool.execute_prompt.return_value = valid_final_response

    # Execute the chain
    results = await chain.execute_with_tool_handling()

    # Verify the results
    assert results is not None
    assert "results" in results
    assert "validation_status" in results
    assert "conversation_id" in results
    assert results["validation_status"] == "success"

    # Verify that the agent_pool was called once
    assert agent_pool.execute_prompt.call_count == 1


@pytest.mark.asyncio
async def test_tool_call_chain_with_validation_failure(
    orchestrator, agent_pool, initial_prompt, expected_response_format,
    no_tool_call_response, invalid_final_response, valid_final_response
):
    """Test ToolCallChain with validation failures and retries."""
    # Create the chain
    chain = ToolCallChain(orchestrator, max_validation_attempts=2)
    chain.agent_pool = agent_pool

    # Configure the chain
    await chain.create_tool_call_chain(
        name="Validation Failure Test",
        initial_prompt=initial_prompt,
        expected_response_format=expected_response_format,
    )

    # Mock the validation node's validate method to fail on the first call and succeed on the second
    validation_mock = MagicMock()

    # Create ValidationResult objects with the correct parameters
    invalid_result = ValidationResult(
        is_valid=False,
        original_response=invalid_final_response.content,
        errors=["Missing required field 'sources'"]
    )

    valid_result = ValidationResult(
        is_valid=True,
        original_response=valid_final_response.content
    )

    validation_mock.validate.side_effect = [invalid_result, valid_result]

    # Find the validation node and replace its validate method
    validation_nodes = [
        node for node_id, node in chain.nodes.items()
        if hasattr(node, 'response_format')
    ]
    assert len(validation_nodes) == 1
    validation_node = validation_nodes[0]
    original_validate = validation_node.validate
    validation_node.validate = validation_mock.validate

    # Mock the agent_pool.execute_prompt to return an invalid response first, then a valid one
    agent_pool.execute_prompt.side_effect = [
        invalid_final_response,  # First response (fails validation)
        valid_final_response     # Second response (passes validation)
    ]

    # Execute the chain
    results = await chain.execute_with_tool_handling()

    # Verify the results
    assert results is not None
    assert "results" in results
    assert "validation_status" in results
    assert "conversation_id" in results
    assert results["validation_status"] == "success"

    # Verify that validation was called twice
    assert validation_mock.validate.call_count == 2

    # Verify that the agent_pool was called twice
    assert agent_pool.execute_prompt.call_count == 2

    # Restore the original validate method
    validation_node.validate = original_validate


@pytest.mark.asyncio
async def test_tool_call_chain_with_tool_call_failures(
    orchestrator, agent_pool, initial_prompt, expected_response_format,
    tool_call_response, tool_execution_result, valid_final_response
):
    """Test ToolCallChain with tool call failures and retries."""
    # Create the chain
    chain = ToolCallChain(orchestrator, max_tool_call_attempts=2)
    chain.agent_pool = agent_pool

    # Configure the chain
    await chain.create_tool_call_chain(
        name="Tool Call Failure Test",
        initial_prompt=initial_prompt,
        expected_response_format=expected_response_format,
    )

    # Mock the agent_pool.execute_prompt to return different responses in sequence
    agent_pool.execute_prompt.side_effect = [
        tool_call_response,    # Initial response with tool calls
        tool_call_response,    # Another tool call after first tool execution
        valid_final_response   # Final valid response
    ]

    # Mock the tool execution to fail on first attempt and succeed on second
    with patch('llmaestro.chains.chains.execute_tool_call', new_callable=AsyncMock) as mock_execute_tool:
        # First call raises an exception, second call succeeds
        mock_execute_tool.side_effect = [
            Exception("Tool execution failed"),
            tool_execution_result
        ]

        # Execute the chain
        results = await chain.execute_with_tool_handling()

        # Verify the results
        assert results is not None
        assert "results" in results
        assert "validation_status" in results
        assert "conversation_id" in results
        assert results["validation_status"] == "success"

        # Verify that the tool was called twice
        assert mock_execute_tool.call_count == 2

        # Verify that the agent_pool was called three times
        assert agent_pool.execute_prompt.call_count == 3


@pytest.mark.asyncio
async def test_create_tool_call_example_function(
    orchestrator, agent_pool, initial_prompt, expected_response_format,
    tool_call_response, tool_execution_result, valid_final_response
):
    """Test the create_tool_call_example helper function."""
    # Mock the agent_pool.execute_prompt to return different responses in sequence
    agent_pool.execute_prompt.side_effect = [
        tool_call_response,  # Initial response with tool calls
        valid_final_response  # Response after tool execution
    ]

    # Mock the tool execution
    with patch('llmaestro.chains.chains.execute_tool_call', new_callable=AsyncMock) as mock_execute_tool:
        mock_execute_tool.return_value = tool_execution_result

        # Execute the helper function
        results = await create_tool_call_example(
            orchestrator=orchestrator,
            agent_pool=agent_pool,
            initial_prompt=initial_prompt,
            expected_response_format=expected_response_format,
        )

        # Verify the results
        assert results is not None
        assert "results" in results
        assert "validation_status" in results
        assert "conversation_id" in results
        assert results["validation_status"] == "success"

        # Verify that the tool was called
        mock_execute_tool.assert_called_once()

        # Verify that the agent_pool was called twice
        assert agent_pool.execute_prompt.call_count == 2


@pytest.mark.asyncio
async def test_tool_call_chain_max_attempts_exceeded(
    orchestrator, agent_pool, initial_prompt, expected_response_format,
    tool_call_response, invalid_final_response
):
    """Test ToolCallChain when max validation attempts are exceeded."""
    # Create the chain with only 1 validation attempt
    chain = ToolCallChain(orchestrator, max_validation_attempts=1)
    chain.agent_pool = agent_pool

    # Configure the chain
    await chain.create_tool_call_chain(
        name="Max Attempts Test",
        initial_prompt=initial_prompt,
        expected_response_format=expected_response_format,
    )

    # Mock the agent_pool.execute_prompt to return an invalid response
    agent_pool.execute_prompt.return_value = invalid_final_response

    # Mock the validation node's validate method to always fail
    validation_mock = MagicMock()

    # Create a ValidationResult with the correct parameters
    invalid_result = ValidationResult(
        is_valid=False,
        original_response=invalid_final_response.content,
        errors=["Missing required field 'sources'"]
    )

    validation_mock.validate.return_value = invalid_result

    # Find the validation node and replace its validate method
    validation_nodes = [
        node for node_id, node in chain.nodes.items()
        if hasattr(node, 'response_format')
    ]
    assert len(validation_nodes) == 1
    validation_node = validation_nodes[0]
    original_validate = validation_node.validate
    validation_node.validate = validation_mock.validate

    # Execute the chain
    results = await chain.execute_with_tool_handling()

    # Verify the results
    assert results is not None
    assert "results" in results
    assert "validation_status" in results
    assert "conversation_id" in results
    assert results["validation_status"] == "failed"

    # Verify that validation was called once
    assert validation_mock.validate.call_count == 1

    # Verify that the agent_pool was called once
    assert agent_pool.execute_prompt.call_count == 1

    # Restore the original validate method
    validation_node.validate = original_validate
