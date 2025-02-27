"""Basic hello world tests for LLM functionality."""
import pytest
from typing import Dict, Any

from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.core.models import LLMResponse
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.prompts.types import PromptMetadata
from llmaestro.llm.responses import ResponseFormat, ResponseFormatType


@pytest.fixture
async def hello_world_prompt() -> MemoryPrompt:
    """Create a simple hello world prompt for testing."""
    return MemoryPrompt(
        name="hello_world",
        description="Simple hello world test prompt",
        system_prompt="You are a helpful AI assistant.",
        user_prompt="Say hello!",
        metadata=PromptMetadata(
            type="test",
            expected_response=ResponseFormat(format=ResponseFormatType.TEXT, schema=None),
            tags=["test"],
            is_active=True
        )
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_basic_llm_response(test_settings, llm_registry: LLMRegistry, hello_world_prompt: MemoryPrompt):
    """Test that we can get a basic response from the LLM."""
    if not test_settings.use_real_tokens:
        pytest.skip("Skipping test that requires LLM API tokens")

    # Arrange
    model_name = llm_registry.get_registered_models()[0]
    llm_instance = await llm_registry.create_instance(model_name)

    # Act
    response = await llm_instance.interface.process(hello_world_prompt)

    # Assert
    assert isinstance(response, LLMResponse)
    assert response.success is True
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_llm_streaming_response(test_settings, llm_registry: LLMRegistry, hello_world_prompt: MemoryPrompt):
    """Test that we can stream responses from the LLM."""
    if not test_settings.use_real_tokens:
        pytest.skip("Skipping test that requires LLM API tokens")

    # Arrange
    model_name = llm_registry.get_registered_models()[0]
    llm_instance = await llm_registry.create_instance(model_name)
    collected_content = []

    # Act
    async for chunk in llm_instance.interface.stream(hello_world_prompt):
        assert isinstance(chunk, LLMResponse)
        assert chunk.success is True
        collected_content.append(chunk.content)

    # Assert
    assert len(collected_content) > 0
    full_response = "".join(collected_content)
    assert isinstance(full_response, str)
    assert len(full_response) > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_llm_direct_string_prompt(test_settings, llm_registry: LLMRegistry):
    """Test that we can send a direct string prompt to the LLM."""
    if not test_settings.use_real_tokens:
        pytest.skip("Skipping test that requires LLM API tokens")

    # Arrange
    model_name = llm_registry.get_registered_models()[0]
    llm_instance = await llm_registry.create_instance(model_name)
    test_prompt = "What is 2+2?"

    # Act
    response = await llm_instance.interface.process(test_prompt)

    # Assert
    assert isinstance(response, LLMResponse)
    assert response.success is True
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert "4" in response.content.lower()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_llm_with_variables(test_settings, llm_registry: LLMRegistry):
    """Test that we can use variables in prompts."""
    if not test_settings.use_real_tokens:
        pytest.skip("Skipping test that requires LLM API tokens")

    # Arrange
    model_name = llm_registry.get_registered_models()[0]
    llm_instance = await llm_registry.create_instance(model_name)

    prompt = MemoryPrompt(
        name="greeting",
        description="Test prompt with variables",
        system_prompt="You are a friendly assistant.",
        user_prompt="Say hello to {name}!",
        metadata=PromptMetadata(
            type="test",
            expected_response=ResponseFormat(format=ResponseFormatType.TEXT, schema=None),
            tags=["test"],
            is_active=True
        )
    )

    variables = {"name": "Alice"}

    # Act
    response = await llm_instance.interface.process(prompt, variables)

    # Assert
    assert isinstance(response, LLMResponse)
    assert response.success is True
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert "alice" in response.content.lower()
