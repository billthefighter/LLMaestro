"""Test structured output functionality with LLMs."""

import json
import pytest
from typing import List, Optional
from pydantic import BaseModel
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.prompts.types import PromptMetadata
from llmaestro.llm.responses import ResponseFormatType
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.core.models import LLMResponse
from llmaestro.llm.responses import ResponseFormat
from unittest.mock import AsyncMock, MagicMock
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from openai import AsyncOpenAI
from llmaestro.default_library.defined_providers.openai.interface import OpenAIInterface

class Person(BaseModel):
    """Sample person model for testing structured output."""
    name: str
    age: int
    hobbies: List[str]
    occupation: Optional[str] = None


class NestedStructure(BaseModel):
    """Sample nested structure for testing complex schemas."""
    title: str
    people: List[Person]
    team_size: int
    department: Optional[str] = None


@pytest.fixture
def person_prompt() -> MemoryPrompt:
    """Create a prompt for testing simple structured output."""
    return MemoryPrompt(
        name="person_extractor",
        description="Extract person information",
        system_prompt="""You are an expert at extracting structured information about people.
Extract the requested information accurately and return it in the specified JSON format.
Make sure to include all required fields and validate against the provided schema.""",
        user_prompt="Please create a profile for a fictional person with their name, age, and hobbies.",
        expected_response=ResponseFormat.from_pydantic_model(
            model=Person,
            convert_to_json_schema=True,
            format_type=ResponseFormatType.JSON_SCHEMA
        ),
        metadata=PromptMetadata(
            type="person_extraction",
            tags=["structured", "person", "test"]
        )
    )


@pytest.fixture
def nested_prompt() -> MemoryPrompt:
    """Create a prompt for testing nested structured output."""
    return MemoryPrompt(
        name="team_extractor",
        description="Extract team information with nested person objects",
        system_prompt="""You are an expert at extracting structured information about teams and their members.
Extract the requested information accurately and return it in the specified JSON format.
Make sure to include all required fields and validate against the provided schema.

Important requirements:
1. The team_size field must exactly match the number of people in the people list
2. The people list must contain at least 2 team members
3. Each person must have a name, age, and at least one hobby""",
        user_prompt="Please create a profile for a fictional team with a title, list of team members (at least 2), and team size. Remember that the team_size must match the exact number of people in the list.",
        metadata=PromptMetadata(
            type="team_extraction",
            tags=["structured", "team", "test"]
        ),
        expected_response=ResponseFormat.from_pydantic_model(
            model=NestedStructure,
            convert_to_json_schema=True,
            format_type=ResponseFormatType.JSON_SCHEMA
        ),
    )


@pytest.fixture
def schema():
    """Test schema fixture."""
    return {
        "type": "object",
        "properties": {
            "color": {"type": "string", "enum": ["red", "blue", "green"]},
            "count": {"type": "integer", "minimum": 1},
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["color", "count", "tags"]
    }

@pytest.fixture
def test_prompt(schema):
    """Test prompt fixture."""
    return MemoryPrompt(
        name="schema_test",
        description="Test direct schema validation",
        system_prompt="You are a helpful assistant that provides structured data.",
        user_prompt="Please provide a color (red, blue, or green), a count (positive integer), and some tags.",
        metadata=PromptMetadata(
            type="schema_test",
            tags=["structured", "schema", "test"]
        ),
        expected_response=ResponseFormat(
            format=ResponseFormatType.JSON_SCHEMA,
            response_schema=json.dumps(schema),
            convert_to_json_schema=False
        )
    )

@pytest.fixture
def mock_chat_completion():
    """Mock ChatCompletion fixture."""
    #TODO: Implement mock chat completion
    pass

@pytest.mark.asyncio
@pytest.mark.integration
async def test_simple_structured_output(test_settings, llm_registry: LLMRegistry, person_prompt: MemoryPrompt):
    """Test simple structured output with a Person model."""
    if not test_settings.use_real_tokens:
        pytest.skip("Skipping test that requires LLM API tokens")

    # Arrange
    model_name = "gpt-4o-mini-2024-07-18"  # Specify GPT-4 mini model
    llm_instance = await llm_registry.create_instance(model_name)

    # Act
    response = await llm_instance.interface.process(person_prompt)

    # Assert
    assert isinstance(response, LLMResponse)
    assert response.success is True
    assert isinstance(response.content, str)

    # Debug print the exact response
    print("\nPerson Response Content:")
    print(response.content)
    print("\n")

    # Parse the response into a Person model
    person_data = Person.model_validate_json(response.content)

    # Validate the structured data
    assert isinstance(person_data.name, str)
    assert isinstance(person_data.age, int)
    assert isinstance(person_data.hobbies, list)
    assert len(person_data.hobbies) > 0
    assert all(isinstance(hobby, str) for hobby in person_data.hobbies)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_nested_structured_output(test_settings, llm_registry: LLMRegistry, nested_prompt: MemoryPrompt):
    """Test nested structured output with a NestedStructure model."""
    if not test_settings.use_real_tokens:
        pytest.skip("Skipping test that requires LLM API tokens")

    # Arrange
    model_name = "gpt-4o-mini-2024-07-18"  # Specify GPT-4 mini model
    llm_instance = await llm_registry.create_instance(model_name)

    # Act
    response = await llm_instance.interface.process(nested_prompt)

    # Assert
    assert isinstance(response, LLMResponse)
    assert response.success is True
    assert isinstance(response.content, str)

    # Debug print the exact response
    print("\nNested Structure Response Content:")
    print(response.content)
    print("\n")

    # Parse the response into a NestedStructure model
    team_data = NestedStructure.model_validate_json(response.content)

    # Validate the structured data
    assert isinstance(team_data.title, str)
    assert isinstance(team_data.people, list)
    assert len(team_data.people) >= 2
    assert isinstance(team_data.team_size, int)
    assert team_data.team_size == len(team_data.people)

    # Validate nested Person objects
    for person in team_data.people:
        assert isinstance(person.name, str)
        assert isinstance(person.age, int)
        assert isinstance(person.hobbies, list)
        assert len(person.hobbies) > 0
        assert all(isinstance(hobby, str) for hobby in person.hobbies)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_direct_schema_output_mocked(test_settings, llm_registry: LLMRegistry, test_prompt, mock_chat_completion):
    """Test structured output with direct JSON schema using mocked responses."""
    # Arrange
    model_name = "gpt-4o-mini-2024-07-18"  # Specify GPT-4 mini model
    llm_instance = await llm_registry.create_instance(model_name)

    # Mock the chat completion call
    interface = llm_instance.interface
    if isinstance(interface, OpenAIInterface):
        interface.client.chat.completions.create = AsyncMock(return_value=mock_chat_completion)

    # Act
    response = await llm_instance.interface.process(test_prompt)

    # Assert
    assert isinstance(response, LLMResponse)
    assert response.success is True
    assert isinstance(response.content, str)

    # Parse and validate the response
    parsed_data = json.loads(response.content)

    # Debug print the exact response
    print("\nMocked Response Content:")
    print(response.content)
    print("\n")

    # Validate the structured data
    assert isinstance(parsed_data, dict)
    assert "color" in parsed_data
    assert parsed_data["color"] in ["red", "blue", "green"]
    assert "count" in parsed_data
    assert isinstance(parsed_data["count"], int)
    assert parsed_data["count"] >= 1
    assert "tags" in parsed_data
    assert isinstance(parsed_data["tags"], list)
    assert all(isinstance(tag, str) for tag in parsed_data["tags"])

@pytest.mark.real_tokens
async def test_direct_schema_output_real(test_settings, llm_registry: LLMRegistry, test_prompt):
    """Test structured output with direct JSON schema using real API tokens."""
    if not test_settings.use_real_tokens:
        pytest.skip("Skipping test that requires LLM API tokens")

    # Arrange
    model_name = "gpt-4o-mini-2024-07-18"  # Specify GPT-4 mini model
    llm_instance = await llm_registry.create_instance(model_name)

    # Act
    response = await llm_instance.interface.process(test_prompt)

    # Assert
    assert isinstance(response, LLMResponse)
    assert response.success is True
    assert isinstance(response.content, str)

    # Debug print the exact response
    print("\nReal API Response Content:")
    print(response.content)
    print("\n")

    # Parse and validate the response
    parsed_data = json.loads(response.content)

    # Validate the structured data
    assert isinstance(parsed_data, dict)
    assert "color" in parsed_data
    assert parsed_data["color"] in ["red", "blue", "green"]
    assert "count" in parsed_data
    assert isinstance(parsed_data["count"], int)
    assert parsed_data["count"] >= 1
    assert "tags" in parsed_data
    assert isinstance(parsed_data["tags"], list)
    assert all(isinstance(tag, str) for tag in parsed_data["tags"])

@pytest.mark.asyncio
@pytest.mark.integration
async def test_pydantic_model_direct_parse(test_settings, llm_registry: LLMRegistry):
    """Test using a Pydantic model directly with beta.chat.completions.parse."""
    if not test_settings.use_real_tokens:
        pytest.skip("Skipping test that requires LLM API tokens")

    # Arrange
    model_name = "gpt-4o-mini-2024-07-18"
    llm_instance = await llm_registry.create_instance(model_name)

    # Create a simple prompt that expects a Person response
    prompt = MemoryPrompt(
        name="person_direct",
        description="Create a person profile",
        system_prompt="Create a profile for a fictional person.",
        user_prompt="Create a profile for a software engineer in their 30s who enjoys coding and hiking.",
        expected_response=ResponseFormat.from_pydantic_model(
            model=Person,
            convert_to_json_schema=False,  # Important: Don't convert to JSON schema
            format_type=ResponseFormatType.PYDANTIC
        ),
        metadata=PromptMetadata(type="person_creation")
    )

    # Act
    response = await llm_instance.interface.process(prompt)

    # Assert
    assert isinstance(response, LLMResponse)
    assert response.success is True

    # Debug output
    print("\nDirect Parse Response Content:")
    print(response.content)
    print("\n")

    # Validate response can be parsed into Person model
    person = Person.model_validate_json(response.content)
    assert isinstance(person, Person)
    assert isinstance(person.name, str)
    assert isinstance(person.age, int)
    assert 30 <= person.age <= 39  # Verify age matches prompt requirements
    assert isinstance(person.hobbies, list)
    assert len(person.hobbies) >= 2  # Should have at least coding and hiking
    assert any("coding" in hobby.lower() for hobby in person.hobbies)
    assert any("hiking" in hobby.lower() for hobby in person.hobbies)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_nested_pydantic_model_direct_parse(test_settings, llm_registry: LLMRegistry):
    """Test using a nested Pydantic model directly with beta.chat.completions.parse."""
    if not test_settings.use_real_tokens:
        pytest.skip("Skipping test that requires LLM API tokens")

    # Arrange
    model_name = "gpt-4o-mini-2024-07-18"
    llm_instance = await llm_registry.create_instance(model_name)

    # Create a prompt that expects a NestedStructure response
    prompt = MemoryPrompt(
        name="team_direct",
        description="Create a development team profile",
        system_prompt="""Create a profile for a software development team.
The team should have:
- A descriptive title
- At least 3 team members
- Each team member should have realistic details
- Team size should match the number of people""",
        user_prompt="Create a profile for a full-stack development team working on a modern web application.",
        expected_response=ResponseFormat.from_pydantic_model(
            model=NestedStructure,
            convert_to_json_schema=False,  # Important: Don't convert to JSON schema
            format_type=ResponseFormatType.PYDANTIC
        ),
        metadata=PromptMetadata(type="team_creation")
    )

    # Act
    response = await llm_instance.interface.process(prompt)

    # Assert
    assert isinstance(response, LLMResponse)
    assert response.success is True

    # Debug output
    print("\nNested Parse Response Content:")
    print(response.content)
    print("\n")

    # Validate response can be parsed into NestedStructure model
    team = NestedStructure.model_validate_json(response.content)
    assert isinstance(team, NestedStructure)
    assert isinstance(team.title, str)
    assert isinstance(team.people, list)
    assert len(team.people) >= 3  # Verify minimum team size
    assert team.team_size == len(team.people)  # Verify team_size matches actual size

    # Validate nested Person objects
    for person in team.people:
        assert isinstance(person, Person)
        assert isinstance(person.name, str)
        assert isinstance(person.age, int)
        assert isinstance(person.hobbies, list)
        assert len(person.hobbies) > 0


@pytest.mark.asyncio
async def test_pydantic_model_validation_error_handling(test_settings, llm_registry: LLMRegistry):
    """Test handling of validation errors when using Pydantic models with parse endpoint."""
    if not test_settings.use_real_tokens:
        pytest.skip("Skipping test that requires LLM API tokens")

    # Arrange
    model_name = "gpt-4o-mini-2024-07-18"
    llm_instance = await llm_registry.create_instance(model_name)

    # Create a prompt that's likely to generate invalid data
    prompt = MemoryPrompt(
        name="person_invalid",
        description="Create an invalid person profile",
        system_prompt="Create a profile for a person, but make some fields invalid.",
        user_prompt="Create a profile for someone with an invalid age (use a negative number).",
        expected_response=ResponseFormat.from_pydantic_model(
            model=Person,
            convert_to_json_schema=False,
            format_type=ResponseFormatType.PYDANTIC
        ),
        metadata=PromptMetadata(type="person_creation_invalid")
    )

    # Act
    response = await llm_instance.interface.process(prompt)

    # Assert
    assert isinstance(response, LLMResponse)
    # The response might still be successful, but validation should fail
    try:
        Person.model_validate_json(response.content)
        pytest.fail("Expected validation error but got none")
    except Exception as e:
        assert "validation error" in str(e).lower()
