"""Test structured output functionality with LLMs."""

import json
import pytest
from typing import List, Optional
from pydantic import BaseModel
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.prompts.types import ResponseFormat, PromptMetadata
from llmaestro.llm.responses import ResponseFormatType
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.core.models import LLMResponse


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
        expected_response=ResponseFormat(
                format=ResponseFormatType.JSON_SCHEMA,
                response_schema=json.dumps(Person.model_json_schema()),
                pydantic_model=Person,
                convert_to_json_schema=True
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
Make sure to include all required fields and validate against the provided schema.""",
        user_prompt="Please create a profile for a fictional team with a title, list of team members (at least 2), and team size.",
        metadata=PromptMetadata(
            type="team_extraction",
            tags=["structured", "team", "test"]
        ),
        expected_response=ResponseFormat(
                format=ResponseFormatType.JSON_SCHEMA,
                response_schema=json.dumps(NestedStructure.model_json_schema()),
                pydantic_model=NestedStructure,
                convert_to_json_schema=True
            ),
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_simple_structured_output(test_settings, llm_registry: LLMRegistry, person_prompt: MemoryPrompt):
    """Test simple structured output with a Person model."""
    if not test_settings.use_real_tokens:
        pytest.skip("Skipping test that requires LLM API tokens")

    # Arrange
    model_name = llm_registry.get_registered_models()[0]
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
    model_name = llm_registry.get_registered_models()[0]
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
async def test_direct_schema_output(test_settings, llm_registry: LLMRegistry):
    """Test structured output with direct JSON schema (without Pydantic model)."""
    if not test_settings.use_real_tokens:
        pytest.skip("Skipping test that requires LLM API tokens")

    # Arrange
    schema = {
        "type": "object",
        "properties": {
            "color": {"type": "string", "enum": ["red", "blue", "green"]},
            "count": {"type": "integer", "minimum": 1},
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["color", "count", "tags"]
    }

    prompt = MemoryPrompt(
        name="schema_test",
        description="Test direct schema validation",
        system_prompt="You are a helpful assistant that provides structured data.",
        user_prompt="Please provide a color (red, blue, or green), a count (positive integer), and some tags.",
        metadata=PromptMetadata(
            type="schema_test",
            expected_response=ResponseFormat(
                format=ResponseFormatType.JSON_SCHEMA,
                response_schema=json.dumps(schema),
                convert_to_json_schema=False
            ),
            tags=["structured", "schema", "test"]
        )
    )

    # Act
    model_name = llm_registry.get_registered_models()[0]
    llm_instance = await llm_registry.create_instance(model_name)
    response = await llm_instance.interface.process(prompt)

    # Assert
    assert isinstance(response, LLMResponse)
    assert response.success is True
    assert isinstance(response.content, str)

    # Debug print the exact response
    print("\nDirect Schema Response Content:")
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
