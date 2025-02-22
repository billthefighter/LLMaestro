import pytest
from datetime import datetime
from typing import Dict, List, Optional

from llmaestro.prompts.base import BasePrompt, PromptVariable, SerializableType, VersionedPrompt
from llmaestro.prompts.types import PromptMetadata, ResponseFormat, VersionInfo
from llmaestro.prompts.memory import MemoryPrompt


@pytest.fixture
def version_info() -> VersionInfo:
    """Sample version info for testing."""
    return VersionInfo(
        number="1.0.0",
        author="test",
        timestamp=datetime.now(),
        description="Initial version",
        change_type="initial",
    )


@pytest.fixture
def valid_prompt_data() -> Dict:
    """Base data for creating test prompts."""
    return {
        "name": "test_prompt",
        "description": "Test prompt",
        "system_prompt": "You are a test assistant. Context: {context}",
        "user_prompt": "Hello {user_name}, {query}",
        "metadata": PromptMetadata(
            type="test",
            expected_response=ResponseFormat(
                format="json",
                schema='{"type": "object"}'
            ),
        ),
    }


@pytest.fixture
def valid_versioned_prompt_data(valid_prompt_data: Dict, version_info: VersionInfo) -> Dict:
    """Base data for creating versioned test prompts."""
    data = valid_prompt_data.copy()
    data["current_version"] = version_info
    return data


@pytest.fixture
def sample_variables() -> List[PromptVariable]:
    """Sample prompt variables for testing."""
    return [
        PromptVariable(
            name="user_name",
            description="Name of the user to address",
            expected_input_type=SerializableType.STRING
        ),
        PromptVariable(
            name="query",
            description="The user's query",
            expected_input_type=SerializableType.STRING
        ),
        PromptVariable(
            name="context",
            description="Additional context for the assistant",
            expected_input_type=SerializableType.STRING
        ),
        PromptVariable(
            name="items",
            description="List of items to process",
            expected_input_type=SerializableType.LIST,
            string_conversion_template=lambda x: "\n".join(f"- {item}" for item in x)
        ),
        PromptVariable(
            name="count",
            description="Number of items",
            expected_input_type=SerializableType.INTEGER
        ),
        PromptVariable(
            name="settings",
            description="Processing settings",
            expected_input_type=SerializableType.DICT
        ),
        PromptVariable(
            name="response_schema",
            description="Expected response schema",
            expected_input_type=SerializableType.SCHEMA
        )
    ]


@pytest.fixture
def sample_variable_values() -> Dict:
    """Sample values for prompt variables."""
    return {
        "user_name": "Alice",
        "query": "help me process these items",
        "context": "test context",
        "items": ["item1", "item2", "item3"],
        "count": 3,
        "settings": {"mode": "fast", "detailed": True},
        "response_schema": {
            "type": "object",
            "properties": {
                "processed_items": {"type": "array"},
                "total": {"type": "integer"}
            }
        }
    }


@pytest.fixture
def base_prompt(valid_prompt_data: Dict, sample_variables: List[PromptVariable]) -> MemoryPrompt:
    """Test prompt without versioning."""
    data = valid_prompt_data.copy()
    data["variables"] = sample_variables
    return MemoryPrompt(**data)


@pytest.fixture
def versioned_prompt(valid_versioned_prompt_data: Dict, sample_variables: List[PromptVariable]) -> VersionedPrompt:
    """Test prompt with versioning."""
    class TestVersionedPrompt(VersionedPrompt):
        async def save(self) -> bool:
            return True

        @classmethod
        async def load(cls, identifier: str) -> Optional[VersionedPrompt]:
            return None

    data = valid_versioned_prompt_data.copy()
    data["variables"] = sample_variables
    return TestVersionedPrompt(**data)


@pytest.fixture
def variables_model(base_prompt: MemoryPrompt):
    """Get the variables model for the test prompt."""
    return base_prompt.get_variables_model()


@pytest.fixture
def invalid_variable_values() -> Dict:
    """Sample invalid values for prompt variables to test validation."""
    return {
        "user_name": 123,  # Should be string
        "query": ["not", "a", "string"],  # Should be string
        "context": None,  # Cannot be None
        "items": "not a list",  # Should be list
        "count": "3",  # Should be integer
        "settings": ["not", "a", "dict"],  # Should be dict
        "response_schema": lambda x: x  # Cannot serialize function
    }
