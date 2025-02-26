"""Tests for the base prompt functionality."""
import pytest
from datetime import datetime
from pydantic import BaseModel, ValidationError
import base64
from typing import Dict, List, Set, Type, Union, Optional

from llmaestro.prompts.base import BasePrompt, PromptMetadata, FileAttachment, PromptVariable, SerializableType
from llmaestro.prompts.types import VersionInfo, ResponseFormat
from llmaestro.llm.enums import MediaType
from llmaestro.llm.models import LLMProfile
from llmaestro.llm.llm_registry import LLMRegistry


def test_get_variables_model(base_prompt: BasePrompt):
    """Test getting the variables model."""
    # Arrange & Act
    model = base_prompt.get_variables_model()

    # Assert
    assert model is not None
    assert issubclass(model, BaseModel)

    # Verify model validates correctly
    valid_data = {
        "user_name": "test",
        "query": "test query",
        "context": "test context",
        "items": ["item1"],
        "count": 1,
        "settings": {"key": "value"},
        "response_schema": {"type": "object"}
    }
    instance = model(**valid_data)
    assert instance is not None


def test_get_variable_types(base_prompt: BasePrompt):
    """Test getting variable types."""
    # Act
    types = base_prompt.get_variable_types()

    # Assert
    assert isinstance(types, dict)
    assert "user_name" in types
    assert types["user_name"] == SerializableType.STRING
    assert "count" in types
    assert types["count"] == SerializableType.INTEGER


def test_get_required_variables(base_prompt: BasePrompt):
    """Test getting required variables from templates."""
    # Act
    required = base_prompt.get_required_variables()

    # Assert
    assert isinstance(required, set)
    assert "user_name" in required
    assert "query" in required
    assert "context" in required


def test_add_and_clear_attachments(base_prompt: BasePrompt):
    """Test adding and clearing attachments."""
    # Arrange
    content = "test content"
    media_type = "text/plain"  # Use MIME type string instead of enum
    file_name = "test.txt"

    # Act - Add attachment
    base_prompt.add_attachment(content, media_type, file_name)

    # Assert attachment was added
    assert len(base_prompt.attachments) == 1
    assert base_prompt.attachments[0].content == content
    assert isinstance(base_prompt.attachments[0].media_type, MediaType)
    assert base_prompt.attachments[0].file_name == file_name

    # Act - Clear attachments
    base_prompt.clear_attachments()

    # Assert attachments were cleared
    assert len(base_prompt.attachments) == 0


def test_render_success(base_prompt: BasePrompt):
    """Test rendering prompt with valid variables."""
    # Arrange
    variables = {
        "user_name": "Alice",
        "query": "test",
        "context": "ctx",
        "items": ["item1", "item2"],
        "count": 2,
        "settings": {"mode": "test"},
        "response_schema": {"type": "object"}
    }

    # Act
    system, user, attachments = base_prompt.render(variables=variables)

    # Assert
    assert "test assistant" in system.lower()
    assert variables["user_name"] in user
    assert isinstance(attachments, list)


@pytest.mark.parametrize("variables,expected_error", [
    ({"user_name": 123, "query": "test", "context": "ctx"}, ValidationError),
    ({}, ValueError),  # Missing required variables
])
def test_render_failures(base_prompt: BasePrompt, variables: Dict, expected_error: Type[Exception]):
    """Test rendering prompt with invalid variables."""
    with pytest.raises(expected_error):
        base_prompt.render(variables=variables)


def test_render_with_model(base_prompt: BasePrompt, sample_variable_values: Dict):
    """Test rendering using the variables model."""
    # Arrange
    base_prompt._create_variables_model()  # Ensure model is created
    VariablesModel = base_prompt.get_variables_model()
    assert VariablesModel is not None, "Variables model should be created"
    variables = VariablesModel(**sample_variable_values)

    # Act
    system, user, attachments = base_prompt.render(variables=variables)

    # Assert
    assert "test assistant" in system.lower()
    assert sample_variable_values["user_name"] in user
    assert isinstance(attachments, list)


def test_validate_template_invalid(base_prompt: BasePrompt):
    """Test template validation with invalid templates."""
    # Arrange
    base_prompt.system_prompt = "Invalid {brace"

    # Act & Assert
    with pytest.raises(ValueError, match="Unbalanced braces"):
        base_prompt._validate_template()


def test_extract_template_vars(base_prompt: BasePrompt):
    """Test extracting template variables."""
    # Act
    vars = base_prompt._extract_template_vars()

    # Assert
    assert isinstance(vars, set)
    assert "user_name" in vars
    assert "query" in vars
    assert "context" in vars
