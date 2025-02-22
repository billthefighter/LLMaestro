"""Tests for prompt mixins."""
import pytest
from datetime import datetime
from typing import Dict, List, Set, Type, Union, Optional, Any, Tuple
from pydantic import BaseModel, ValidationError, PrivateAttr

from llmaestro.prompts.mixins import TokenCountingMixin, VersionMixin
from llmaestro.llm.enums import ModelFamily
from llmaestro.llm.models import LLMProfile, LLMCapabilities, LLMMetadata
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.token_utils import TokenCounter


class TestPrompt(TokenCountingMixin):
    """Test implementation of TokenCountingMixin."""

    _system: str = PrivateAttr()
    _user: str = PrivateAttr()

    def __init__(self, system: str, user: str, llm_registry: LLMRegistry):
        super().__init__(llm_registry=llm_registry)
        self._system = system
        self._user = user

    @property
    def system_prompt(self) -> str:
        return self._system

    @property
    def user_prompt(self) -> str:
        return self._user

    def _extract_template_vars(self) -> Set[str]:
        """Simple implementation for testing."""
        import re
        pattern = r"\{([^}]+)\}"
        system_vars = set(re.findall(pattern, self._system))
        user_vars = set(re.findall(pattern, self._user))
        return system_vars.union(user_vars)

    def render(self, **kwargs) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Simple implementation for testing."""
        try:
            system = self._system.format(**kwargs)
            user = self._user.format(**kwargs)
            return system, user, []
        except KeyError as e:
            raise ValueError(f"Missing required variable: {str(e)}")


@pytest.fixture
def token_counting_prompt(llm_registry: LLMRegistry) -> TestPrompt:
    """Create a test prompt with token counting capabilities."""
    return TestPrompt(
        system="Test system prompt for {user_name}",
        user="Test user prompt with {query} and {context}",
        llm_registry=llm_registry
    )


def test_initialization_requires_registry():
    """Test that initialization requires LLM registry."""
    with pytest.raises(TypeError, match="missing.*required.*argument.*llm_registry"):
        # Note: We're intentionally not passing llm_registry to test the error
        TestPrompt(  # type: ignore
            system="Test system prompt",
            user="Test user prompt"
        )


def test_initialization_with_registry(llm_registry: LLMRegistry):
    """Test successful initialization with LLM registry."""
    prompt = TestPrompt(
        system="Test system prompt",
        user="Test user prompt",
        llm_registry=llm_registry
    )
    assert prompt.token_counter is not None
    assert isinstance(prompt.token_counter, TokenCounter)
    assert prompt._llm_registry is llm_registry


@pytest.mark.parametrize("model_family", [
    ModelFamily.GPT,
    #ModelFamily.CLAUDE, skipping because API key is required
])
def test_estimate_tokens(
    token_counting_prompt: TestPrompt,
    sample_variable_values: Dict,
    mock_LLMProfile: LLMProfile,
    model_family: ModelFamily,
):
    """Test token estimation with different models."""
    # Arrange
    mock_LLMProfile.capabilities.family = model_family  # Update family for test

    # Act
    result = token_counting_prompt.estimate_tokens(
        model_family,
        mock_LLMProfile.name,
        sample_variable_values
    )

    # Assert
    assert isinstance(result, dict)
    assert "total_tokens" in result
    assert "has_variables" in result
    assert result["model_family"] == model_family.name
    assert result["model_name"] == mock_LLMProfile.name


@pytest.fixture
def mock_small_context_profile() -> LLMProfile:
    """Create a mock LLM profile with a small context window for testing validation."""
    return LLMProfile(
        capabilities=LLMCapabilities(
            name="mock-small-context",
            family=ModelFamily.GPT,
            max_context_window=50,  # Small context window
            typical_speed=50.0,
            input_cost_per_1k_tokens=0.01,
            output_cost_per_1k_tokens=0.02,
            supports_streaming=True,
        ),
        metadata=LLMMetadata(
            release_date=datetime.now(),
            min_api_version="2024-02-29",
        ),
    )


@pytest.mark.parametrize("max_completion_tokens,expected_valid", [
    (10, True),  # Small enough for the context window
    (100, False),  # Too many tokens for small context window
])
def test_validate_context(
    token_counting_prompt: TestPrompt,
    sample_variable_values: Dict,
    mock_small_context_profile: LLMProfile,
    max_completion_tokens: int,
    expected_valid: bool,
):
    """Test context validation with different token limits."""
    # Arrange - Register the mock profile with the registry
    token_counting_prompt._llm_registry.register(mock_small_context_profile)

    # Act
    is_valid, error = token_counting_prompt.validate_context(
        mock_small_context_profile.capabilities.family,
        mock_small_context_profile.name,
        max_completion_tokens,
        sample_variable_values
    )

    # Assert
    assert is_valid == expected_valid
    if not expected_valid:
        assert error  # Error message should be non-empty for invalid cases
