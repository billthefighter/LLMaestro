"""Root test configuration and common fixtures."""
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import pytest
from pydantic import BaseModel

from llmaestro.llm.models import (
    LLMCapabilities,
    LLMProfile,
    LLMMetadata,
    VisionCapabilities,
)
from llmaestro.llm.capabilities import RangeConfig
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.prompts.base import BasePrompt
from tests.test_prompts.conftest import sample_variables


class TestConfig(BaseModel):
    """Test configuration for fixtures."""
    use_real_tokens: bool = False
    test_provider: str = "openai"
    test_model: str = "gpt-4-turbo-preview"


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--use-llm-tokens",
        action="store_true",
        default=False,
        help="run tests that require LLM API tokens"
    )


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    )


@pytest.fixture(scope="session")
def test_settings(request) -> TestConfig:
    """Get test settings."""
    return TestConfig(
        use_real_tokens=request.config.getoption("--use-llm-tokens")
    )


@pytest.fixture
def llm_registry(mock_LLMProfile: LLMProfile) -> LLMRegistry:
    """Create a test LLMRegistry with default configurations."""
    registry = LLMRegistry.create_default()
    # Register the mock model
    registry._models[mock_LLMProfile.name] = mock_LLMProfile
    return registry


@pytest.fixture
def base_prompt() -> BasePrompt:
    """Create a base prompt for testing."""
    prompt = BasePrompt(
        name="test_prompt",
        description="Test prompt",
        system_prompt="You are a test assistant. Context: {context}",
        user_prompt="Hello {name}, {query}",
        variables=sample_variables(),
    )
    return prompt
