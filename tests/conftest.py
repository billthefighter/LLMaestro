"""Root test configuration and common fixtures."""
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import yaml

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
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.prompts.base import PromptVariable, SerializableType
from llmaestro.default_library.default_llm_factory import LLMDefaultFactory
from llmaestro.llm.credentials import APIKey
import asyncio


# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
    use_tokens = request.config.getoption("--use-llm-tokens")
    logger = logging.getLogger(__name__)
    logger.info(f"Using real tokens: {use_tokens}")
    return TestConfig(use_real_tokens=use_tokens)


@pytest.fixture(scope="session")
def config_api_key(test_settings) -> Optional[APIKey]:
    """Load OpenAI API key from config.yaml if use_llm_tokens is enabled."""
    logger = logging.getLogger(__name__)

    if not test_settings.use_real_tokens:
        logger.info("Not using real tokens, returning None for API key")
        return None

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    if not config_path.exists():
        logger.warning("config.yaml not found")
        pytest.skip("config.yaml not found")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    openai_key = config.get("llm", {}).get("providers", {}).get("openai", {}).get("api_key")
    if not openai_key:
        logger.warning("OpenAI API key not found in config.yaml")
        pytest.skip("OpenAI API key not found in config.yaml")

    logger.info("Successfully loaded API key from config")
    return APIKey(
        key=openai_key,
        description="OpenAI API key loaded from config.yaml"
    )


@pytest.fixture(scope="session")
def llm_registry(test_settings, config_api_key) -> LLMRegistry:
    """Create a test LLMRegistry with configurations from config.yaml if available."""
    logger = logging.getLogger(__name__)

    if test_settings.use_real_tokens:
        if not config_api_key:
            logger.warning("Real tokens requested but no API key available")
            pytest.skip("Real tokens requested but no API key available")
        credential = {"openai": config_api_key}
        logger.info("Using real API key for registry")
    else:
        credential = {"openai": APIKey(key="TEST_KEY_NOT_REAL!")}
        logger.info("Using test API key for registry")

    factory = LLMDefaultFactory(credentials=credential)
    registry = asyncio.run(factory.DefaultLLMRegistryFactory())
    return registry


@pytest.fixture
def sample_variables() -> List[PromptVariable]:
    """Create sample prompt variables for testing."""
    print("Creating sample variables")  # Debug print
    vars = [
        PromptVariable(
            name="context",
            description="Additional context for the assistant",
            expected_input_type=SerializableType.STRING
        ),
        PromptVariable(
            name="name",
            description="Name to use in prompt",
            expected_input_type=SerializableType.STRING
        ),
        PromptVariable(
            name="query",
            description="User query",
            expected_input_type=SerializableType.STRING
        )
    ]
    print(f"Sample variables type: {type(vars)}")  # Debug print
    print(f"Sample variables content: {vars}")  # Debug print
    return vars


@pytest.fixture
def base_prompt(sample_variables) -> MemoryPrompt:
    """Create a base prompt for testing."""
    print(f"In base_prompt, sample_variables type: {type(sample_variables)}")  # Debug print
    print(f"In base_prompt, sample_variables content: {sample_variables}")  # Debug print
    prompt = MemoryPrompt(
        name="test_prompt",
        description="Test prompt",
        system_prompt="You are a test assistant. Context: {context}",
        user_prompt="Hello {name}, {query}",
        variables=sample_variables,
    )
    return prompt
