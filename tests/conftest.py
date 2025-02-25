"""Root test configuration and common fixtures."""
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import pytest
from pydantic import BaseModel

from llmaestro.config import (
    AgentPoolConfig,
    AgentTypeConfig,
    ConfigurationManager,
    SystemConfig,
    UserConfig,
)
from llmaestro.config.base import LLMProfileReference
from llmaestro.llm.models import (
    LLMCapabilities,
    LLMProfile,
    ModelFamily,
    LLMMetadata,
    VisionCapabilities,
)
from llmaestro.llm.capabilities import RangeConfig
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.capability_detector import BaseCapabilityDetector
from llmaestro.llm.token_utils import BaseTokenizer, TokenCounter, TokenizerRegistry
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
def test_config_dir() -> Path:
    """Get the test configuration directory."""
    return Path(__file__).parent / "config_files"


@pytest.fixture
def system_config(test_config_dir: Path) -> SystemConfig:
    """Create a test system configuration."""
    config_path = test_config_dir / "test_system_config.yml"
    return SystemConfig.from_yaml(config_path)


@pytest.fixture
def user_config(test_config_dir: Path) -> UserConfig:
    """Create a test user configuration."""
    config_path = test_config_dir / "test_user_config.yml"
    return UserConfig.from_yaml(config_path)


class MockCapabilityDetector(BaseCapabilityDetector):
    """Mock capability detector for testing."""

    @classmethod
    async def detect_capabilities(cls, model_name: str, api_key: str) -> LLMCapabilities:
        """Return mock capabilities."""
        return LLMCapabilities(
            name=model_name,
            family=ModelFamily.GPT,
            max_context_window=4096,
            typical_speed=50.0,
            input_cost_per_1k_tokens=0.01,
            output_cost_per_1k_tokens=0.02,
            supports_streaming=True,
        )


@pytest.fixture
def llm_registry(mock_LLMProfile: LLMProfile) -> LLMRegistry:
    """Create a test LLMRegistry with default configurations."""
    registry = LLMRegistry.create_default()
    # Register the mock model
    registry._models[mock_LLMProfile.name] = mock_LLMProfile
    return registry


@pytest.fixture
def mock_LLMProfile(test_settings: TestConfig) -> LLMProfile:
    """Create a mock LLMProfile for testing."""
    return LLMProfile(
        capabilities=LLMCapabilities(
            name=test_settings.test_model,
            family=ModelFamily.GPT,
            max_context_window=4096,
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


@pytest.fixture
def agent_pool_config(user_config: UserConfig) -> AgentPoolConfig:
    """Create a test agent pool configuration."""
    return user_config.agents


@pytest.fixture
def config_manager(
    user_config: UserConfig,
    system_config: SystemConfig,
    llm_registry: LLMRegistry,
    mock_LLMProfile: LLMProfile,
    test_settings: TestConfig
) -> ConfigurationManager:
    """Create a test configuration manager with all components initialized."""
    # First, ensure the system config uses our mock detector
    if test_settings.test_provider in system_config.providers:
        provider_config = system_config.providers[test_settings.test_provider]
        if isinstance(provider_config, dict):
            provider_config["capabilities_detector"] = "tests.conftest.MockCapabilityDetector"
        else:
            provider_config.capabilities_detector = "tests.conftest.MockCapabilityDetector"

    # Create the manager with our pre-configured system config
    manager = ConfigurationManager(
        user_config=user_config,
        system_config=system_config,
        llm_registry=llm_registry  # Pass our pre-configured registry
    )

    # Register our mock model
    llm_registry.register(mock_LLMProfile)

    return manager


class MockTokenizer(BaseTokenizer):
    """Mock tokenizer for testing."""

    def count_tokens(self, text: str) -> int:
        """Simple token count estimation."""
        return len(text.split())

    def encode(self, text: str) -> List[int]:
        """Simple encoding."""
        return [1] * len(text.split())

    @classmethod
    def supports_model(cls, model_family: ModelFamily) -> bool:
        """Support all model families."""
        return True


@pytest.fixture
def mock_token_counter(llm_registry: LLMRegistry) -> TokenCounter:
    """Create a mock token counter."""
    # Register our mock tokenizer for all model families
    for family in ModelFamily:
        TokenizerRegistry.register(family, MockTokenizer)
    return TokenCounter(llm_registry=llm_registry)


@pytest.fixture
def base_prompt(mock_token_counter: TokenCounter) -> BasePrompt:
    """Create a base prompt for testing."""
    prompt = BasePrompt(
        name="test_prompt",
        description="Test prompt",
        system_prompt="You are a test assistant. Context: {context}",
        user_prompt="Hello {name}, {query}",
        variables=sample_variables(),
    )
    prompt.token_counter = mock_token_counter
    return prompt
