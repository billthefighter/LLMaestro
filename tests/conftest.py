"""Root test configuration and common fixtures."""
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

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
    RangeConfig,
    VisionCapabilities,
)
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.provider_registry import Provider, ProviderRegistry
from llmaestro.config import TestConfig


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


@pytest.fixture
def llm_registry() -> LLMRegistry:
    """Create a test LLMRegistry with default configurations."""
    return LLMRegistry.create_default()


@pytest.fixture
def mock_LLMProfile(test_settings: TestConfig, llm_registry: LLMRegistry) -> LLMProfile:
    """Create a mock LLMProfile for testing."""
    return llm_registry.get_model("mock-model") or next(iter(llm_registry._models.values()))


@pytest.fixture
def provider_registry(system_config: SystemConfig, llm_registry: LLMRegistry) -> ProviderRegistry:
    """Create a test ProviderRegistry with default configurations."""
    return llm_registry.provider_registry


@pytest.fixture
def agent_pool_config(user_config: UserConfig) -> AgentPoolConfig:
    """Create a test agent pool configuration."""
    return user_config.agents


@pytest.fixture
def config_manager(
    user_config: UserConfig,
    system_config: SystemConfig,
    llm_registry: LLMRegistry,
    provider_registry: ProviderRegistry,
    mock_LLMProfile: LLMProfile,
    test_settings: TestConfig
) -> ConfigurationManager:
    """Create a test configuration manager with all components initialized."""
    # Create a test provider with our mock model
    test_provider = Provider(
        name=test_settings.test_provider,
        api_base=f"https://api.{test_settings.test_provider}.com/v1",
        capabilities_detector=f"{test_settings.test_provider}.CapabilityDetector",
        rate_limits={"requests_per_minute": 60},
        features=set(),
        models={mock_LLMProfile.name: mock_LLMProfile}
    )
    provider_registry.register_provider(test_settings.test_provider, test_provider)

    # Create manager without registering models yet
    manager = ConfigurationManager.__new__(ConfigurationManager)
    ConfigurationManager.__init__(manager, user_config=user_config, system_config=system_config)

    # Set our test registries
    manager._provider_registry = provider_registry
    manager._llm_registry = llm_registry

    # Now register models
    manager._register_models()

    return manager
