"""Test fixtures for configuration management."""
import os
from pathlib import Path

import pytest

from llmaestro.core.config import (
    AgentPoolConfig,
    AgentTypeConfig,
    ConfigurationManager,
    SystemConfig,
    UserConfig,
)
from llmaestro.llm.models import ModelRegistry
from llmaestro.llm.provider_registry import ProviderRegistry


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
def provider_registry(system_config: SystemConfig) -> ProviderRegistry:
    """Create a test provider registry."""
    registry = ProviderRegistry()
    for name, config in system_config.providers.items():
        registry.register_provider(name, config)
    return registry


@pytest.fixture
def model_registry(provider_registry: ProviderRegistry) -> ModelRegistry:
    """Create a test model registry."""
    return ModelRegistry(provider_registry)


@pytest.fixture
def agent_pool_config(user_config: UserConfig) -> AgentPoolConfig:
    """Create a test agent pool configuration."""
    agent_types = {}
    for name, config in user_config.agents["agent_types"].items():
        agent_types[name] = AgentTypeConfig(
            provider=config["provider"],
            model=config["model"],
            max_tokens=config.get("max_tokens", 8192),
            temperature=config.get("temperature", 0.7),
            description=config.get("description", f"{name} test agent"),
        )

    return AgentPoolConfig(
        max_agents=user_config.agents["max_agents"],
        default_agent_type=user_config.agents["default_agent_type"],
        agent_types=agent_types,
    )


@pytest.fixture
def config_manager(
    user_config: UserConfig,
    system_config: SystemConfig,
    provider_registry: ProviderRegistry,
    model_registry: ModelRegistry,
    agent_pool_config: AgentPoolConfig,
) -> ConfigurationManager:
    """Create a test configuration manager with all components initialized."""
    manager = ConfigurationManager()
    manager.initialize(user_config, system_config)

    # Override the registries with our test instances
    manager._provider_registry = provider_registry
    manager._model_registry = model_registry
    manager._agent_pool_config = agent_pool_config

    return manager
