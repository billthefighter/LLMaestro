"""Tests for base level fixtures."""
from pathlib import Path

from llmaestro.core.config import (
    AgentPoolConfig,
    ConfigurationManager,
    SystemConfig,
    UserConfig,
)
from llmaestro.llm.models import ModelRegistry
from llmaestro.llm.provider_registry import ProviderRegistry


def test_test_config_dir(test_config_dir):
    """Test that test_config_dir fixture returns a valid path."""
    assert isinstance(test_config_dir, Path)
    assert test_config_dir.exists()
    assert test_config_dir.is_dir()
    assert (test_config_dir / "test_system_config.yml").exists()
    assert (test_config_dir / "test_user_config.yml").exists()


def test_system_config(system_config):
    """Test that system_config fixture returns a valid SystemConfig."""
    assert isinstance(system_config, SystemConfig)
    assert "openai" in system_config.providers


def test_user_config(user_config):
    """Test that user_config fixture returns a valid UserConfig."""
    assert isinstance(user_config, UserConfig)
    assert "openai" in user_config.api_keys


def test_provider_registry(provider_registry):
    """Test that provider_registry fixture returns a valid ProviderRegistry."""
    assert isinstance(provider_registry, ProviderRegistry)
    assert provider_registry.get_provider("openai") is not None


def test_model_registry(model_registry):
    """Test that model_registry fixture returns a valid ModelRegistry."""
    assert isinstance(model_registry, ModelRegistry)


def test_agent_pool_config(agent_pool_config):
    """Test that agent_pool_config fixture returns a valid AgentPoolConfig."""
    assert isinstance(agent_pool_config, AgentPoolConfig)
    assert "general" in agent_pool_config.agent_types
    assert "fast" in agent_pool_config.agent_types


def test_config_manager(config_manager):
    """Test that config_manager fixture returns a valid ConfigurationManager."""
    assert isinstance(config_manager, ConfigurationManager)
    assert config_manager.provider_registry is not None
    assert config_manager.model_registry is not None
    assert config_manager.user_config is not None
    assert config_manager.system_config is not None
