"""Tests for base level fixtures."""
from pathlib import Path
import logging
import pytest

from llmaestro.config import (
    AgentPoolConfig,
    ConfigurationManager,
    SystemConfig,
    UserConfig,
)
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.provider_registry import ProviderRegistry
from llmaestro.llm.models import LLMProfile


logger = logging.getLogger(__name__)


def test_mock_LLMProfile(mock_LLMProfile, test_settings):
    """Test that mock_LLMProfile fixture returns a valid LLMProfile."""
    assert isinstance(mock_LLMProfile, LLMProfile)
    assert mock_LLMProfile.capabilities.name == test_settings.test_model
    # Test core capabilities are properly loaded
    assert mock_LLMProfile.capabilities.max_context_window > 0
    if mock_LLMProfile.capabilities.max_output_tokens is not None:
        assert mock_LLMProfile.capabilities.max_output_tokens > 0
    assert isinstance(mock_LLMProfile.capabilities.supports_streaming, bool)
    assert isinstance(mock_LLMProfile.capabilities.input_cost_per_1k_tokens, float)


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
    assert provider_registry is not None
    assert len(provider_registry.list_providers()) > 0


def test_llm_registry(llm_registry):
    """Test that llm_registry fixture returns a valid LLMRegistry."""
    assert isinstance(llm_registry, LLMRegistry)
    assert len(llm_registry.models) > 0

    # Log available models using both method and property
    logger.info("Available models in llm_registry (using list_models()):")
    models = llm_registry.list_models()
    for model in models:
        logger.info(f"- {model}")

    logger.info("Available models in llm_registry (using models property):")
    for model in llm_registry.models:
        logger.info(f"- {model}")

    # Also check internal state for provider info
    logger.info("Model registry state with provider info:")
    for model_name in llm_registry.models:
        profile = llm_registry.get_model(model_name)
        if profile and hasattr(profile, 'capabilities'):
            provider = getattr(profile.capabilities, 'provider', 'unknown')
            logger.info(f"- {model_name} (provider: {provider})")
        else:
            logger.info(f"- {model_name} (provider: unknown)")


def test_agent_pool_config(agent_pool_config):
    """Test that agent_pool_config fixture returns a valid AgentPoolConfig."""
    assert isinstance(agent_pool_config, AgentPoolConfig)
    assert "general" in agent_pool_config.agent_types
    assert "fast" in agent_pool_config.agent_types


def test_config_manager(config_manager):
    """Test that config_manager fixture returns a valid ConfigurationManager."""
    assert isinstance(config_manager, ConfigurationManager)
    assert config_manager._provider_registry is not None
    assert config_manager._llm_registry is not None
    assert config_manager.user_config is not None
    assert config_manager.system_config is not None

    # Log available models from config manager's model registry
    logger.info("Available models in config_manager.llm_registry:")
    for model in config_manager._llm_registry.models:
        logger.info(f"- {model}")

    # Log available providers
    logger.info("Available providers in config_manager.provider_registry:")
    providers = config_manager._provider_registry.list_providers()
    for provider in providers:
        logger.info(f"- {provider.name}")
        logger.info("  Models:")
        for model_name in provider.models:
            logger.info(f"    - {model_name}")


def test_mock_llm_profile(mock_LLMProfile):
    """Test that mock_LLMProfile fixture returns a valid LLMProfile."""
    assert mock_LLMProfile is not None
    assert mock_LLMProfile.capabilities is not None
