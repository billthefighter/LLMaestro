"""Tests for the ConfigurationManager class."""

import pytest
from pathlib import Path
from typing import Dict, Any

from llmaestro.config.manager import ConfigurationManager
from llmaestro.config.user import UserConfig
from llmaestro.config.system import SystemConfig
from llmaestro.config.agent import AgentPoolConfig, AgentTypeConfig
from llmaestro.llm import LLMRegistry
from llmaestro.llm.interfaces.base import BaseLLMInterface


@pytest.fixture
def sample_user_config() -> Dict[str, Any]:
    """Sample user configuration for testing."""
    return {
        "api_keys": {
            "openai": "test-key-openai",
            "anthropic": "test-key-anthropic"
        },
        "default_model": {
            "provider": "anthropic",
            "name": "claude-3-opus",
            "settings": {
                "max_tokens": 4096,
                "temperature": 0.7
            }
        },
        "agents": {
            "default": {
                "provider": "anthropic",
                "model": "claude-3-opus",
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "fast": {
                "provider": "anthropic",
                "model": "claude-3-sonnet",
                "max_tokens": 2048,
                "temperature": 0.8
            }
        }
    }


@pytest.fixture
def sample_system_config() -> Dict[str, Any]:
    """Sample system configuration for testing."""
    return {
        "providers": {
            "anthropic": {
                "name": "anthropic",
                "api_base": "https://api.anthropic.com/v1",
                "capabilities_detector": "llmaestro.llm.capability_detector.AnthropicCapabilityDetector",
                "rate_limits": {
                    "requests_per_minute": 50
                }
            },
            "openai": {
                "name": "openai",
                "api_base": "https://api.openai.com/v1",
                "capabilities_detector": "llmaestro.llm.capability_detector.OpenAICapabilityDetector",
                "rate_limits": {
                    "requests_per_minute": 60
                }
            }
        }
    }


@pytest.fixture
def config_manager(sample_user_config, sample_system_config) -> ConfigurationManager:
    """Create a ConfigurationManager instance for testing."""
    user_config = UserConfig.model_validate(sample_user_config)
    system_config = SystemConfig.model_validate(sample_system_config)
    return ConfigurationManager.from_configs(user_config=user_config, system_config=system_config)


@pytest.mark.unit
class TestConfigurationManager:
    """Test suite for ConfigurationManager."""

    def test_initialization(self, config_manager: ConfigurationManager):
        """Test that ConfigurationManager initializes correctly."""
        assert isinstance(config_manager.user_config, UserConfig)
        assert isinstance(config_manager.system_config, SystemConfig)
        assert isinstance(config_manager.llm_registry, LLMRegistry)
        assert isinstance(config_manager.agent_pool_config, AgentPoolConfig)

    def test_agents_property(self, config_manager: ConfigurationManager):
        """Test the agents property returns correct AgentPoolConfig."""
        agents = config_manager.agents
        assert isinstance(agents, AgentPoolConfig)
        assert "default" in agents.agent_types
        assert "fast" in agents.agent_types

    def test_get_agent_config(self, config_manager: ConfigurationManager):
        """Test getting agent configurations."""
        # Test default agent config
        default_config = config_manager.get_agent_config("default")
        assert isinstance(default_config, AgentTypeConfig)
        assert default_config.provider == "anthropic"
        assert default_config.model == "claude-3-opus"
        assert default_config.max_tokens == 4096

        # Test specific agent config
        fast_config = config_manager.get_agent_config("fast")
        assert isinstance(fast_config, AgentTypeConfig)
        assert fast_config.provider == "anthropic"
        assert fast_config.model == "claude-3-sonnet"
        assert fast_config.max_tokens == 2048

    @pytest.mark.asyncio
    async def test_get_model_interface(self, config_manager: ConfigurationManager):
        """Test getting model interfaces."""
        # Test getting interface for default agent
        interface = await config_manager.get_model_interface()
        assert isinstance(interface, BaseLLMInterface)

        # Test getting interface for specific agent
        fast_interface = await config_manager.get_model_interface("fast")
        assert isinstance(fast_interface, BaseLLMInterface)

    def test_from_yaml_files(self, tmp_path: Path):
        """Test creating ConfigurationManager from YAML files."""
        # Create temporary config files
        user_config_path = tmp_path / "user_config.yml"
        system_config_path = tmp_path / "system_config.yml"

        user_config_path.write_text("""
            api_keys:
                openai: test-key-openai
                anthropic: test-key-anthropic
            default_model:
                provider: anthropic
                name: claude-3-opus
                settings:
                    max_tokens: 4096
                    temperature: 0.7
            agents:
                default:
                    provider: anthropic
                    model: claude-3-opus
                    max_tokens: 4096
                    temperature: 0.7
        """)

        system_config_path.write_text("""
            providers:
                anthropic:
                    name: anthropic
                    api_base: https://api.anthropic.com/v1
                    capabilities_detector: llmaestro.llm.capability_detector.AnthropicCapabilityDetector
                    rate_limits:
                        requests_per_minute: 50
        """)

        config_manager = ConfigurationManager.from_yaml_files(
            user_config_path=str(user_config_path),
            system_config_path=str(system_config_path)
        )

        assert isinstance(config_manager, ConfigurationManager)
        assert config_manager.user_config.api_keys["openai"] == "test-key-openai"
        assert config_manager.system_config.providers["anthropic"].name == "anthropic"

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Test creating ConfigurationManager from environment variables."""
        # Create temporary system config file
        system_config_path = tmp_path / "system_config.yml"
        system_config_path.write_text("""
            providers:
                anthropic:
                    name: anthropic
                    api_base: https://api.anthropic.com/v1
        """)

        # Set environment variables
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-anthropic")

        config_manager = ConfigurationManager.from_env(system_config_path=str(system_config_path))

        assert isinstance(config_manager, ConfigurationManager)
        assert config_manager.user_config.api_keys["openai"] == "test-key-openai"
        assert config_manager.user_config.api_keys["anthropic"] == "test-key-anthropic"
