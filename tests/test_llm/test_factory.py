"""Tests for LLM interface factory functions."""
import pytest
from typing import Dict
from unittest.mock import Mock, patch

from llmaestro.core.models import AgentConfig
from llmaestro.llm.interfaces.factory import create_llm_interface, create_interface_for_model
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.llm.interfaces.provider_interfaces.anthropic import AnthropicLLM
from llmaestro.llm.interfaces.provider_interfaces.openai import OpenAIInterface
from llmaestro.llm.models import ModelDescriptor, ModelFamily, ModelRegistry

# Test data
TEST_API_KEY = "test-api-key-123"

@pytest.fixture
def mock_model_descriptor():
    """Create a mock model descriptor."""
    mock = Mock(spec=ModelDescriptor)
    mock.name = "test-model"
    mock.capabilities = Mock()
    mock.capabilities.supported_media_types = []
    return mock

@pytest.fixture
def mock_registry(mock_model_descriptor):
    """Create a mock model registry that returns our test model."""
    mock = Mock(spec=ModelRegistry)
    mock.get_model.return_value = mock_model_descriptor
    return mock

@pytest.fixture
def agent_config():
    """Create a basic agent config for testing."""
    return AgentConfig(
        provider="anthropic",
        model_name="test-model",
        api_key=TEST_API_KEY,
        max_tokens=100,
        temperature=0.7,
        top_p=1.0,
        max_context_tokens=8192,
        token_tracking=True,
    )

class TestCreateLLMInterface:
    """Tests for the provider-based factory function."""

    @patch('src.llmaestro.llm.interfaces.base.ModelRegistry')
    def test_create_anthropic_interface(self, mock_registry_cls, mock_registry, agent_config):
        """Should create an Anthropic interface when provider is 'anthropic'."""
        mock_registry_cls.return_value = mock_registry
        agent_config.provider = "anthropic"
        interface = create_llm_interface(agent_config)

        assert isinstance(interface, AnthropicLLM)
        assert interface.config == agent_config

    @patch('src.llmaestro.llm.interfaces.base.ModelRegistry')
    def test_create_openai_interface(self, mock_registry_cls, mock_registry, agent_config):
        """Should create an OpenAI interface when provider is 'openai'."""
        mock_registry_cls.return_value = mock_registry
        agent_config.provider = "openai"
        interface = create_llm_interface(agent_config)

        assert isinstance(interface, OpenAIInterface)
        assert interface.config == agent_config

    def test_unsupported_provider(self, agent_config):
        """Should raise ValueError for unsupported providers."""
        agent_config.provider = "unsupported"
        with pytest.raises(ValueError, match="Unsupported LLM provider: unsupported"):
            create_llm_interface(agent_config)

    @patch('src.llmaestro.llm.interfaces.base.ModelRegistry')
    def test_case_insensitive_provider(self, mock_registry_cls, mock_registry, agent_config):
        """Should handle provider names case-insensitively."""
        mock_registry_cls.return_value = mock_registry
        agent_config.provider = "ANTHROPIC"
        interface = create_llm_interface(agent_config)

        assert isinstance(interface, AnthropicLLM)
        assert interface.config == agent_config

class TestCreateInterfaceForModel:
    """Tests for the model-based factory function."""

    @patch('src.llmaestro.llm.interfaces.base.ModelRegistry')
    def test_create_anthropic_interface(self, mock_registry_cls, mock_registry, mock_model_descriptor, agent_config):
        """Should create an Anthropic interface for Claude models."""
        mock_registry_cls.return_value = mock_registry
        mock_model_descriptor.family = ModelFamily.CLAUDE
        interface = create_interface_for_model(mock_model_descriptor, agent_config)

        assert isinstance(interface, AnthropicLLM)
        assert interface.config == agent_config

    @patch('src.llmaestro.llm.interfaces.base.ModelRegistry')
    def test_create_openai_interface(self, mock_registry_cls, mock_registry, mock_model_descriptor, agent_config):
        """Should create an OpenAI interface for GPT models."""
        mock_registry_cls.return_value = mock_registry
        mock_model_descriptor.family = ModelFamily.GPT
        interface = create_interface_for_model(mock_model_descriptor, agent_config)

        assert isinstance(interface, OpenAIInterface)
        assert interface.config == agent_config

    def test_huggingface_not_implemented(self, mock_model_descriptor, agent_config):
        """Should raise NotImplementedError for HuggingFace models."""
        mock_model_descriptor.family = ModelFamily.HUGGINGFACE
        with pytest.raises(NotImplementedError):
            create_interface_for_model(mock_model_descriptor, agent_config)

    def test_unsupported_model_family(self, mock_model_descriptor, agent_config):
        """Should raise ValueError for unsupported model families."""
        mock_model_descriptor.family = "unsupported"
        with pytest.raises(ValueError, match="Unsupported model family"):
            create_interface_for_model(mock_model_descriptor, agent_config)

    @patch('src.llmaestro.llm.interfaces.base.ModelRegistry')
    def test_interface_inheritance(self, mock_registry_cls, mock_registry, mock_model_descriptor, agent_config):
        """Should ensure all created interfaces inherit from BaseLLMInterface."""
        mock_registry_cls.return_value = mock_registry
        for family in [ModelFamily.CLAUDE, ModelFamily.GPT]:
            mock_model_descriptor.family = family
            interface = create_interface_for_model(mock_model_descriptor, agent_config)
            assert isinstance(interface, BaseLLMInterface)

    @patch('src.llmaestro.llm.interfaces.base.ModelRegistry')
    def test_config_propagation(self, mock_registry_cls, mock_registry, mock_model_descriptor, agent_config):
        """Should ensure config is properly propagated to created interfaces."""
        mock_registry_cls.return_value = mock_registry
        mock_model_descriptor.family = ModelFamily.CLAUDE
        interface = create_interface_for_model(mock_model_descriptor, agent_config)

        assert interface.config == agent_config
        assert interface.config.model_name == "test-model"
        assert interface.config.api_key == TEST_API_KEY
