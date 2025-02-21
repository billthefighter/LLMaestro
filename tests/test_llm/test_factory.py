"""Tests for LLM interface factory functions."""
import pytest
from typing import Dict
from unittest.mock import Mock, patch


from llmaestro.llm.interfaces.factory import create_llm_interface, create_interface_for_model
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.llm.interfaces.provider_interfaces.anthropic import AnthropicLLM
from llmaestro.llm.interfaces.provider_interfaces.openai import OpenAIInterface
from llmaestro.llm.models import LLMProfile, ModelFamily

# Test data
TEST_API_KEY = "test-api-key-123"



class TestCreateLLMInterface:
    """Tests for the provider-based factory function."""

    @patch('src.llmaestro.llm.interfaces.base.LLMRegistry')
    def test_create_anthropic_interface(self, mock_registry_cls, mock_registry, agent_config):
        """Should create an Anthropic interface when provider is 'anthropic'."""
        mock_registry_cls.return_value = mock_registry
        agent_config.provider = "anthropic"
        interface = create_llm_interface(agent_config)

        assert isinstance(interface, AnthropicLLM)
        assert interface.config == agent_config

    @patch('src.llmaestro.llm.interfaces.base.LLMRegistry')
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

    @patch('src.llmaestro.llm.interfaces.base.LLMRegistry')
    def test_case_insensitive_provider(self, mock_registry_cls, mock_registry, agent_config):
        """Should handle provider names case-insensitively."""
        mock_registry_cls.return_value = mock_registry
        agent_config.provider = "ANTHROPIC"
        interface = create_llm_interface(agent_config)

        assert isinstance(interface, AnthropicLLM)
        assert interface.config == agent_config

class TestCreateInterfaceForModel:
    """Tests for the model-based factory function."""

    @patch('src.llmaestro.llm.interfaces.base.LLMRegistry')
    def test_create_anthropic_interface(self, mock_registry_cls, mock_registry, mock_LLMProfile, agent_config):
        """Should create an Anthropic interface for Claude models."""
        mock_registry_cls.return_value = mock_registry
        mock_LLMProfile.family = ModelFamily.CLAUDE
        interface = create_interface_for_model(mock_LLMProfile, agent_config)

        assert isinstance(interface, AnthropicLLM)
        assert interface.config == agent_config

    @patch('src.llmaestro.llm.interfaces.base.LLMRegistry')
    def test_create_openai_interface(self, mock_registry_cls, mock_registry, mock_LLMProfile, agent_config):
        """Should create an OpenAI interface for GPT models."""
        mock_registry_cls.return_value = mock_registry
        mock_LLMProfile.family = ModelFamily.GPT
        interface = create_interface_for_model(mock_LLMProfile, agent_config)

        assert isinstance(interface, OpenAIInterface)
        assert interface.config == agent_config

    def test_huggingface_not_implemented(self, mock_LLMProfile, agent_config):
        """Should raise NotImplementedError for HuggingFace models."""
        mock_LLMProfile.family = ModelFamily.HUGGINGFACE
        with pytest.raises(NotImplementedError):
            create_interface_for_model(mock_LLMProfile, agent_config)

    def test_unsupported_model_family(self, mock_LLMProfile, agent_config):
        """Should raise ValueError for unsupported model families."""
        mock_LLMProfile.family = "unsupported"
        with pytest.raises(ValueError, match="Unsupported model family"):
            create_interface_for_model(mock_LLMProfile, agent_config)

    @patch('src.llmaestro.llm.interfaces.base.LLMRegistry')
    def test_interface_inheritance(self, mock_registry_cls, mock_registry, mock_LLMProfile, agent_config):
        """Should ensure all created interfaces inherit from BaseLLMInterface."""
        mock_registry_cls.return_value = mock_registry
        for family in [ModelFamily.CLAUDE, ModelFamily.GPT]:
            mock_LLMProfile.family = family
            interface = create_interface_for_model(mock_LLMProfile, agent_config)
            assert isinstance(interface, BaseLLMInterface)

    @patch('src.llmaestro.llm.interfaces.base.LLMRegistry')
    def test_config_propagation(self, mock_registry_cls, mock_registry, mock_LLMProfile, agent_config):
        """Should ensure config is properly propagated to created interfaces."""
        mock_registry_cls.return_value = mock_registry
        mock_LLMProfile.family = ModelFamily.CLAUDE
        interface = create_interface_for_model(mock_LLMProfile, agent_config)

        assert interface.config == agent_config
        assert interface.config.model_name == "test-model"
        assert interface.config.api_key == TEST_API_KEY
