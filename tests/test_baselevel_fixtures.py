"""Tests for base level fixtures."""
from pathlib import Path
import logging
import pytest
from llmaestro.llm.models import LLMProfile
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.prompts.base import BasePrompt


def test_test_settings_initialization(test_settings):
    """Test that test_settings fixture initializes correctly."""
    assert hasattr(test_settings, "use_real_tokens")
    assert hasattr(test_settings, "test_provider")
    assert hasattr(test_settings, "test_model")
    assert isinstance(test_settings.use_real_tokens, bool)
    assert isinstance(test_settings.test_provider, str)
    assert isinstance(test_settings.test_model, str)


def test_llm_registry_initialization(llm_registry):
    """Test that llm_registry fixture initializes correctly."""
    assert isinstance(llm_registry, LLMRegistry)


def test_base_prompt_initialization(base_prompt, sample_variables):
    """Test that base_prompt fixture initializes correctly."""
    assert isinstance(base_prompt, BasePrompt)
    assert base_prompt.name == "test_prompt"
    assert base_prompt.description == "Test prompt"
    assert "{context}" in base_prompt.system_prompt
    assert "{name}" in base_prompt.user_prompt
    assert "{query}" in base_prompt.user_prompt


@pytest.mark.parametrize("use_tokens", [True, False])
def test_settings_with_different_token_settings(request, use_tokens):
    """Test test_settings fixture with different token settings."""
    from conftest import TestConfig

    # Create settings directly with the desired token setting
    settings = TestConfig(use_real_tokens=use_tokens)
    assert settings.use_real_tokens == use_tokens
    assert settings.test_provider == "openai"  # verify default value
    assert settings.test_model == "gpt-4-turbo-preview"  # verify default value
