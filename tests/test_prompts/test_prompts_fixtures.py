"""Tests for prompts fixtures instantiation."""

import pytest
from datetime import datetime
from typing import Dict, List

from llmaestro.prompts.base import BasePrompt, PromptVariable, VersionedPrompt
from llmaestro.prompts.types import PromptMetadata, ResponseFormat, VersionInfo
from llmaestro.prompts.memory import MemoryPrompt


def test_version_info_fixture(version_info: VersionInfo):
    """Test that version_info fixture instantiates without error."""
    assert version_info is not None
    assert isinstance(version_info, VersionInfo)


def test_valid_prompt_data_fixture(valid_prompt_data: Dict):
    """Test that valid_prompt_data fixture instantiates without error."""
    assert valid_prompt_data is not None
    assert isinstance(valid_prompt_data, dict)


def test_valid_versioned_prompt_data_fixture(valid_versioned_prompt_data: Dict):
    """Test that valid_versioned_prompt_data fixture instantiates without error."""
    assert valid_versioned_prompt_data is not None
    assert isinstance(valid_versioned_prompt_data, dict)


def test_sample_variables_fixture(sample_variables: List[PromptVariable]):
    """Test that sample_variables fixture instantiates without error."""
    assert sample_variables is not None
    assert isinstance(sample_variables, list)
    assert all(isinstance(var, PromptVariable) for var in sample_variables)


def test_sample_variable_values_fixture(sample_variable_values: Dict):
    """Test that sample_variable_values fixture instantiates without error."""
    assert sample_variable_values is not None
    assert isinstance(sample_variable_values, dict)


def test_base_prompt_fixture(base_prompt: MemoryPrompt):
    """Test that base_prompt fixture instantiates without error."""
    assert base_prompt is not None
    assert isinstance(base_prompt, MemoryPrompt)


def test_versioned_prompt_fixture(versioned_prompt: VersionedPrompt):
    """Test that versioned_prompt fixture instantiates without error."""
    assert versioned_prompt is not None
    assert isinstance(versioned_prompt, VersionedPrompt)


def test_variables_model_fixture(variables_model):
    """Test that variables_model fixture instantiates without error."""
    assert variables_model is not None


def test_invalid_variable_values_fixture(invalid_variable_values: Dict):
    """Test that invalid_variable_values fixture instantiates without error."""
    assert invalid_variable_values is not None
    assert isinstance(invalid_variable_values, dict)
