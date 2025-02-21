"""Tests for model descriptors and registry functionality."""
import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import json
import yaml

from llmaestro.llm.models import (
    ModelFamily,
    RangeConfig,
    LLMCapabilities,
    LLMProfile,
)

# Test data based on claude.yaml



def test_model_family_enum():
    assert ModelFamily.GPT == "gpt"
    assert ModelFamily.CLAUDE == "claude"
    assert ModelFamily.HUGGINGFACE == "huggingface"

def test_range_config():
    config = RangeConfig(min_value=0.0, max_value=1.0, default_value=0.5)
    assert config.min_value == 0.0
    assert config.max_value == 1.0
    assert config.default_value == 0.5

    with pytest.raises(ValueError):
        RangeConfig(min_value=1.0, max_value=0.0, default_value=0.5)

def test_model_capabilities(claude_2_data):
    capabilities = LLMCapabilities(**claude_2_data["capabilities"])
    assert capabilities.supports_streaming is True
    assert capabilities.max_context_window == 100000
    assert capabilities.input_cost_per_1k_tokens == 0.008
    assert capabilities.supported_languages == {"en"}
    assert isinstance(capabilities.temperature, RangeConfig)

def test_model_descriptor(claude_2_data):
    descriptor = LLMProfile(
        name=claude_2_data["name"],
        family=ModelFamily(claude_2_data["family"]),
        capabilities=LLMCapabilities(**claude_2_data["capabilities"]),
        is_preview=claude_2_data["is_preview"],
        is_deprecated=claude_2_data["is_deprecated"],
        min_api_version=claude_2_data["min_api_version"],
        release_date=datetime.fromisoformat(claude_2_data["release_date"])
    )

    assert descriptor.name == "claude-2"
    assert descriptor.family == ModelFamily.CLAUDE
    assert descriptor.capabilities.max_context_window == 100000
    assert descriptor.is_deprecated is False

def test_llm_registry_registration(llm_registry, claude_2_data):
    descriptor = LLMProfile(
        name=claude_2_data["name"],
        family=ModelFamily(claude_2_data["family"]),
        capabilities=LLMCapabilities(**claude_2_data["capabilities"]),
        is_preview=claude_2_data["is_preview"],
        is_deprecated=claude_2_data["is_deprecated"],
        min_api_version=claude_2_data["min_api_version"],
        release_date=datetime.fromisoformat(claude_2_data["release_date"])
    )

    llm_registry.register(descriptor)
    assert llm_registry.get_model("claude-2") == descriptor
    assert len(llm_registry.get_family_models(ModelFamily.CLAUDE)) == 1

def test_llm_registry_capability_filtering(llm_registry, claude_2_data):
    descriptor = LLMProfile(
        name=claude_2_data["name"],
        family=ModelFamily(claude_2_data["family"]),
        capabilities=LLMCapabilities(**claude_2_data["capabilities"]),
        is_preview=claude_2_data["is_preview"],
        is_deprecated=claude_2_data["is_deprecated"],
        min_api_version=claude_2_data["min_api_version"],
        release_date=datetime.fromisoformat(claude_2_data["release_date"])
    )

    llm_registry.register(descriptor)

    # Test filtering by capability
    models = llm_registry.get_models_by_capability("supports_streaming")
    assert len(models) == 1
    assert models[0].name == "claude-2"

    # Test filtering with constraints
    models = llm_registry.get_models_by_capability(
        "supports_streaming",
        min_context_window=200000  # Higher than Claude-2's window
    )
    assert len(models) == 0

def test_llm_registry_serialization(llm_registry, claude_2_data):
    descriptor = LLMProfile(
        name=claude_2_data["name"],
        family=ModelFamily(claude_2_data["family"]),
        capabilities=LLMCapabilities(**claude_2_data["capabilities"]),
        is_preview=claude_2_data["is_preview"],
        is_deprecated=claude_2_data["is_deprecated"],
        min_api_version=claude_2_data["min_api_version"],
        release_date=datetime.fromisoformat(claude_2_data["release_date"])
    )

    llm_registry.register(descriptor)

    # Test JSON serialization
    with tempfile.NamedTemporaryFile(suffix='.json') as tf:
        llm_registry.to_json(tf.name)
        loaded_registry = LLMRegistry.from_json(tf.name)
        loaded_model = loaded_registry.get_model("claude-2")
        assert loaded_model is not None
        assert loaded_model.model_dump(exclude_none=True) == descriptor.model_dump(exclude_none=True)

    # Test YAML serialization
    with tempfile.NamedTemporaryFile(suffix='.yaml') as tf:
        llm_registry.to_yaml(tf.name)
        loaded_registry = LLMRegistry.from_yaml(tf.name)
        loaded_model = loaded_registry.get_model("claude-2")
        assert loaded_model is not None
        assert loaded_model.model_dump(exclude_none=True) == descriptor.model_dump(exclude_none=True)

def test_llm_registry_validation(llm_registry, claude_2_data):
    descriptor = LLMProfile(
        name=claude_2_data["name"],
        family=ModelFamily(claude_2_data["family"]),
        capabilities=LLMCapabilities(**claude_2_data["capabilities"]),
        is_preview=claude_2_data["is_preview"],
        is_deprecated=claude_2_data["is_deprecated"],
        min_api_version=claude_2_data["min_api_version"],
        release_date=datetime.fromisoformat(claude_2_data["release_date"])
    )

    llm_registry.register(descriptor)

    # Test valid model
    validation_result = llm_registry.validate_model("claude-2")
    assert validation_result == (True, None)

    # Test unknown model
    is_valid, message = llm_registry.validate_model("unknown-model")
    assert not is_valid
    assert "Unknown model" in message
