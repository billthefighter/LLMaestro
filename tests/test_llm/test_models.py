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
    ModelCapabilities,
    ModelDescriptor,
    ModelRegistry,
    ModelCapabilitiesTable
)

# Test data based on claude.yaml
@pytest.fixture
def claude_2_data():
    return {
        "name": "claude-2",
        "family": "claude",
        "capabilities": {
            "supports_streaming": True,
            "supports_function_calling": False,
            "supports_vision": False,
            "supports_embeddings": False,
            "max_context_window": 100000,
            "max_output_tokens": 4096,
            "typical_speed": 70.0,
            "input_cost_per_1k_tokens": 0.008,
            "output_cost_per_1k_tokens": 0.024,
            "daily_request_limit": 150000,
            "supports_json_mode": False,
            "supports_system_prompt": True,
            "supports_message_role": True,
            "supports_tools": False,
            "supports_parallel_requests": True,
            "supported_languages": ["en"],
            "supported_mime_types": [],
            "temperature": {
                "min_value": 0.0,
                "max_value": 1.0,
                "default_value": 0.7
            },
            "top_p": {
                "min_value": 0.0,
                "max_value": 1.0,
                "default_value": 1.0
            },
            "supports_frequency_penalty": False,
            "supports_presence_penalty": False,
            "supports_stop_sequences": True,
            "supports_semantic_search": True,
            "supports_code_completion": True,
            "supports_chat_memory": False,
            "supports_few_shot_learning": True
        },
        "is_preview": False,
        "is_deprecated": False,
        "min_api_version": "2023-09-01",
        "release_date": "2023-07-11"
    }

@pytest.fixture
def model_registry():
    return ModelRegistry()

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
    capabilities = ModelCapabilities(**claude_2_data["capabilities"])
    assert capabilities.supports_streaming is True
    assert capabilities.max_context_window == 100000
    assert capabilities.input_cost_per_1k_tokens == 0.008
    assert capabilities.supported_languages == {"en"}
    assert isinstance(capabilities.temperature, RangeConfig)

def test_model_descriptor(claude_2_data):
    descriptor = ModelDescriptor(
        name=claude_2_data["name"],
        family=ModelFamily(claude_2_data["family"]),
        capabilities=ModelCapabilities(**claude_2_data["capabilities"]),
        is_preview=claude_2_data["is_preview"],
        is_deprecated=claude_2_data["is_deprecated"],
        min_api_version=claude_2_data["min_api_version"],
        release_date=datetime.fromisoformat(claude_2_data["release_date"])
    )

    assert descriptor.name == "claude-2"
    assert descriptor.family == ModelFamily.CLAUDE
    assert descriptor.capabilities.max_context_window == 100000
    assert descriptor.is_deprecated is False

def test_model_registry_registration(model_registry, claude_2_data):
    descriptor = ModelDescriptor(
        name=claude_2_data["name"],
        family=ModelFamily(claude_2_data["family"]),
        capabilities=ModelCapabilities(**claude_2_data["capabilities"]),
        is_preview=claude_2_data["is_preview"],
        is_deprecated=claude_2_data["is_deprecated"],
        min_api_version=claude_2_data["min_api_version"],
        release_date=datetime.fromisoformat(claude_2_data["release_date"])
    )

    model_registry.register(descriptor)
    assert model_registry.get_model("claude-2") == descriptor
    assert len(model_registry.get_family_models(ModelFamily.CLAUDE)) == 1

def test_model_registry_capability_filtering(model_registry, claude_2_data):
    descriptor = ModelDescriptor(
        name=claude_2_data["name"],
        family=ModelFamily(claude_2_data["family"]),
        capabilities=ModelCapabilities(**claude_2_data["capabilities"]),
        is_preview=claude_2_data["is_preview"],
        is_deprecated=claude_2_data["is_deprecated"],
        min_api_version=claude_2_data["min_api_version"],
        release_date=datetime.fromisoformat(claude_2_data["release_date"])
    )

    model_registry.register(descriptor)

    # Test filtering by capability
    models = model_registry.get_models_by_capability("supports_streaming")
    assert len(models) == 1
    assert models[0].name == "claude-2"

    # Test filtering with constraints
    models = model_registry.get_models_by_capability(
        "supports_streaming",
        min_context_window=200000  # Higher than Claude-2's window
    )
    assert len(models) == 0

def test_model_registry_serialization(model_registry, claude_2_data):
    descriptor = ModelDescriptor(
        name=claude_2_data["name"],
        family=ModelFamily(claude_2_data["family"]),
        capabilities=ModelCapabilities(**claude_2_data["capabilities"]),
        is_preview=claude_2_data["is_preview"],
        is_deprecated=claude_2_data["is_deprecated"],
        min_api_version=claude_2_data["min_api_version"],
        release_date=datetime.fromisoformat(claude_2_data["release_date"])
    )

    model_registry.register(descriptor)

    # Test JSON serialization
    with tempfile.NamedTemporaryFile(suffix='.json') as tf:
        model_registry.to_json(tf.name)
        loaded_registry = ModelRegistry.from_json(tf.name)
        loaded_model = loaded_registry.get_model("claude-2")
        assert loaded_model is not None
        assert loaded_model.model_dump(exclude_none=True) == descriptor.model_dump(exclude_none=True)

    # Test YAML serialization
    with tempfile.NamedTemporaryFile(suffix='.yaml') as tf:
        model_registry.to_yaml(tf.name)
        loaded_registry = ModelRegistry.from_yaml(tf.name)
        loaded_model = loaded_registry.get_model("claude-2")
        assert loaded_model is not None
        assert loaded_model.model_dump(exclude_none=True) == descriptor.model_dump(exclude_none=True)

def test_model_registry_validation(model_registry, claude_2_data):
    descriptor = ModelDescriptor(
        name=claude_2_data["name"],
        family=ModelFamily(claude_2_data["family"]),
        capabilities=ModelCapabilities(**claude_2_data["capabilities"]),
        is_preview=claude_2_data["is_preview"],
        is_deprecated=claude_2_data["is_deprecated"],
        min_api_version=claude_2_data["min_api_version"],
        release_date=datetime.fromisoformat(claude_2_data["release_date"])
    )

    model_registry.register(descriptor)

    # Test valid model
    validation_result = model_registry.validate_model("claude-2")
    assert validation_result == (True, None)

    # Test unknown model
    is_valid, message = model_registry.validate_model("unknown-model")
    assert not is_valid
    assert "Unknown model" in message

def test_model_capabilities_table(claude_2_data):
    descriptor = ModelDescriptor(
        name=claude_2_data["name"],
        family=ModelFamily(claude_2_data["family"]),
        capabilities=ModelCapabilities(**claude_2_data["capabilities"]),
        is_preview=claude_2_data["is_preview"],
        is_deprecated=claude_2_data["is_deprecated"],
        min_api_version=claude_2_data["min_api_version"],
        release_date=datetime.fromisoformat(claude_2_data["release_date"])
    )

    # Test conversion to database model
    db_model = ModelCapabilitiesTable.from_descriptor(descriptor)
    assert db_model.model_name == "claude-2"
    assert db_model.family == "claude"
    assert isinstance(db_model.capabilities, dict)

    # Test conversion back to descriptor
    recovered_descriptor = db_model.to_descriptor()
    assert recovered_descriptor.dict(exclude_none=True) == descriptor.dict(exclude_none=True)
