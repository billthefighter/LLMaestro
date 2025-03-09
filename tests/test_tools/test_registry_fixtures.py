"""Tests for verifying the registry fixtures."""
import pytest

from llmaestro.tools.core import BasicFunctionGuard, ToolParams
from llmaestro.tools.registry import ToolRegistry


def test_function_guard_fixture(test_function_guard):
    """Test that the function guard fixture is created correctly."""
    assert isinstance(test_function_guard, BasicFunctionGuard)
    assert test_function_guard.is_safe_to_run(message="test")
    assert test_function_guard(message="test") == "test"
    assert test_function_guard(message="test", count=3) == "testtesttest"


def test_tool_params_fixture(test_tool_params):
    """Test that the tool params fixture is created correctly."""
    assert isinstance(test_tool_params, ToolParams)
    assert test_tool_params.name == "test_tool_function"
    assert "message" in test_tool_params.parameters["properties"]
    assert "count" in test_tool_params.parameters["properties"]
    assert test_tool_params.parameters["required"] == ["message"]


def test_model_params_fixture(test_model_params):
    """Test that the model params fixture is created correctly."""
    assert isinstance(test_model_params, ToolParams)
    assert test_model_params.name == "TestToolInput"
    assert "message" in test_model_params.parameters["properties"]
    assert "count" in test_model_params.parameters["properties"]


def test_empty_registry_fixture(empty_registry):
    """Test that the empty registry fixture is created correctly."""
    assert isinstance(empty_registry, ToolRegistry)
    assert len(empty_registry.get_all_tools()) == 0
    assert len(empty_registry.get_categories()) == 0


def test_populated_registry_fixture(populated_registry):
    """Test that the populated registry fixture is created correctly."""
    assert isinstance(populated_registry, ToolRegistry)
    assert len(populated_registry.get_all_tools()) == 2
    assert len(populated_registry.get_categories()) == 1
    assert "test" in populated_registry.get_categories()
    assert "test_function" in populated_registry.get_all_tools()
    assert "test_model" in populated_registry.get_all_tools()
