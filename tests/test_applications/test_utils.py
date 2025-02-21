"""Test utilities for application tests."""
import json
from typing import Any, Dict, Optional

from llmaestro.llm.interfaces import LLMResponse
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.llm.models import LLMCapabilities, ModelFamily
from llmaestro.llm.llm_registry import LLMRegistry
import pytest


class MockLLM(BaseLLMInterface):
    """Mock LLM implementation for testing."""

    def __init__(self, registry: LLMRegistry, response: Dict[str, Any]):
        self._registry = registry
        self._response = response

    @property
    def model_family(self) -> ModelFamily:
        return ModelFamily.CLAUDE

    @property
    def model_name(self) -> str:
        return "claude-3-sonnet-20240229"

    @property
    def capabilities(self) -> LLMCapabilities:
        model = self._registry.get_model(self.model_name)
        if model is None:
            return LLMCapabilities(
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                max_context_window=200000,
                input_cost_per_1k_tokens=0.015,
                output_cost_per_1k_tokens=0.015,
            )
        return model.capabilities

    async def process(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(self._response),
            metadata={"tokens": 100, "model": "test-model"},
        )

def test_extract_json_from_llm_response():
    """Test extracting JSON from LLM response."""
    # Test with valid JSON response
    response = LLMResponse(
        content='{"key": "value"}',
        success=True,
        provider="mock",
        provider_metadata={"test": True},
    )
    result = extract_json_from_llm_response(response)
    assert result == {"key": "value"}

    # Test with invalid JSON response
    response = LLMResponse(
        content="Not a JSON string",
        success=True,
        provider="mock",
        provider_metadata={"test": True},
    )
    with pytest.raises(ValueError):
        extract_json_from_llm_response(response)
