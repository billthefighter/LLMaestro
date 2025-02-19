"""Shared fixtures for application tests."""
import json
from typing import Any, Dict, Optional

import pytest

from llmaestro.llm.interfaces import LLMResponse


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return LLMResponse(
        content="Mock response",
        success=True,
        provider="mock",
        provider_metadata={"test": True},
    )


@pytest.fixture
def mock_llm_interface(mock_llm_response: Dict[str, Any]):
    """Mock LLM interface for testing."""
    class MockLLM:
        async def process(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
            return LLMResponse(
                content=json.dumps(mock_llm_response),
                metadata={"tokens": 100, "model": "test-model"},
            )
    return MockLLM()
