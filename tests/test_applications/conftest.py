"""Shared fixtures for application tests."""
import json
from typing import Any, Dict, Optional

import pytest

from llmaestro.llm.interfaces import LLMResponse


@pytest.fixture
def mock_llm_response() -> Dict[str, Any]:
    """Mock LLM response for testing."""
    return {
        "summary": "Updated Config class to use Pydantic BaseModel",
        "affected_readmes": ["README.md", "docs/api/config.md"],
        "needs_readme_updates": True,
        "suggested_updates": {
            "README.md": "Update configuration section to reflect new Pydantic model usage",
            "docs/api/config.md": "Add documentation for new Dict-based configuration",
        },
    }


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
