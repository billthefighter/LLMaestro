"""Tests for validating model connectivity with a simple hello world prompt."""
import pytest
from typing import Dict, List, Optional, Union, Any
import json
import os
from pathlib import Path
from anthropic import AsyncAnthropic
from anthropic.types import ContentBlock, Message
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from src.llm.models import (
    ModelRegistry,
    register_all_models,
)

# Test message that should work across all models
HELLO_WORLD_PROMPT = "Say hello world"
EXPECTED_RESPONSE_SUBSTRING = "hello world"

# Dictionary to store test results for badge generation
TEST_RESULTS: Dict[str, bool] = {}

def save_test_results():
    """Save test results to a JSON file for badge generation."""
    results_dir = Path("test-results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "model_connectivity.json", "w") as f:
        json.dump(TEST_RESULTS, f, indent=2)

def get_message_text(content: Union[ContentBlock, Dict[str, Any], str]) -> str:
    """Extract text content from an Anthropic message block."""
    if isinstance(content, dict):
        return content.get('text', content.get('value', str(content)))
    if hasattr(content, 'text'):
        return str(getattr(content, 'text'))
    if hasattr(content, 'value'):
        return str(getattr(content, 'value'))
    return str(content)

@pytest.mark.asyncio
class TestModelConnectivity:
    """Test suite for validating model connectivity."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup before tests and save results after."""
        yield
        save_test_results()

    @pytest.fixture
    async def model_registry(self):
        """Initialize and return the model registry."""
        return await register_all_models(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    async def test_claude_3_5_sonnet_connectivity(self, model_registry):
        """Test connectivity to Claude 3.5 Sonnet."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        model_name = "claude-3-5-sonnet-latest"
        model = model_registry.get_model(model_name)
        if not model:
            pytest.skip(f"Model {model_name} not registered")

        try:
            # Create Anthropic client
            client = AsyncAnthropic(api_key=api_key)

            # Create a simple message
            response = await client.messages.create(
                model=model_name,
                max_tokens=100,
                messages=[{"role": "user", "content": HELLO_WORLD_PROMPT}]
            )

            # Extract text from the response
            message_text = get_message_text(response.content[0])
            assert EXPECTED_RESPONSE_SUBSTRING in message_text.lower()

            TEST_RESULTS["claude-3-5-sonnet"] = True

        except Exception as e:
            TEST_RESULTS["claude-3-5-sonnet"] = False
            raise e

    @pytest.mark.parametrize("model_name", ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"])
    async def test_openai_model_connectivity(self, model_registry, model_name: str):
        """Test connectivity to OpenAI models."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        model = model_registry.get_model(model_name)
        if not model:
            pytest.skip(f"Model {model_name} not registered")

        try:
            # Create OpenAI client
            client = AsyncOpenAI(api_key=api_key)

            # Create a simple message
            response: ChatCompletion = await client.chat.completions.create(
                model=model_name,
                max_tokens=100,
                messages=[{"role": "user", "content": HELLO_WORLD_PROMPT}]
            )

            # Extract and check response content
            message_content: Optional[str] = response.choices[0].message.content
            assert message_content is not None
            assert EXPECTED_RESPONSE_SUBSTRING in message_content.lower()

            TEST_RESULTS[model_name] = True

        except Exception as e:
            TEST_RESULTS[model_name] = False
            raise e
