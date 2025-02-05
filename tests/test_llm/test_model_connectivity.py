"""Tests for validating model connectivity with a simple hello world prompt."""
import pytest
from typing import Dict, List, Optional, Union, Any, Literal
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
    ModelFamily,
)

# Test message that should work across all models
HELLO_WORLD_PROMPT = "Say hello world"
EXPECTED_RESPONSE_SUBSTRING = "hello world"

# Dictionary to store test results for badge generation
TestResult = Literal["success", "failure", "skip"]
TEST_RESULTS: Dict[str, TestResult] = {}

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

    async def test_model_connectivity(self, model_registry):
        """Test connectivity for all registered models."""
        # Get API keys
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # Test each model family
        for family in ModelFamily:
            # Record skip status for all models in a family if API key is missing
            if family == ModelFamily.CLAUDE and not anthropic_api_key:
                models = model_registry.get_family_models(family)
                for model in models:
                    TEST_RESULTS[model.name] = "skip"
                pytest.skip("ANTHROPIC_API_KEY not set")
            elif family == ModelFamily.GPT and not openai_api_key:
                models = model_registry.get_family_models(family)
                for model in models:
                    TEST_RESULTS[model.name] = "skip"
                pytest.skip("OPENAI_API_KEY not set")
            elif family == ModelFamily.HUGGINGFACE:
                models = model_registry.get_family_models(family)
                for model in models:
                    TEST_RESULTS[model.name] = "skip"
                continue  # Skip HuggingFace models for now

            # Get all models for this family
            models = model_registry.get_family_models(family)
            if not models:
                continue

            for model in models:
                try:
                    if family == ModelFamily.CLAUDE:
                        # Test Anthropic model
                        client = AsyncAnthropic(api_key=anthropic_api_key)
                        response = await client.messages.create(
                            model=model.name,
                            max_tokens=100,
                            messages=[{"role": "user", "content": HELLO_WORLD_PROMPT}]
                        )
                        message_text = get_message_text(response.content[0])
                        assert EXPECTED_RESPONSE_SUBSTRING in message_text.lower()
                        TEST_RESULTS[model.name] = "success"

                    elif family == ModelFamily.GPT:
                        # Test OpenAI model
                        client = AsyncOpenAI(api_key=openai_api_key)
                        response = await client.chat.completions.create(
                            model=model.name,
                            max_tokens=100,
                            messages=[{"role": "user", "content": HELLO_WORLD_PROMPT}]
                        )
                        message_content = response.choices[0].message.content
                        assert message_content is not None
                        assert EXPECTED_RESPONSE_SUBSTRING in message_content.lower()
                        TEST_RESULTS[model.name] = "success"

                except Exception as e:
                    TEST_RESULTS[model.name] = "failure"
                    raise e
