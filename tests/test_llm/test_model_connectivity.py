"""Tests for validating model connectivity with a simple hello world prompt."""
import pytest
from typing import Dict, List, Optional, Union, Any, Literal
import json
import os
from pathlib import Path
import subprocess

from src.llm.models import (
    ModelRegistry,
    register_all_models,
    ModelFamily,
    ModelDescriptor,
)
from src.llm.interfaces import create_interface_for_model, BaseLLMInterface
from src.core.models import AgentConfig

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
    badges_dir = results_dir / "badges"
    badges_dir.mkdir(exist_ok=True)

    with open(results_dir / "model_connectivity.json", "w") as f:
        json.dump(TEST_RESULTS, f, indent=2)

    # Generate badges
    try:
        subprocess.run(["python", "scripts/generate_model_badges.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to generate badges: {e}")

@pytest.mark.asyncio
class TestModelConnectivity:
    """Test suite for validating model connectivity."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup before tests and save results after."""
        yield
        # Ensure results are saved even if tests are skipped
        save_test_results()

    @pytest.fixture
    async def model_registry(self):
        """Initialize and return the model registry."""
        registry = await register_all_models(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Pre-register all models as skipped
        for family in ModelFamily:
            models = registry.get_family_models(family)
            for model in models:
                TEST_RESULTS[model.name] = "skip"

        return registry

    def get_api_key(self, family: ModelFamily) -> Optional[str]:
        """Get the appropriate API key for a model family."""
        if family == ModelFamily.CLAUDE:
            return os.getenv("ANTHROPIC_API_KEY")
        elif family == ModelFamily.GPT:
            return os.getenv("OPENAI_API_KEY")
        return None

    def get_test_models(self, model_registry) -> List[ModelDescriptor]:
        """Get all registered models for testing."""
        models = []
        for family in ModelFamily:
            models.extend(model_registry.get_family_models(family))
        return models

    @pytest.mark.parametrize("model", "get_test_models")
    async def test_model_connectivity(self, model: ModelDescriptor, model_registry):
        """Test connectivity for a specific model."""
        api_key = self.get_api_key(model.family)
        if not api_key:
            TEST_RESULTS[model.name] = "skip"
            pytest.skip(f"API key not set for {model.family.value}")

        try:
            # Create interface using factory with proper config
            config = AgentConfig(
                provider=model.family.value,
                model_name=model.name,
                api_key=api_key,
                max_tokens=100
            )
            interface: BaseLLMInterface = create_interface_for_model(model, config)

            # Test connectivity with a simple message
            response = await interface.process(
                input_data=HELLO_WORLD_PROMPT,
                system_prompt="You are a helpful assistant. Keep responses brief."
            )

            # Check response
            assert response.content is not None
            assert EXPECTED_RESPONSE_SUBSTRING in response.content.lower()
            TEST_RESULTS[model.name] = "success"

        except Exception as e:
            TEST_RESULTS[model.name] = "failure"
            raise e
