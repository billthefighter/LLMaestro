"""Tests for validating model connectivity with a simple hello world prompt."""
import pytest
from typing import Dict, List, Optional, Union, Any, Literal
import json
import os
from pathlib import Path
import subprocess
import yaml

from src.llm.models import (
    ModelRegistry,
    register_all_models,
    ModelFamily,
    ModelDescriptor,
    ModelCapabilities,
)
from src.llm.interfaces.factory import create_interface_for_model, BaseLLMInterface
from src.core.models import AgentConfig
from src.core.config import get_config, Config, StorageConfig, VisualizationConfig, LoggingConfig

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
    def config(self) -> Optional[Config]:
        """Get configuration, trying config file first then environment variables."""
        try:
            # Create config directly from YAML data, bypassing schema validation
            config_path = Path("config/config.yaml")

            # Create minimal config if it doesn't exist
            if not config_path.exists():
                config_path.parent.mkdir(exist_ok=True)
                minimal_config = {
                    "llm": {
                        "provider": "anthropic",
                        "model": "claude-3-sonnet-latest",
                        "api_key": "dummy-key",
                        "max_tokens": 100
                    },
                    "storage": {
                        "path": "chain_storage",
                        "format": "json"
                    },
                    "visualization": {
                        "host": "localhost",
                        "port": 8765
                    },
                    "logging": {
                        "level": "INFO",
                        "file": "orchestrator.log"
                    }
                }
                with open(config_path, 'w') as f:
                    yaml.safe_dump(minimal_config, f)

            with open(config_path) as f:
                config_data = yaml.safe_load(f)

            return Config(
                llm=AgentConfig(**config_data["llm"]),
                storage=StorageConfig(**(config_data.get("storage", {}))),
                visualization=VisualizationConfig(**(config_data.get("visualization", {}))),
                logging=LoggingConfig(**(config_data.get("logging", {}))),
            )
        except Exception as e:
            print(f"Warning: Failed to load config: {e}")
            return None

    @pytest.fixture
    async def model_registry(self):
        """Initialize and return the model registry."""
        registry = ModelRegistry.from_yaml(Path("src/llm/models/claude.yaml"))

        # Pre-register all models as skipped
        for family in ModelFamily:
            models = registry.get_family_models(family)
            for model in models:
                TEST_RESULTS[model.name] = "skip"

        return registry

    @pytest.fixture
    def model(self, request, model_registry):
        """Provide a model for testing."""
        models = self.get_test_models(model_registry)
        return models[request.param] if request.param < len(models) else None

    def get_api_key(self, family: ModelFamily, config: Optional[Config] = None) -> Optional[str]:
        """Get the appropriate API key for a model family."""
        if family == ModelFamily.CLAUDE:
            # Try config first, then environment variable
            if config and config.llm.provider.lower() == "anthropic":
                return config.llm.api_key
            return os.getenv("ANTHROPIC_API_KEY")
        elif family == ModelFamily.GPT:
            # Try config first, then environment variable
            if config and config.llm.provider.lower() == "openai":
                return config.llm.api_key
            return os.getenv("OPENAI_API_KEY")
        return None

    def get_test_models(self, model_registry) -> List[ModelDescriptor]:
        """Get all registered models for testing."""
        models = []
        for family in ModelFamily:
            models.extend(model_registry.get_family_models(family))
        return models

    @pytest.mark.parametrize("model", range(10), indirect=True)
    async def test_model_connectivity(self, model: ModelDescriptor, model_registry, config):
        """Test connectivity for a specific model."""
        if model is None:
            pytest.skip("No model available at this index")

        api_key = self.get_api_key(model.family, config)
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
            interface: BaseLLMInterface = create_interface_for_model(model, config, model_registry)

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
