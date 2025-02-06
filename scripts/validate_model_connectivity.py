"""Script for validating model connectivity and generating status badges."""
import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Literal, Optional

import yaml

from src.core.config import Config, LoggingConfig, StorageConfig, VisualizationConfig
from src.core.models import AgentConfig
from src.llm.interfaces.factory import create_interface_for_model
from src.llm.models import (
    ModelDescriptor,
    ModelFamily,
    ModelRegistry,
)

# Test message that should work across all models
HELLO_WORLD_PROMPT = "Say hello world"
EXPECTED_RESPONSE_SUBSTRING = "hello world"

# Type for test results
TestResult = Literal["success", "failure", "skip"]
TEST_RESULTS: Dict[str, TestResult] = {}


def get_api_key(family: ModelFamily, config: Optional[Config] = None) -> Optional[str]:
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


def load_config() -> Optional[Config]:
    """Load configuration from file or create minimal config."""
    try:
        config_path = Path("config/config.yaml")

        # Create minimal config if it doesn't exist
        if not config_path.exists():
            config_path.parent.mkdir(exist_ok=True)
            minimal_config = {
                "llm": {
                    "provider": "anthropic",
                    "model": "claude-3-sonnet-latest",
                    "api_key": "dummy-key",
                    "max_tokens": 100,
                },
                "storage": {"path": "chain_storage", "format": "json"},
                "visualization": {"host": "localhost", "port": 8765},
                "logging": {"level": "INFO", "file": "orchestrator.log"},
            }
            with open(config_path, "w") as f:
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


async def test_model(model: ModelDescriptor, config: Optional[Config]) -> TestResult:
    """Test connectivity for a specific model with detailed error handling."""
    api_key = get_api_key(model.family, config)
    if not api_key:
        print(f"⚠️  Skipping {model.name}: API key not set for {model.family.value}")
        return "skip"

    try:
        # Create interface using factory with proper config
        model_config = AgentConfig(provider=model.family.value, model_name=model.name, api_key=api_key, max_tokens=100)
        interface = create_interface_for_model(model, model_config, model_registry)

        # Test connectivity with a simple message
        response = await interface.process(
            input_data=HELLO_WORLD_PROMPT, system_prompt="You are a helpful assistant. Keep responses brief."
        )

        if response.content is None:
            print(f"❌ {model.name}: No response content")
            return "failure"

        if EXPECTED_RESPONSE_SUBSTRING in response.content.lower():
            print(f"✅ {model.name}: Success")
            return "success"
        else:
            print(f"❌ {model.name}: Unexpected response - {response.content}")
            return "failure"

    except Exception as e:
        error_type = type(e).__name__
        if "401" in str(e):
            print(f"❌ {model.name}: Authentication error (401) - Please check your API key")
        elif "403" in str(e):
            print(f"❌ {model.name}: Authorization error (403) - Please check your API permissions")
        elif "429" in str(e):
            print(f"❌ {model.name}: Rate limit exceeded (429) - Please try again later")
        else:
            print(f"❌ {model.name}: {error_type} - {str(e)}")
        return "failure"


def save_results():
    """Save test results and generate badges."""
    results_dir = Path("test-results")
    results_dir.mkdir(exist_ok=True)
    badges_dir = results_dir / "badges"
    badges_dir.mkdir(exist_ok=True)

    with open(results_dir / "model_connectivity.json", "w") as f:
        json.dump(TEST_RESULTS, f, indent=2)

    try:
        subprocess.run(["python", "scripts/generate_model_badges.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to generate badges: {e}")


async def main():
    """Main function to run model connectivity validation."""
    global model_registry
    config = load_config()

    # Initialize model registry
    model_registry = ModelRegistry.from_yaml(Path("src/llm/models/claude.yaml"))

    # Pre-register all models as skipped
    for family in ModelFamily:
        models = model_registry.get_family_models(family)
        for model in models:
            TEST_RESULTS[model.name] = "skip"

    # Test each model
    for family in ModelFamily:
        models = model_registry.get_family_models(family)
        for model in models:
            TEST_RESULTS[model.name] = await test_model(model, config)

    # Save results and generate badges
    save_results()


if __name__ == "__main__":
    asyncio.run(main())
