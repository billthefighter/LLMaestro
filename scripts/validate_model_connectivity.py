"""Script for validating model connectivity and generating status badges."""
import asyncio
import json
import os
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


def get_model_registry() -> ModelRegistry:
    """Initialize model registry with all available models."""
    # Load Claude models
    claude_path = Path("src/llm/models/claude.yaml")
    registry = ModelRegistry.from_yaml(claude_path) if claude_path.exists() else ModelRegistry()

    # Load OpenAI models
    openai_path = Path("src/llm/models/openai.yaml")
    if openai_path.exists():
        openai_registry = ModelRegistry.from_yaml(openai_path)
        # Add OpenAI models to the registry
        for family in ModelFamily:
            models = openai_registry.get_family_models(family)
            for model in models:
                registry.add_model(model)

    return registry


def get_api_key(family: ModelFamily, config: Optional[Config] = None) -> Optional[str]:
    """Get the appropriate API key for a model family."""
    if family == ModelFamily.CLAUDE:
        # Try environment variable first, then config
        env_key = os.getenv("ANTHROPIC_API_KEY")
        if env_key:
            return env_key
        if config and config.llm.provider.lower() == "anthropic":
            return config.llm.api_key
    elif family == ModelFamily.GPT:
        # Try environment variable first, then config
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            return env_key
        if config and config.llm.provider.lower() == "openai":
            return config.llm.api_key
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


def create_badge_data(model_name: str, result: TestResult) -> Dict:
    """Create badge data for a model."""
    status_colors = {"success": "brightgreen", "failure": "red", "skip": "yellow"}

    status_messages = {"success": "operational", "failure": "failed", "skip": "no API key"}

    return {"schemaVersion": 1, "label": model_name, "message": status_messages[result], "color": status_colors[result]}


async def test_model(model: ModelDescriptor, config: Optional[Config], registry: ModelRegistry) -> TestResult:
    """Test connectivity for a specific model with detailed error handling."""
    api_key = get_api_key(model.family, config)
    if not api_key:
        print(f"⚠️  Skipping {model.name}: API key not set for {model.family.value}")
        return "skip"

    try:
        # Create interface using factory with proper config
        model_config = AgentConfig(
            provider=model.family.value,
            model_name=model.name,  # Ensure model name is set
            api_key=api_key,
            max_tokens=100,
        )
        interface = create_interface_for_model(model, model_config, registry)

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
        error_str = str(e)
        if "401" in error_str:
            print(f"❌ {model.name}: Authentication error (401) - Please check your API key")
            if "invalid x-api-key" in error_str.lower():
                print(
                    f"   Note: Environment variable for {model.family.value} appears to be invalid or not set correctly"
                )
        elif "403" in error_str:
            print(f"❌ {model.name}: Authorization error (403) - Please check your API permissions")
        elif "429" in error_str:
            print(f"❌ {model.name}: Rate limit exceeded (429) - Please try again later")
        else:
            print(f"❌ {model.name}: {error_type} - {error_str}")
        return "failure"


def save_results():
    """Save test results and generate badges."""
    results_dir = Path("test-results")
    results_dir.mkdir(exist_ok=True)
    badges_dir = results_dir / "badges"
    badges_dir.mkdir(exist_ok=True)

    # Save test results
    with open(results_dir / "model_connectivity.json", "w") as f:
        json.dump(TEST_RESULTS, f, indent=2)

    # Generate individual badge files
    for model_name, result in TEST_RESULTS.items():
        badge_data = create_badge_data(model_name, result)
        badge_file = badges_dir / f"{model_name}.json"
        with open(badge_file, "w") as f:
            json.dump(badge_data, f, indent=2)

    print("\nGenerated badges:")
    for model_name, result in TEST_RESULTS.items():
        print(f"{model_name}: {result}")


async def test_model_family(family: ModelFamily, config: Optional[Config], yaml_path: Path):
    """Test all models in a specific family."""
    if not yaml_path.exists():
        print(f"⚠️  No models found for {family.value} (missing {yaml_path})")
        return

    try:
        registry = ModelRegistry.from_yaml(yaml_path)
        models = registry.get_family_models(family)

        if not models:
            print(f"⚠️  No {family.value} models found in {yaml_path}")
            return

        print(f"\nTesting {family.value} models:")
        for model in models:
            TEST_RESULTS[model.name] = await test_model(model, config, registry)
    except Exception as e:
        print(f"❌ Error loading {family.value} models: {str(e)}")


async def main():
    """Main function to run model connectivity validation."""
    config = load_config()

    # Test Claude models
    claude_path = Path("src/llm/models/claude.yaml")
    await test_model_family(ModelFamily.CLAUDE, config, claude_path)

    # Test OpenAI models
    openai_path = Path("src/llm/models/openai.yaml")
    await test_model_family(ModelFamily.GPT, config, openai_path)

    # Save results and generate badges
    save_results()


if __name__ == "__main__":
    asyncio.run(main())
