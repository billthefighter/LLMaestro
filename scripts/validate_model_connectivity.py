"""Script for validating model connectivity and generating status badges."""
import asyncio
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional

import yaml
from llmaestro.core.models import AgentConfig
from llmaestro.llm.interfaces.factory import create_interface_for_model
from llmaestro.llm.models import LLMProfile, LLMRegistry, ModelFamily
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.types import PromptMetadata, ResponseFormat, VersionInfo


class TestPrompt(BasePrompt):
    """Test prompt implementation."""

    async def save(self) -> bool:
        """Mock save implementation."""
        return True

    @classmethod
    async def load(cls, identifier: str) -> Optional["BasePrompt"]:
        """Mock load implementation."""
        return None


# Type for test results
TestResult = Literal["success", "failure", "skip"]
TEST_RESULTS: Dict[str, TestResult] = {}

# Test message that should work across all models
HELLO_WORLD_PROMPT = TestPrompt(
    name="Hello World Test",
    description="Simple test prompt for model connectivity",
    system_prompt="You are a helpful assistant. Keep responses brief.",
    user_prompt="Say hello world",
    metadata=PromptMetadata(type="test", expected_response=ResponseFormat(format="text", schema=None), tags=["test"]),
    current_version=VersionInfo(
        number="1.0.0", author="system", timestamp=datetime.now(), description="Initial version", change_type="create"
    ),
)
EXPECTED_RESPONSE_SUBSTRING = "hello world"


def get_llm_registry() -> LLMRegistry:
    """Initialize model registry with all available models."""
    registry = LLMRegistry()
    models_dir = Path("src/llm/models")

    if not models_dir.exists():
        print(f"⚠️  Models directory not found at {models_dir}")
        return registry

    # Load all YAML files in the models directory
    for yaml_path in models_dir.glob("*.yaml"):
        try:
            loaded_registry = LLMRegistry.from_yaml(yaml_path)
            for model in loaded_registry._models.values():
                registry.register(model)
            print(f"✅ Loaded models from {yaml_path}")
        except Exception as e:
            print(f"❌ Error loading {yaml_path}: {str(e)}")

    return registry


def get_api_key(family: ModelFamily) -> Optional[str]:
    """Get the appropriate API key for a model family."""
    env_keys = {
        ModelFamily.CLAUDE: "ANTHROPIC_API_KEY",
        ModelFamily.GPT: "OPENAI_API_KEY",
        ModelFamily.GEMINI: "GOOGLE_API_KEY",
    }
    env_var = env_keys.get(family)
    return os.getenv(env_var) if env_var else None


@dataclass
class Config:
    """Minimal configuration for model testing."""

    llm: AgentConfig
    storage_path: str = "chain_storage"
    logging_level: str = "INFO"


def load_config() -> Optional[Config]:
    """Load configuration from file or create minimal config."""
    try:
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            return None

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        llm_config = config_data.get("llm", {})
        default_settings = llm_config.get("default_settings", {})

        agent_config = AgentConfig(
            provider=llm_config.get("default_provider", "anthropic"),
            model_name="claude-3-5-sonnet-latest",  # Default model
            api_key="dummy-key",  # Will be overridden by environment variables
            max_tokens=default_settings.get("max_tokens", 100),
            temperature=default_settings.get("temperature", 0.7),
        )

        return Config(llm=agent_config)
    except Exception as e:
        print(f"Warning: Failed to load config: {e}")
        return None


async def test_model(model: LLMProfile, config: Optional[Config], registry: LLMRegistry) -> TestResult:
    """Test connectivity for a specific model with detailed error handling."""
    api_key = get_api_key(ModelFamily(model.family))
    if not api_key:
        print(f"⚠️  Skipping {model.name}: API key not set for {model.family}")
        return "skip"

    try:
        # Create interface using factory with proper config
        model_config = AgentConfig(
            provider=model.family.lower(),
            model_name=model.name,
            api_key=api_key,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_tokens=100,
        )
        interface = create_interface_for_model(model, model_config, registry)

        # Test connectivity with a simple BasePrompt
        response = await interface.process(prompt=HELLO_WORLD_PROMPT, variables=None)

        if response.content is None:
            print(f"❌ {model.name}: No response content")
            return "failure"

        # Case-insensitive comparison and ignore punctuation
        response_text = "".join(c.lower() for c in response.content if c.isalnum() or c.isspace())
        expected_text = "".join(c.lower() for c in EXPECTED_RESPONSE_SUBSTRING if c.isalnum() or c.isspace())

        if expected_text in response_text:
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
        elif "403" in error_str:
            print(f"❌ {model.name}: Authorization error (403) - Please check your API permissions")
        elif "429" in error_str:
            print(f"❌ {model.name}: Rate limit exceeded (429) - Please try again later")
        else:
            print(f"❌ {model.name}: {error_type} - {error_str}")
        return "failure"


def save_results(registry: LLMRegistry):
    """Save test results and generate badges."""
    results_dir = Path("test-results")
    results_dir.mkdir(exist_ok=True)

    # Save test results with model metadata
    results_with_metadata = {}
    for model_name, result in TEST_RESULTS.items():
        model = registry.get_model(model_name)
        if model:
            results_with_metadata[model_name] = {
                "result": result,
                "family": model.family,
                "description": model.description if hasattr(model, "description") else None,
            }

    # Save full results
    with open(results_dir / "model_connectivity.json", "w") as f:
        json.dump(results_with_metadata, f, indent=2)

    print("\nTest Results:")
    for model_name, data in results_with_metadata.items():
        print(f"{model_name} ({data['family']}): {data['result']}")

    # Run badge generation script
    try:
        subprocess.run(["python", "scripts/generate_model_badges.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate badges: {e}")


async def test_model_family(family: ModelFamily, config: Optional[Config], registry: LLMRegistry):
    """Test all models in a specific family."""
    models = registry.get_family_models(family)

    if not models:
        print(f"⚠️  No {family} models found in registry")
        return

    print(f"\nTesting {family} models:")
    for model in models:
        TEST_RESULTS[model.name] = await test_model(model, config, registry)


async def main():
    """Main function to run model connectivity validation."""
    config = load_config()
    registry = get_llm_registry()

    # Test all model families from the registry
    for family in ModelFamily:
        await test_model_family(family, config, registry)

    # Save results and generate badges
    save_results(registry)


if __name__ == "__main__":
    asyncio.run(main())
