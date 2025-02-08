"""Script for validating model connectivity and generating status badges."""
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional

import yaml

from src.core.models import AgentConfig
from src.llm.interfaces.factory import create_interface_for_model
from src.llm.models import (
    ModelDescriptor,
    ModelFamily,
    ModelRegistry,
)
from src.prompts.base import BasePrompt
from src.prompts.types import PromptMetadata, ResponseFormat, VersionInfo


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
                registry.register(model)

    # Load Gemini models
    gemini_path = Path("src/llm/models/gemini.yaml")
    if gemini_path.exists():
        gemini_registry = ModelRegistry.from_yaml(gemini_path)
        # Add Gemini models to the registry
        for family in ModelFamily:
            models = gemini_registry.get_family_models(family)
            for model in models:
                registry.register(model)

    return registry


def get_api_key(family: ModelFamily) -> Optional[str]:
    """Get the appropriate API key for a model family."""
    if family == ModelFamily.CLAUDE:
        return os.getenv("ANTHROPIC_API_KEY")
    elif family == ModelFamily.GPT:
        return os.getenv("OPENAI_API_KEY")
    elif family == ModelFamily.GEMINI:
        return os.getenv("GOOGLE_API_KEY")
    return None


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

        # Create minimal config if it doesn't exist
        if not config_path.exists():
            config_path.parent.mkdir(exist_ok=True)
            minimal_config = {
                "llm": {
                    "default_provider": "anthropic",
                    "providers": {
                        "anthropic": {
                            "api_key": os.getenv("ANTHROPIC_API_KEY", "dummy-key"),
                            "models": ["claude-3-sonnet-20240229"],
                        },
                        "google": {"api_key": os.getenv("GOOGLE_API_KEY", "dummy-key"), "models": ["gemini-pro"]},
                    },
                    "default_settings": {"max_tokens": 100, "temperature": 0.7},
                }
            }
            with open(config_path, "w") as f:
                yaml.safe_dump(minimal_config, f)

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Extract the relevant provider config based on model family
        llm_config = config_data.get("llm", {})
        default_settings = llm_config.get("default_settings", {})

        # Create AgentConfig with default provider settings
        agent_config = AgentConfig(
            provider=llm_config.get("default_provider", "anthropic"),
            model_name="claude-3-sonnet-20240229",  # Default model
            api_key="dummy-key",  # Will be overridden by environment variables
            max_tokens=default_settings.get("max_tokens", 100),
            temperature=default_settings.get("temperature", 0.7),
        )

        return Config(llm=agent_config)
    except Exception as e:
        print(f"Warning: Failed to load config: {e}")
        return None


def create_badge_data(model_name: str, result: TestResult) -> Dict:
    """Create badge data for a model."""
    status_colors: Dict[TestResult, str] = {"success": "brightgreen", "failure": "red", "skip": "yellow"}

    status_messages: Dict[TestResult, str] = {"success": "operational", "failure": "failed", "skip": "no API key"}

    return {"schemaVersion": 1, "label": model_name, "message": status_messages[result], "color": status_colors[result]}


async def test_model(model: ModelDescriptor, config: Optional[Config], registry: ModelRegistry) -> TestResult:
    """Test connectivity for a specific model with detailed error handling."""
    api_key = get_api_key(ModelFamily(model.family))
    if not api_key:
        print(f"⚠️  Skipping {model.name}: API key not set for {model.family}")
        return "skip"

    try:
        # Create interface using factory with proper config
        model_config = AgentConfig(
            provider=model.family.lower(),
            model_name=model.name,  # Ensure model name is set
            api_key=api_key,
            google_api_key=os.getenv("GOOGLE_API_KEY"),  # Add Google API key
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
            if "invalid x-api-key" in error_str.lower():
                print(f"   Note: Environment variable for {model.family} appears to be invalid or not set correctly")
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
        print(f"⚠️  No models found for {family} (missing {yaml_path})")
        return

    try:
        registry = ModelRegistry.from_yaml(yaml_path)
        models = registry.get_family_models(family)

        if not models:
            print(f"⚠️  No {family} models found in {yaml_path}")
            return

        print(f"\nTesting {family} models:")
        for model in models:
            TEST_RESULTS[model.name] = await test_model(model, config, registry)
    except Exception as e:
        print(f"❌ Error loading {family} models: {str(e)}")


async def main():
    """Main function to run model connectivity validation."""
    config = load_config()

    # Test Claude models
    claude_path = Path("src/llm/models/claude.yaml")
    await test_model_family(ModelFamily.CLAUDE, config, claude_path)

    # Test OpenAI models
    openai_path = Path("src/llm/models/openai.yaml")
    await test_model_family(ModelFamily.GPT, config, openai_path)

    # Test Gemini models
    gemini_path = Path("src/llm/models/gemini.yaml")
    await test_model_family(ModelFamily.GEMINI, config, gemini_path)

    # Save results and generate badges
    save_results()


if __name__ == "__main__":
    asyncio.run(main())
