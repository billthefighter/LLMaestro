import json
import yaml
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import pytest

from llmaestro.prompts.loader import PromptLoader, FilePrompt
from llmaestro.llm.interfaces.provider_interfaces.anthropic import AnthropicLLM
from llmaestro.llm.interfaces.base import ConversationContext
from llmaestro.config import AgentTypeConfig
from llmaestro.prompts.types import VersionInfo
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.token_utils import TokenCounter
from llmaestro.llm.rate_limiter import RateLimiter, RateLimitConfig, SQLiteQuotaStorage


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--use-llm-tokens",
        action="store_true",
        default=False,
        help="Run tests that require LLM API tokens"
    )


@pytest.fixture
def llm_registry():
    """Load the model registry from YAML files."""
    registry = LLMRegistry()
    registry = LLMRegistry.from_yaml(Path("src/llm/models/claude.yaml"))
    print("Available models in registry:", list(registry._models.keys()))
    return registry


class YAMLFilePrompt(FilePrompt):
    """FilePrompt implementation that loads YAML files."""

    @classmethod
    async def load(cls, identifier: str) -> Optional["YAMLFilePrompt"]:
        """Load a prompt from a YAML file."""
        path = Path(identifier)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            # Add version information if not present
            if "current_version" not in data:
                version = VersionInfo(
                    number=data.get("version", "1.0.0"),
                    timestamp=datetime.now(),
                    author=data.get("author", "Test Author"),
                    description=data.get("description", ""),
                    change_type="initial"
                )
                data["current_version"] = version.model_dump()

            return cls(**data, file_path=path)
        except Exception as e:
            print(f"Error loading prompt from {path}: {e}")
            return None


@pytest.fixture
async def hello_world_prompt():
    """Load the hello world prompt from the yaml file."""
    loader = PromptLoader({"file": YAMLFilePrompt})
    prompt = await loader.load_prompt("file", "src/prompts/tasks/hello_world.yaml")
    assert prompt is not None
    return prompt


@pytest.fixture
def llm_config():
    """Load the LLM configuration from the config file."""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    llm_config = config_data["llm"]
    model_name = "claude-3-5-sonnet-latest"  # Use the same model as in config
    return AgentConfig(
        provider="anthropic",
        model_name=model_name,
        api_key=llm_config["api_key"],
        max_tokens=1024,
        temperature=0.7
    )


@pytest.fixture
async def anthropic_llm(llm_config, llm_registry):
    """Create an instance of the Anthropic LLM interface."""
    # Create a test instance with custom initialization
    llm = AnthropicLLM.__new__(AnthropicLLM)  # Create instance without calling __init__

    # Set up basic attributes
    llm.config = llm_config
    llm.context = ConversationContext([])
    llm._total_tokens = 0
    llm._token_counter = TokenCounter()

    # Set the registry before any other initialization
    llm._registry = llm_registry
    llm._model_descriptor = llm_registry.get_model(llm.config.model_name)

    if not llm._model_descriptor:
        raise ValueError(f"Could not find descriptor for model {llm.config.model_name}")

    # Initialize storage and rate limiter
    db_path = os.path.join("data", f"rate_limiter_{llm.config.provider}.db")
    os.makedirs("data", exist_ok=True)
    llm.storage = SQLiteQuotaStorage(db_path)

    # Initialize rate limiter if enabled
    if llm.config.rate_limit.enabled:
        llm.rate_limiter = RateLimiter(
            config=RateLimitConfig(
                requests_per_minute=llm.config.rate_limit.requests_per_minute,
                requests_per_hour=llm.config.rate_limit.requests_per_hour,
                max_daily_tokens=llm.config.rate_limit.max_daily_tokens,
                alert_threshold=llm.config.rate_limit.alert_threshold,
            ),
            storage=llm.storage,
        )
    else:
        llm.rate_limiter = None

    # Initialize Anthropic client
    llm.client = AnthropicLLM(api_key=llm.config.api_key)

    return llm


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    #"not config.getoption('--use-llm-tokens')",
    "config.getoption('--use-llm-tokens')",
    reason="Test requires --use-llm-tokens flag to run with real LLM"
)
async def test_hello_world_integration(hello_world_prompt, anthropic_llm):
    """
    Integration test for the hello world prompt using Claude LLM.

    This test:
    1. Loads the hello world prompt
    2. Sends it to Claude LLM
    3. Verifies the response matches the expected schema
    """
    # Arrange
    test_name = "Test User"
    variables = {"name": test_name}

    # Render the prompt
    system_prompt, user_prompt = hello_world_prompt.render(**variables)

    # Format messages for LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Act
    response = await anthropic_llm.process(
        input_data=messages,
        system_prompt=None  # System prompt is already included in messages
    )

    # Assert
    assert response is not None, "Response should not be None"
    assert response.content is not None, "Response content should not be None"

    # Parse response content as JSON
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError as e:
        pytest.fail(f"Failed to parse response as JSON: {response.content}")

    # Assert
    assert "message" in result, "Response should contain a message field"
    assert "timestamp" in result, "Response should contain a timestamp field"
    assert isinstance(result["message"], str), "Message should be a string"
    assert isinstance(result["timestamp"], str), "Timestamp should be a string"
    assert test_name in result["message"], f"Message should contain the name '{test_name}'"

    # Verify timestamp is in valid format
    try:
        datetime.fromisoformat(result["timestamp"].replace('Z', '+00:00'))
    except ValueError as e:
        pytest.fail(f"Invalid timestamp format: {e}")
