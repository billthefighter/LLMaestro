"""Test fixtures for LLM-specific functionality."""
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Set
from unittest.mock import AsyncMock, MagicMock, Mock

import base64
import numpy as np
import pytest
import tiktoken
from PIL import Image
from anthropic import Anthropic
from anthropic.types import Message, MessageParam, TextBlock
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from llmaestro.config.agent import AgentTypeConfig, AgentRuntimeConfig
from llmaestro.llm.interfaces.provider_interfaces.anthropic import AnthropicLLM
from llmaestro.llm.rate_limiter import RateLimitConfig
from llmaestro.llm.models import LLMCapabilities, LLMProfile, ModelFamily, RangeConfig
from llmaestro.llm.token_utils import TokenCounter, BaseTokenizer, TokenizerRegistry

# Test constants
TEST_API_KEY = "test-api-key"
TEST_RESPONSE = {
    "id": "test_response_id",
    "content": [{"type": "text", "text": "Test response"}],
    "usage": {"input_tokens": 10, "output_tokens": 20}
}


@pytest.fixture
def test_response():
    """Test response data."""
    return TEST_RESPONSE


@pytest.fixture
def mock_tokenizer(monkeypatch):
    """Create a mock tokenizer using monkeypatch."""
    class TestTokenizer(BaseTokenizer):
        def count_tokens(self, text: str) -> int:
            return len(text.split())

        def encode(self, text: str) -> List[int]:
            return [1] * len(text.split())

        @classmethod
        def supports_model(cls, model_family: ModelFamily) -> bool:
            return True

    return TestTokenizer("test-model")


@pytest.fixture
def mock_anthropic_client(monkeypatch):
    """Create a mock Anthropic client using monkeypatch."""
    async def mock_create(*args, **kwargs):
        return TEST_RESPONSE

    def mock_count_tokens(*args, **kwargs):
        return 5

    mock_messages = MagicMock()
    mock_messages.create = AsyncMock(side_effect=mock_create)

    mock_client = MagicMock()
    mock_client.messages = mock_messages
    mock_client.count_tokens = mock_count_tokens

    monkeypatch.setattr(Anthropic, "__init__", lambda self, **kwargs: None)
    monkeypatch.setattr(Anthropic, "messages", mock_messages)
    return mock_client


@pytest.fixture
def test_agent_config(test_settings, mock_LLMProfile):
    """Test configuration for AnthropicLLM."""
    return AgentTypeConfig(
        provider=test_settings.test_provider,
        model=test_settings.test_model,
        max_tokens=1024,
        temperature=0.7,
        runtime=AgentRuntimeConfig(
            rate_limit=RateLimitConfig(
                requests_per_minute=50,
                requests_per_hour=3500,
                max_daily_tokens=1000000,
                alert_threshold=0.8
            )
        )
    )


@pytest.fixture
def test_anthropic_llm(test_agent_config, mock_anthropic_client, mock_LLMProfile, mock_tokenizer, monkeypatch):
    """Create AnthropicLLM instance with mocked dependencies."""
    monkeypatch.setattr(TokenizerRegistry, "get_tokenizer", lambda *args: mock_tokenizer)

    # Create a concrete implementation for testing
    class TestAnthropicLLM(AnthropicLLM):
        async def process_async(self, messages, **kwargs):
            return TEST_RESPONSE

    return TestAnthropicLLM(config=test_agent_config)


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img_array = np.zeros((64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

    return {
        "content": img_base64,
        "media_type": "image/png"
    }


@pytest.fixture
def mock_token_counter(monkeypatch):
    """Mock token counter using monkeypatch."""
    def mock_estimate(*args, **kwargs):
        return {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }

    monkeypatch.setattr(TokenCounter, "estimate_messages_with_images", mock_estimate)
    return None


@pytest.fixture
def sample_text() -> str:
    """Sample text for token counting."""
    return "Hello world! This is a test message."


@pytest.fixture
def sample_messages() -> List[Dict[str, str]]:
    """Sample messages for token estimation."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How are you?"}
    ]


@pytest.fixture
def mock_tiktoken(monkeypatch):
    """Mock tiktoken encoding using monkeypatch."""
    def mock_encode(text: str) -> List[int]:
        return [1, 2, 3, 4, 5]

    mock_encoding = MagicMock()
    mock_encoding.encode = mock_encode
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _: mock_encoding)
    return mock_encoding


@pytest.fixture
def mock_hf_tokenizer(monkeypatch):
    """Mock HuggingFace tokenizer using monkeypatch."""
    def mock_encode(text: str) -> List[int]:
        return [1, 2, 3, 4, 5]

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _: {"encode": mock_encode})
    return None


@pytest.fixture
def sample_image_data() -> List[Dict[str, int]]:
    """Sample image dimension data for token estimation."""
    return [
        {"width": 512, "height": 512},  # Small image
        {"width": 1024, "height": 1024},  # Large image
    ]
