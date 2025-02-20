from unittest.mock import AsyncMock, MagicMock, Mock, patch
import numpy as np
import pytest
from typing import List, Dict, Any, Set
import tiktoken
from PIL import Image
import io
import base64
from datetime import datetime
from anthropic import Anthropic
from anthropic.types import Message, MessageParam, TextBlock
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from llmaestro.core.models import AgentConfig, TokenUsage
from llmaestro.llm.interfaces.provider_interfaces.anthropic import AnthropicLLM
from llmaestro.llm.rate_limiter import RateLimitConfig
from llmaestro.llm.token_utils import TokenCounter, BaseTokenizer, TokenizerRegistry
from llmaestro.llm.models import ModelRegistry, ModelCapabilities, ModelDescriptor, ModelFamily, RangeConfig

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
def mock_model_registry_simple(monkeypatch):
    """Create a simple mock model registry using monkeypatch."""
    def mock_get_model(*args, **kwargs):
        return ModelDescriptor(
            name="test-model",
            family=ModelFamily.CLAUDE,
            capabilities=ModelCapabilities(
                supports_streaming=False,
                supports_vision=False,
                max_context_window=8192,
                max_output_tokens=1024,
                supported_media_types=set()
            ),
            min_api_version="2024-02-29",
            release_date=datetime(2024, 2, 29)
        )

    monkeypatch.setattr(ModelRegistry, "get_model", mock_get_model)
    return ModelRegistry()

@pytest.fixture
def mock_model_registry_full(monkeypatch):
    """Create a full mock model registry using monkeypatch."""
    def mock_get_model(*args, **kwargs):
        return ModelDescriptor(
            name="claude-3-sonnet-20240229",
            family=ModelFamily.CLAUDE,
            capabilities=ModelCapabilities(
                supports_streaming=True,
                supports_vision=True,
                max_context_window=200000,
                max_output_tokens=4096,
                typical_speed=100.0,
                input_cost_per_1k_tokens=0.015,
                output_cost_per_1k_tokens=0.015,
                daily_request_limit=150000,
                supports_json_mode=True,
                supports_system_prompt=True,
                supports_message_role=True,
                supports_tools=True,
                supported_media_types={"image/jpeg", "image/png"},
                temperature=RangeConfig(min_value=0.0, max_value=1.0, default_value=0.7),
                top_p=RangeConfig(min_value=0.0, max_value=1.0, default_value=1.0)
            ),
            min_api_version="2024-02-29",
            release_date=datetime(2024, 2, 29)
        )

    monkeypatch.setattr(ModelRegistry, "get_model", mock_get_model)
    return ModelRegistry()

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
def test_config():
    """Test configuration for AnthropicLLM."""
    return AgentConfig(
        provider="anthropic",
        model_name="claude-3-sonnet-20240229",
        api_key=TEST_API_KEY,
        max_tokens=1024,
        temperature=0.7,
        rate_limit={
            "requests_per_minute": 50,
            "requests_per_hour": 3500,
            "max_daily_tokens": 1000000,
            "alert_threshold": 0.8
        }
    )

@pytest.fixture
def anthropic_llm(test_config, mock_anthropic_client, mock_model_registry_full, mock_tokenizer, monkeypatch):
    """Create AnthropicLLM instance with mocked dependencies."""
    monkeypatch.setattr(TokenizerRegistry, "get_tokenizer", lambda *args: mock_tokenizer)
    return AnthropicLLM(config=test_config)

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img_array = np.zeros((64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    img_byte_arr = io.BytesIO()
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
