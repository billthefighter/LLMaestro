"""Tests for Anthropic LLM interface."""
import pytest
from typing import Dict, List
import base64
import io
from PIL import Image
import numpy as np
from datetime import datetime

from anthropic import Anthropic
from anthropic.types import Message, MessageParam, TextBlock

from src.llm.interfaces.anthropic import AnthropicLLM
from src.core.models import AgentConfig, TokenUsage, RateLimitConfig
from src.llm.models import ModelFamily, ModelRegistry, ModelDescriptor, ModelCapabilities, RangeConfig
from src.llm.token_utils import TokenCounter, TokenizerRegistry, BaseTokenizer
from src.llm.rate_limiter import RateLimiter, SQLiteQuotaStorage
from src.llm.interfaces.base import ImageInput

# Test Data
@pytest.fixture
def test_response():
    """Test response data."""
    return {
        "id": "test_response_id",
        "content": [
            {"type": "text", "text": "Test response"}
        ]
    }

@pytest.fixture
def mock_model_registry():
    """Create a mock model registry with our test model."""
    registry = ModelRegistry()

    # Create test model descriptor
    capabilities = ModelCapabilities(
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
        supported_mime_types={"image/jpeg", "image/png"},
        temperature=RangeConfig(min_value=0.0, max_value=1.0, default_value=0.7),
        top_p=RangeConfig(min_value=0.0, max_value=1.0, default_value=1.0)
    )

    descriptor = ModelDescriptor(
        name="claude-3-sonnet-20240229",
        family=ModelFamily.CLAUDE,
        capabilities=capabilities,
        min_api_version="2024-02-29",
        release_date=datetime(2024, 2, 29)
    )

    registry.register(descriptor)
    return registry

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    class MockTokenizer(BaseTokenizer):
        def __init__(self, model_name: str):
            super().__init__(model_name)
            self.model_name = model_name

        def count_tokens(self, text: str) -> int:
            return len(text.split())  # Simple word-based counting

        def encode(self, text: str) -> List[int]:
            return [1] * len(text.split())

        @classmethod
        def supports_model(cls, model_family: ModelFamily) -> bool:
            return True

    return MockTokenizer

class Usage:
    def __init__(self):
        self.input_tokens = 10
        self.output_tokens = 20

class MockResponse:
    def __init__(self):
        self.id = "test_response_id"
        self.content = [TextBlock(text="Test response", type="text")]
        self.usage = Usage()

class MockMessages:
    async def create(self, **kwargs):
        return MockResponse()

class MockClient:
    def __init__(self, api_key):
        self.messages = MockMessages()

@pytest.fixture
def mock_anthropic_client(monkeypatch):
    """Mock Anthropic client using monkeypatch."""
    def mock_init(self, api_key):
        self.messages = MockClient(api_key).messages

    monkeypatch.setattr(Anthropic, "__init__", mock_init)
    return MockClient("test_api_key")

@pytest.fixture
def test_config():
    """Test configuration for AnthropicLLM."""
    return AgentConfig(
        provider="anthropic",
        model_name="claude-3-sonnet-20240229",  # Updated to match registry
        api_key="test_api_key",
        max_tokens=1024,
        temperature=0.7,
        rate_limit=RateLimitConfig(
            requests_per_minute=50,
            requests_per_hour=3500,
            max_daily_tokens=1000000,
            alert_threshold=0.8
        )
    )

@pytest.fixture
def anthropic_llm(test_config, mock_anthropic_client, mock_model_registry, mock_tokenizer, monkeypatch):
    """Create AnthropicLLM instance with mocked dependencies."""
    def mock_registry_init(self):
        self._models = mock_model_registry._models

    def mock_get_tokenizer(model_family: ModelFamily, model_name: str) -> BaseTokenizer:
        return mock_tokenizer(model_name)

    monkeypatch.setattr(ModelRegistry, "__init__", mock_registry_init)
    monkeypatch.setattr(TokenizerRegistry, "get_tokenizer", mock_get_tokenizer)
    return AnthropicLLM(config=test_config)

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a small test image
    img_array = np.zeros((64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Convert to base64
    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

    return ImageInput(
        content=img_base64,
        mime_type="image/png"
    )

@pytest.fixture
def mock_token_counter(monkeypatch):
    """Mock token counter."""
    def mock_estimate(*args, **kwargs):
        return {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }

    monkeypatch.setattr(TokenCounter, "estimate_messages_with_images", mock_estimate)
    return None

# Unit Tests
@pytest.mark.unit
class TestAnthropicLLM:
    """Unit tests for AnthropicLLM class."""

    def test_init(self, test_config, mock_model_registry, mock_tokenizer, monkeypatch):
        """Test initialization."""
        def mock_registry_init(self):
            self._models = mock_model_registry._models

        def mock_get_tokenizer(model_family: ModelFamily, model_name: str) -> BaseTokenizer:
            return mock_tokenizer(model_name)

        monkeypatch.setattr(ModelRegistry, "__init__", mock_registry_init)
        monkeypatch.setattr(TokenizerRegistry, "get_tokenizer", mock_get_tokenizer)
        llm = AnthropicLLM(config=test_config)
        assert llm.config == test_config
        assert llm.model_family == ModelFamily.CLAUDE
        assert isinstance(llm._token_counter, TokenCounter)

    def test_get_image_dimensions(self, anthropic_llm, sample_image):
        """Test image dimension extraction."""
        dimensions = anthropic_llm._get_image_dimensions(sample_image)
        assert dimensions["width"] == 64
        assert dimensions["height"] == 64

    def test_create_message_param(self, anthropic_llm):
        """Test message parameter creation."""
        content_blocks = [
            {"type": "text", "text": "Test message"}
        ]
        message = anthropic_llm._create_message_param(
            role="user",
            content_blocks=content_blocks
        )
        assert isinstance(message, dict)
        assert message["role"] == "user"
        assert message["content"] == content_blocks

    def test_format_messages(self, anthropic_llm):
        """Test message formatting."""
        input_messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"}
        ]
        formatted = anthropic_llm._format_messages(input_messages)
        assert len(formatted) == 2
        assert formatted[0]["role"] == "system"
        assert formatted[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_rate_limit_check(self, anthropic_llm, monkeypatch):
        """Test rate limit checking."""
        async def mock_check_and_update(*args, **kwargs):
            return True, None

        monkeypatch.setattr(RateLimiter, "check_and_update", mock_check_and_update)

        messages = [{"role": "user", "content": "Test"}]
        can_proceed, _ = await anthropic_llm._check_rate_limits(messages)
        assert can_proceed is True

# Integration Tests
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    "not config.getoption('--use-llm-tokens')",
    reason="Test requires --use-llm-tokens flag to run with real LLM"
)
class TestAnthropicLLMIntegration:
    """Integration tests for AnthropicLLM class."""

    async def test_simple_completion(self, anthropic_llm, mock_token_counter):
        """Test basic completion without images."""
        messages = [
            {"role": "user", "content": "Say hello!"}
        ]

        response = await anthropic_llm.process(
            input_data=messages
        )

        assert response.content is not None
        assert isinstance(response.content, str)
        assert response.metadata["id"] == "test_response_id"
        assert response.token_usage is not None
        assert response.token_usage.prompt_tokens == 10
        assert response.token_usage.completion_tokens == 20

    async def test_completion_with_system_prompt(self, anthropic_llm, mock_token_counter):
        """Test completion with system prompt."""
        messages = [
            {"role": "user", "content": "What's your role?"}
        ]
        system_prompt = "You are a helpful assistant."

        response = await anthropic_llm.process(
            input_data=messages,
            system_prompt=system_prompt
        )

        assert response.content is not None
        assert isinstance(response.content, str)
        assert response.metadata["id"] == "test_response_id"
        assert response.token_usage is not None
        assert response.token_usage.prompt_tokens == 10
        assert response.token_usage.completion_tokens == 20

    async def test_completion_with_image(self, anthropic_llm, sample_image, mock_token_counter):
        """Test completion with image input."""
        messages = [
            {"role": "user", "content": "What's in this image?"}
        ]

        response = await anthropic_llm.process(
            input_data=messages,
            images=[sample_image]
        )

        assert response.content is not None
        assert isinstance(response.content, str)
        assert response.metadata["id"] == "test_response_id"
        assert response.token_usage is not None
        assert response.token_usage.prompt_tokens == 10
        assert response.token_usage.completion_tokens == 20

    async def test_rate_limiting(self, anthropic_llm, mock_token_counter):
        """Test rate limiting functionality."""
        messages = [
            {"role": "user", "content": "Test message"}
        ]

        # Make multiple requests to test rate limiting
        responses = []
        for _ in range(3):
            response = await anthropic_llm.process(input_data=messages)
            responses.append(response)

        assert all(r.content is not None for r in responses)
        assert all(r.metadata["id"] == "test_response_id" for r in responses)
        assert all(r.token_usage is not None for r in responses)
        assert all(r.token_usage.prompt_tokens == 10 for r in responses)
        assert all(r.token_usage.completion_tokens == 20 for r in responses)
