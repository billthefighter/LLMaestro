"""Tests for token counting utilities."""
import pytest
from typing import Dict, List
from unittest.mock import MagicMock

import tiktoken
import anthropic
from transformers import AutoTokenizer

from llmaestro.llm.token_utils import (
    BaseTokenizer,
    TiktokenTokenizer,
    AnthropicTokenizer,
    HuggingFaceTokenizer,
    TokenizerRegistry,
    TokenCounter,
)
from llmaestro.llm.models import ModelFamily, ModelCapabilities, ModelDescriptor


# Fixtures
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
    """Mock tiktoken encoding."""
    mock_encoding = MagicMock()
    mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _: mock_encoding)
    return mock_encoding

@pytest.fixture
def mock_anthropic(monkeypatch):
    """Mock Anthropic client."""
    mock_client = MagicMock()
    mock_client.count_tokens.return_value = 5
    monkeypatch.setattr(anthropic, "Anthropic", lambda: mock_client)
    return mock_client

@pytest.fixture
def mock_hf_tokenizer(monkeypatch):
    """Mock HuggingFace tokenizer."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _: mock_tokenizer)
    return mock_tokenizer

@pytest.fixture
def sample_image_data() -> List[Dict[str, int]]:
    """Sample image dimension data for token estimation."""
    return [
        {"width": 512, "height": 512},  # Small image
        {"width": 1024, "height": 1024},  # Large image
    ]

@pytest.fixture
def mock_model_registry(monkeypatch):
    """Mock model registry with vision capabilities."""
    mock_descriptor = MagicMock()
    mock_descriptor.capabilities = ModelCapabilities(
        supports_vision=True,
        vision_config={
            "max_images_per_request": 2,
            "supported_formats": ["png", "jpeg"],
            "max_image_size_mb": 20,
            "cost_per_image": 0.002,
        },
        input_cost_per_1k_tokens=0.01,
        output_cost_per_1k_tokens=0.03,
        max_context_window=100000,
    )

    def mock_get_model(*args):
        return mock_descriptor

    monkeypatch.setattr("llmaestro.llm.token_utils._model_registry.get_model", mock_get_model)
    return mock_descriptor


# Base Tokenizer Tests
@pytest.mark.unit
class TestBaseTokenizer:
    """Tests for BaseTokenizer abstract class."""

    def test_base_tokenizer_is_abstract(self):
        """Verify BaseTokenizer cannot be instantiated."""
        with pytest.raises(TypeError) as exc_info:
            BaseTokenizer()  # type: ignore
        assert "abstract" in str(exc_info.value).lower()


# Tiktoken Tokenizer Tests
@pytest.mark.unit
class TestTiktokenTokenizer:
    """Tests for TiktokenTokenizer implementation."""

    def test_init_with_known_model(self, mock_tiktoken):
        """Test initialization with known model name."""
        tokenizer = TiktokenTokenizer("gpt-3.5-turbo")
        assert tokenizer.encoding is not None

    def test_count_tokens(self, mock_tiktoken, sample_text):
        """Test token counting."""
        tokenizer = TiktokenTokenizer("gpt-3.5-turbo")
        count = tokenizer.count_tokens(sample_text)
        assert count == 5
        mock_tiktoken.encode.assert_called_once_with(sample_text)

    def test_encode(self, mock_tiktoken, sample_text):
        """Test text encoding."""
        tokenizer = TiktokenTokenizer("gpt-3.5-turbo")
        tokens = tokenizer.encode(sample_text)
        assert len(tokens) == 5
        mock_tiktoken.encode.assert_called_once_with(sample_text)

    def test_supports_model(self):
        """Test model family support."""
        assert TiktokenTokenizer.supports_model(ModelFamily.GPT)
        assert not TiktokenTokenizer.supports_model(ModelFamily.CLAUDE)


# Anthropic Tokenizer Tests
@pytest.mark.unit
class TestAnthropicTokenizer:
    """Tests for AnthropicTokenizer implementation."""

    def test_init(self, mock_anthropic):
        """Test initialization."""
        tokenizer = AnthropicTokenizer()
        assert tokenizer.client is not None

    def test_count_tokens(self, mock_anthropic, sample_text):
        """Test token counting."""
        tokenizer = AnthropicTokenizer()
        count = tokenizer.count_tokens(sample_text)
        assert count == 5
        mock_anthropic.count_tokens.assert_called_once_with(sample_text)

    def test_encode_raises_not_implemented(self, mock_anthropic, sample_text):
        """Test encode method raises NotImplementedError."""
        tokenizer = AnthropicTokenizer()
        with pytest.raises(NotImplementedError):
            tokenizer.encode(sample_text)

    def test_supports_model(self):
        """Test model family support."""
        assert AnthropicTokenizer.supports_model(ModelFamily.CLAUDE)
        assert not AnthropicTokenizer.supports_model(ModelFamily.GPT)


# HuggingFace Tokenizer Tests
@pytest.mark.unit
class TestHuggingFaceTokenizer:
    """Tests for HuggingFaceTokenizer implementation."""

    def test_init(self, mock_hf_tokenizer):
        """Test initialization."""
        tokenizer = HuggingFaceTokenizer("bert-base-uncased")
        assert tokenizer.tokenizer is not None

    def test_count_tokens(self, mock_hf_tokenizer, sample_text):
        """Test token counting."""
        tokenizer = HuggingFaceTokenizer("bert-base-uncased")
        count = tokenizer.count_tokens(sample_text)
        assert count == 5
        mock_hf_tokenizer.encode.assert_called_once_with(sample_text)

    def test_encode(self, mock_hf_tokenizer, sample_text):
        """Test text encoding."""
        tokenizer = HuggingFaceTokenizer("bert-base-uncased")
        tokens = tokenizer.encode(sample_text)
        assert len(tokens) == 5
        mock_hf_tokenizer.encode.assert_called_once_with(sample_text)

    def test_supports_model(self):
        """Test model family support."""
        assert HuggingFaceTokenizer.supports_model(ModelFamily.HUGGINGFACE)
        assert not HuggingFaceTokenizer.supports_model(ModelFamily.GPT)


# TokenizerRegistry Tests
@pytest.mark.unit
class TestTokenizerRegistry:
    """Tests for TokenizerRegistry."""

    def test_register_valid_tokenizer(self):
        """Test registering a valid tokenizer."""
        class CustomTokenizer(BaseTokenizer):
            def count_tokens(self, text: str) -> int:
                return 0
            def encode(self, text: str) -> List[int]:
                return []
            @classmethod
            def supports_model(cls, model_family: ModelFamily) -> bool:
                return model_family == ModelFamily.GPT

        TokenizerRegistry.register(ModelFamily.GPT, CustomTokenizer)
        assert TokenizerRegistry._tokenizers[ModelFamily.GPT] == CustomTokenizer

    def test_register_invalid_tokenizer(self):
        """Test registering an invalid tokenizer."""
        class InvalidTokenizer:  # type: ignore
            pass

        with pytest.raises(ValueError):
            TokenizerRegistry.register(ModelFamily.GPT, InvalidTokenizer)  # type: ignore

    def test_get_tokenizer_unknown_model(self):
        """Test getting tokenizer for unknown model."""
        with pytest.raises(ValueError):
            TokenizerRegistry.get_tokenizer(ModelFamily.GPT, "unknown-model")


# TokenCounter Tests
@pytest.mark.unit
class TestTokenCounter:
    """Tests for TokenCounter."""

    def test_init(self):
        """Test initialization."""
        counter = TokenCounter()
        assert counter._tokenizers == {}

    def test_get_tokenizer_caching(self, mock_tiktoken):
        """Test tokenizer caching."""
        counter = TokenCounter()
        tokenizer1 = counter.get_tokenizer(ModelFamily.GPT, "gpt-3.5-turbo")
        tokenizer2 = counter.get_tokenizer(ModelFamily.GPT, "gpt-3.5-turbo")
        assert tokenizer1 is tokenizer2

    @pytest.mark.parametrize("model_family,model_name", [
        (ModelFamily.GPT, "gpt-3.5-turbo"),
        (ModelFamily.CLAUDE, "claude-2"),
    ])
    def test_count_tokens(
        self,
        model_family: ModelFamily,
        model_name: str,
        sample_text: str,
        mock_tiktoken,
        mock_anthropic
    ):
        """Test token counting for different models."""
        counter = TokenCounter()
        count = counter.count_tokens(sample_text, model_family, model_name)
        assert count == 5

    def test_estimate_messages(
        self,
        sample_messages: List[Dict[str, str]],
        mock_tiktoken
    ):
        """Test message token estimation."""
        counter = TokenCounter()
        result = counter.estimate_messages(
            sample_messages,
            ModelFamily.GPT,
            "gpt-3.5-turbo"
        )
        assert "total_tokens" in result
        assert "prompt_tokens" in result
        assert "estimated_completion_tokens" in result

    def test_validate_context_within_limit(self, mock_tiktoken):
        """Test context validation within limits."""
        counter = TokenCounter()
        is_valid, error = counter.validate_context(
            100,  # total tokens
            100,  # max completion tokens
            ModelFamily.GPT,
            "gpt-3.5-turbo"
        )
        assert is_valid
        assert not error

    def test_validate_context_exceeds_limit(self, mock_tiktoken):
        """Test context validation exceeding limits."""
        counter = TokenCounter()
        is_valid, error = counter.validate_context(
            5000,  # total tokens
            5000,  # max completion tokens
            ModelFamily.GPT,
            "gpt-3.5-turbo"
        )
        assert not is_valid
        assert error is not None
        assert "exceed context limit" in error.lower()


# Image Token Counting Tests
@pytest.mark.unit
class TestImageTokenCounting:
    """Tests for image token counting functionality."""

    def test_estimate_image_tokens_claude(self, mock_model_registry):
        """Test image token estimation for Claude models."""
        counter = TokenCounter()
        image_sizes = [512 * 512, 1024 * 1024]  # One small, one large image

        tokens = counter.estimate_image_tokens(
            image_sizes=image_sizes,
            model_family=ModelFamily.CLAUDE,
            model_name="claude-3-sonnet"
        )

        # Verify token calculation
        # Small image: 170 (base) + (512*512)/(32*32) = 170 + 256 = 426
        # Large image: 170 (base) + (1024*1024)/(16*16) = 170 + 4096 = 4266
        expected_tokens = 426 + 4266
        assert tokens == expected_tokens

    def test_estimate_image_tokens_gpt4v(self, mock_model_registry):
        """Test image token estimation for GPT-4V models."""
        counter = TokenCounter()
        image_sizes = [512 * 512, 1024 * 1024]  # One small, one large image

        tokens = counter.estimate_image_tokens(
            image_sizes=image_sizes,
            model_family=ModelFamily.GPT,
            model_name="gpt-4-vision"
        )

        # Verify token calculation
        # Small image: 85 (base) + (512*512)/(64*64) = 85 + 64 = 149
        # Large image: 85 (base) + (1024*1024)/(32*32) = 85 + 1024 = 1109
        expected_tokens = 149 + 1109
        assert tokens == expected_tokens

    def test_estimate_image_tokens_unsupported_model(self, mock_model_registry):
        """Test image token estimation for models without vision support."""
        counter = TokenCounter()
        # Override mock to disable vision support
        mock_model_registry.capabilities.supports_vision = False

        tokens = counter.estimate_image_tokens(
            image_sizes=[512 * 512],
            model_family=ModelFamily.HUGGINGFACE,
            model_name="bert-base"
        )

        assert tokens == 0

    @pytest.mark.parametrize("model_family,model_name,expected_multiplier", [
        (ModelFamily.CLAUDE, "claude-3-sonnet", 1/(32*32)),  # Small image multiplier for Claude
        (ModelFamily.GPT, "gpt-4-vision", 1/(64*64)),  # Small image multiplier for GPT
    ])
    def test_image_size_scaling(
        self,
        mock_model_registry,
        model_family: ModelFamily,
        model_name: str,
        expected_multiplier: float
    ):
        """Test token scaling based on image size."""
        counter = TokenCounter()
        small_image_size = 256 * 256  # Definitely small

        tokens = counter.estimate_image_tokens(
            image_sizes=[small_image_size],
            model_family=model_family,
            model_name=model_name
        )

        # Verify base cost and scaling
        base_cost = 170 if model_family == ModelFamily.CLAUDE else 85
        expected_tokens = base_cost + int(small_image_size * expected_multiplier)
        assert tokens == expected_tokens


# Combined Text and Image Tests
@pytest.mark.unit
class TestCombinedTokenCounting:
    """Tests for combined text and image token counting."""

    def test_estimate_messages_with_images(
        self,
        mock_model_registry,
        sample_messages: List[Dict[str, str]],
        sample_image_data: List[Dict[str, int]]
    ):
        """Test token estimation for messages with images."""
        counter = TokenCounter()

        result = counter.estimate_messages_with_images(
            messages=sample_messages,
            image_data=sample_image_data,
            model_family=ModelFamily.CLAUDE,
            model_name="claude-3-sonnet"
        )

        assert "total_tokens" in result
        assert "image_tokens" in result
        assert result["image_tokens"] > 0
        assert result["total_tokens"] > result["image_tokens"]

    def test_estimate_cost(
        self,
        mock_model_registry,
        sample_messages: List[Dict[str, str]],
        sample_image_data: List[Dict[str, int]]
    ):
        """Test cost estimation including both tokens and images."""
        counter = TokenCounter()

        # First get token counts
        token_counts = counter.estimate_messages_with_images(
            messages=sample_messages,
            image_data=sample_image_data,
            model_family=ModelFamily.CLAUDE,
            model_name="claude-3-sonnet"
        )

        # Then calculate cost
        cost = counter.estimate_cost(
            token_counts=token_counts,
            image_count=len(sample_image_data),
            model_family=ModelFamily.CLAUDE,
            model_name="claude-3-sonnet"
        )

        # Verify cost calculation
        expected_token_cost = (
            (token_counts["prompt_tokens"] / 1000.0) * 0.01 +  # Input tokens
            (token_counts["estimated_completion_tokens"] / 1000.0) * 0.03  # Output tokens
        )
        expected_image_cost = len(sample_image_data) * 0.002  # Image cost
        expected_total = expected_token_cost + expected_image_cost

        assert cost == pytest.approx(expected_total, rel=1e-6)

    def test_zero_cost_for_unknown_model(self, mock_model_registry):
        """Test cost estimation returns zero for unknown models."""
        counter = TokenCounter()
        mock_model_registry.return_value = None  # Simulate unknown model

        cost = counter.estimate_cost(
            token_counts={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            image_count=1,
            model_family=ModelFamily.CLAUDE,
            model_name="unknown-model"
        )

        assert cost == 0.0
