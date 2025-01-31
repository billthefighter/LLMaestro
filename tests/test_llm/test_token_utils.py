"""Tests for token counting utilities."""
import pytest
from typing import Dict, List
from unittest.mock import MagicMock

import tiktoken
import anthropic
from transformers import AutoTokenizer

from src.llm.token_utils import (
    BaseTokenizer,
    TiktokenTokenizer,
    AnthropicTokenizer,
    HuggingFaceTokenizer,
    TokenizerRegistry,
    TokenCounter,
)
from src.llm.models import ModelFamily


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
