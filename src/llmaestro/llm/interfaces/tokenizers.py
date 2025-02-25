"""Tokenizer implementations for different LLM providers."""
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import tiktoken

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError:
    anthropic = None
    AsyncAnthropic = None

from ..enums import ModelFamily


class BaseTokenizer(ABC):
    """Abstract base class for model-specific tokenizers."""

    def __init__(self, model_name: str):
        """Initialize tokenizer with a specific model name."""
        self.model_name = model_name

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in a list of messages."""
        total_tokens = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                # Handle content blocks
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        total_tokens += self.count_tokens(block["text"])
            else:
                total_tokens += self.count_tokens(str(content))
        return total_tokens


class TiktokenTokenizer(BaseTokenizer):
    """Tokenizer for OpenAI models using tiktoken."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        encoding_name = {
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-4": "cl100k_base",
        }.get(model_name, "cl100k_base")
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """OpenAI-specific message token counting."""
        total_tokens = super().count_messages(messages)
        # Add OpenAI's message overhead
        total_tokens += 4 * len(messages)  # 4 tokens per message for metadata
        return total_tokens


class AnthropicTokenizer(BaseTokenizer):
    """Token counter for Anthropic models."""

    def __init__(self, model_name: str, *, api_key: str):
        """Initialize the tokenizer.

        Args:
            model_name: Name of the model to use
            api_key: Anthropic API key
        """
        super().__init__(model_name)
        if AsyncAnthropic is None:
            raise ImportError("Anthropic package is not installed")
        self.client = AsyncAnthropic(api_key=api_key)

    def count_tokens(self, text: str) -> int:
        """Estimate token count for text using a simple approximation.

        This is a fallback method when the API token counting endpoint is not available.
        It uses a simple approximation of 4 characters per token.
        """
        # Simple approximation: ~4 characters per token
        return len(text) // 4 + 1

    def encode(self, text: str) -> List[int]:
        """Anthropic doesn't expose token IDs directly."""
        raise NotImplementedError("Anthropic doesn't expose token IDs")


class HuggingFaceTokenizer(BaseTokenizer):
    """Tokenizer for HuggingFace models."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        if not AutoTokenizer:
            raise ImportError("transformers package is required for HuggingFace models")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)


class SimpleWordTokenizer(BaseTokenizer):
    """Simple word-based tokenizer for models without specific tokenization."""

    def __init__(self, model_name: str):
        """Initialize the tokenizer."""
        super().__init__(model_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using simple word-based approach."""
        # Split on whitespace and punctuation
        words = re.findall(r"\w+|\S", text)
        # Rough estimate: 4 characters per token on average
        return sum(len(word) + 1 for word in words) // 4

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        raise NotImplementedError("SimpleWordTokenizer does not support encoding")


def get_tokenizer(model_family: ModelFamily, model_name: str, api_key: Optional[str] = None) -> BaseTokenizer:
    """Get the appropriate tokenizer for a model family.

    Args:
        model_family: The model family
        model_name: Name of the model
        api_key: Optional API key for providers that require it

    Returns:
        BaseTokenizer: The appropriate tokenizer instance

    Raises:
        ValueError: If no tokenizer is available for the model family
    """
    tokenizer_map = {
        ModelFamily.OPENAI: TiktokenTokenizer,
        ModelFamily.ANTHROPIC: AnthropicTokenizer,
        ModelFamily.HUGGINGFACE: HuggingFaceTokenizer,
        ModelFamily.GOOGLE: SimpleWordTokenizer,
    }

    tokenizer_class = tokenizer_map.get(model_family)
    if not tokenizer_class:
        raise ValueError(f"No tokenizer available for model family {model_family}")

    if model_family == ModelFamily.ANTHROPIC:
        if not api_key:
            raise ValueError("API key is required for Anthropic tokenizer")
        return tokenizer_class(model_name, api_key=api_key)

    return tokenizer_class(model_name)
