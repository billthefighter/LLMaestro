"""Unified token estimation utilities for LLM interfaces."""
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

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


from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.models import ModelFamily

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

    @classmethod
    @abstractmethod
    def supports_model(cls, model_family: ModelFamily) -> bool:
        """Check if this tokenizer supports the given model family."""
        pass


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

    @classmethod
    def supports_model(cls, model_family: ModelFamily) -> bool:
        return model_family == ModelFamily.GPT


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

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count for messages."""
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

    def encode(self, text: str) -> List[int]:
        """Anthropic doesn't expose token IDs directly."""
        raise NotImplementedError("Anthropic doesn't expose token IDs")

    @classmethod
    def supports_model(cls, model_family: ModelFamily) -> bool:
        return model_family == ModelFamily.CLAUDE


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

    @classmethod
    def supports_model(cls, model_family: ModelFamily) -> bool:
        return model_family == ModelFamily.HUGGINGFACE


class SimpleWordTokenizer(BaseTokenizer):
    """Simple word-based tokenizer for models without specific tokenization."""

    def __init__(self, model_name: str):
        """Initialize the tokenizer."""
        super().__init__(model_name)

    @classmethod
    def supports_model(cls, model_family: ModelFamily) -> bool:
        """Check if this tokenizer supports the given model family."""
        return model_family == ModelFamily.GEMINI

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using simple word-based approach."""
        # Split on whitespace and punctuation
        words = re.findall(r"\w+|\S", text)
        # Rough estimate: 4 characters per token on average
        return sum(len(word) + 1 for word in words) // 4

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        # Not implemented for simple tokenizer
        raise NotImplementedError("SimpleWordTokenizer does not support encoding")

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs into text."""
        # Not implemented for simple tokenizer
        raise NotImplementedError("SimpleWordTokenizer does not support decoding")


class TokenizerRegistry:
    """Registry of tokenizers for different model families."""

    _tokenizers: Dict[ModelFamily, Type[BaseTokenizer]] = {}

    @classmethod
    def register(cls, model_family: ModelFamily, tokenizer_class: Type[BaseTokenizer]) -> None:
        """Register a tokenizer for a model family."""
        cls._tokenizers[model_family] = tokenizer_class

    @classmethod
    def get_tokenizer(cls, model_family: ModelFamily, model_name: str, api_key: Optional[str] = None) -> BaseTokenizer:
        """Get a tokenizer for a model family."""
        if model_family not in cls._tokenizers:
            raise ValueError(f"No tokenizer registered for {model_family}")

        tokenizer_class = cls._tokenizers[model_family]

        # Create tokenizer instance based on model family
        if model_family == ModelFamily.CLAUDE:
            if not api_key:
                raise ValueError("API key is required for Anthropic tokenizer")
            return AnthropicTokenizer(model_name, api_key=api_key)
        else:
            # Other tokenizers only need model_name
            return tokenizer_class(model_name)


# Register tokenizers
TokenizerRegistry.register(ModelFamily.GPT, TiktokenTokenizer)
TokenizerRegistry.register(ModelFamily.CLAUDE, AnthropicTokenizer)
TokenizerRegistry.register(ModelFamily.HUGGINGFACE, HuggingFaceTokenizer)
TokenizerRegistry.register(ModelFamily.GEMINI, SimpleWordTokenizer)


class TokenCounter:
    """Unified token counting and estimation."""

    def __init__(self, api_key: Optional[str] = None, llm_registry: Optional[LLMRegistry] = None):
        """Initialize token counter.

        Args:
            api_key: Optional API key for providers that require it (e.g. Anthropic)
            llm_registry: Optional LLMRegistry instance. If not provided, a new instance will be created.
        """
        self._tokenizers: Dict[str, BaseTokenizer] = {}
        self._api_key = api_key
        self._llm_registry = llm_registry or LLMRegistry()

    def get_tokenizer(self, model_family: ModelFamily, model_name: str) -> BaseTokenizer:
        """Get or create a tokenizer for the specified model."""
        cache_key = f"{model_family.name}:{model_name}"
        if cache_key not in self._tokenizers:
            self._tokenizers[cache_key] = TokenizerRegistry.get_tokenizer(model_family, model_name, self._api_key)
        return self._tokenizers[cache_key]

    def count_tokens(self, text: str, model_family: ModelFamily, model_name: str) -> int:
        """Count tokens in text for a specific model."""
        return self.get_tokenizer(model_family, model_name).count_tokens(text)

    def estimate_messages(
        self, messages: List[Dict[str, str]], model_family: ModelFamily, model_name: str
    ) -> Dict[str, int]:
        """Estimate tokens for a list of messages."""
        # Validate model exists
        descriptor = self._llm_registry.get_model(model_name)
        if not descriptor:
            raise ValueError(f"Unknown model {model_name} in family {model_family.name}")

        tokenizer = self.get_tokenizer(model_family, model_name)

        # Count tokens in each message
        total_tokens = 0
        prompt_tokens = 0

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")

            # Count content tokens
            content_tokens = tokenizer.count_tokens(content)
            total_tokens += content_tokens

            # Add role overhead for certain models
            if model_family == ModelFamily.GPT:
                total_tokens += 4  # OpenAI's token overhead per message

            if role != "assistant":
                prompt_tokens += content_tokens

        return {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "estimated_completion_tokens": total_tokens - prompt_tokens,
        }

    def validate_context(
        self, total_tokens: int, max_completion_tokens: int, model_family: ModelFamily, model_name: str
    ) -> Tuple[bool, Optional[str]]:
        """Check if total tokens fit within model's context window."""
        # Get model descriptor for context limit
        descriptor = self._llm_registry.get_model(model_name)
        if not descriptor:
            return False, f"Unknown model {model_name} in family {model_family.name}"

        limit = descriptor.capabilities.max_context_window
        total_needed = total_tokens + max_completion_tokens

        if total_needed > limit:
            return False, (
                f"Would exceed context limit. "
                f"Needs {total_needed} tokens, limit is {limit}. "
                f"(prompt: {total_tokens}, completion: {max_completion_tokens})"
            )

        return True, None

    def estimate_image_tokens(
        self,
        image_sizes: List[int],  # sizes in pixels (width * height)
        model_family: ModelFamily,
        model_name: str,
    ) -> int:
        """Estimate token usage for images based on model-specific formulas.

        Args:
            image_sizes: List of image sizes in total pixels (width * height)
            model_family: The model family (e.g., CLAUDE, GPT)
            model_name: Specific model name

        Returns:
            Estimated token count for the images
        """
        descriptor = self._llm_registry.get_model(model_name)
        if not descriptor or not descriptor.capabilities.supports_vision:
            return 0

        if model_family == ModelFamily.CLAUDE:
            # Claude's image token calculation (approximate)
            # Based on Anthropic's documentation and testing
            total_tokens = 0
            for size in image_sizes:
                # Base cost for image header
                tokens = 170
                # Add scaled cost based on image size
                pixels = size
                if pixels <= 512 * 512:
                    tokens += int(pixels / (32 * 32))  # Small images
                else:
                    tokens += int(pixels / (16 * 16))  # Larger images
                total_tokens += tokens
            return total_tokens

        elif model_family == ModelFamily.GPT:
            # GPT-4V token calculation (approximate)
            # Based on OpenAI's documentation
            total_tokens = 0
            for size in image_sizes:
                tokens = 85  # Base cost
                pixels = size
                if pixels <= 512 * 512:
                    tokens += int(pixels / (64 * 64))
                else:
                    tokens += int(pixels / (32 * 32))
                total_tokens += tokens
            return total_tokens

        return 0

    def estimate_messages_with_images(
        self,
        messages: List[Dict[str, str]],
        image_data: List[Dict[str, Any]],  # List of {width: int, height: int} dicts
        model_family: ModelFamily,
        model_name: str,
    ) -> Dict[str, int]:
        """Estimate tokens for messages including images."""
        # Get base text token count
        text_estimates = self.estimate_messages(messages, model_family, model_name)

        # Calculate image tokens
        image_sizes = [img["width"] * img["height"] for img in image_data]
        image_tokens = self.estimate_image_tokens(image_sizes, model_family, model_name)

        return {
            "total_tokens": text_estimates["total_tokens"] + image_tokens,
            "prompt_tokens": text_estimates["prompt_tokens"] + image_tokens,
            "estimated_completion_tokens": text_estimates["estimated_completion_tokens"],
            "image_tokens": image_tokens,
        }

    def estimate_cost(
        self, token_counts: Dict[str, int], image_count: int, model_family: ModelFamily, model_name: str
    ) -> float:
        """Calculate total cost including both tokens and images."""
        descriptor = self._llm_registry.get_model(model_name)
        if not descriptor:
            return 0.0

        capabilities = descriptor.capabilities

        # Calculate token costs
        token_cost = 0.0
        if capabilities.input_cost_per_1k_tokens is not None:
            token_cost += (token_counts["prompt_tokens"] / 1000.0) * capabilities.input_cost_per_1k_tokens
        if capabilities.output_cost_per_1k_tokens is not None:
            token_cost += (
                token_counts["estimated_completion_tokens"] / 1000.0
            ) * capabilities.output_cost_per_1k_tokens
        elif capabilities.cost_per_1k_tokens > 0:
            token_cost += (token_counts["total_tokens"] / 1000.0) * capabilities.cost_per_1k_tokens

        # Calculate image costs
        image_cost = 0.0
        if capabilities.supports_vision and capabilities.vision_config:
            cost_per_image = capabilities.vision_config.get("cost_per_image", 0.0)
            image_cost = image_count * cost_per_image

        return token_cost + image_cost
