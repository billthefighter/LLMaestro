"""Unified token estimation utilities for LLM interfaces."""
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING

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

if TYPE_CHECKING:
    from .models import ModelFamily, LLMInstance
    from .llm_registry import LLMRegistry
else:
    from .models import ModelFamily
    from .llm_registry import LLMRegistry

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
        return model_family == ModelFamily.OPENAI


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
        return model_family == ModelFamily.ANTHROPIC


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
        return model_family == ModelFamily.GOOGLE

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


def get_tokenizer_for_model(model_family: ModelFamily, model_name: str, api_key: Optional[str] = None) -> BaseTokenizer:
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


class TokenCounter:
    """Unified token counting and estimation."""

    def __init__(self, llm_registry: Optional[LLMRegistry] = None):
        """Initialize token counter.

        Args:
            llm_registry: Optional LLMRegistry instance. If not provided, a new instance will be created.
        """
        self._tokenizers: Dict[str, BaseTokenizer] = {}
        self._llm_registry = llm_registry or LLMRegistry()

    def get_tokenizer(self, model_family: ModelFamily, model_name: str) -> BaseTokenizer:
        """Get or create a tokenizer for the specified model."""
        cache_key = f"{model_family.name}:{model_name}"
        if cache_key not in self._tokenizers:
            # Get the LLM instance to access credentials
            instance = self._llm_registry.models.get(model_name)
            if not instance:
                raise ValueError(f"Unknown model {model_name} in family {model_family.name}")
            
            # Get credentials if needed for this model family
            api_key = None
            if model_family == ModelFamily.ANTHROPIC:
                if instance.credentials:
                    api_key = str(instance.credentials.key)
                if not api_key:
                    raise ValueError(f"API key required for {model_family.name} models")

            self._tokenizers[cache_key] = get_tokenizer_for_model(model_family, model_name, api_key)
        return self._tokenizers[cache_key]

    def count_tokens(self, text: str, model_family: ModelFamily, model_name: str) -> int:
        """Count tokens in text for a specific model."""
        return self.get_tokenizer(model_family, model_name).count_tokens(text)

    def estimate_messages(
        self, messages: List[Dict[str, str]], model_family: ModelFamily, model_name: str
    ) -> Dict[str, int]:
        """Estimate tokens for a list of messages."""
        # Get LLM instance to validate model exists
        instance = self._llm_registry.models.get(model_name)
        if not instance:
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
            if model_family == ModelFamily.OPENAI:
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
        # Get LLM instance for context limit
        instance = self._llm_registry.models.get(model_name)
        if not instance:
            return False, f"Unknown model {model_name} in family {model_family.name}"

        limit = instance.state.profile.capabilities.max_context_window
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
        """Estimate token usage for images based on model-specific formulas."""
        instance = self._llm_registry.models.get(model_name)
        if not instance or not instance.state.profile.vision_capabilities:
            return 0

        if model_family == ModelFamily.ANTHROPIC:
            # Claude's image token calculation (approximate)
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

        elif model_family == ModelFamily.OPENAI:
            # GPT-4V token calculation (approximate)
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
        instance = self._llm_registry.models.get(model_name)
        if not instance:
            return 0.0

        capabilities = instance.state.profile.capabilities

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
        if instance.state.profile.vision_capabilities:
            cost_per_image = instance.state.profile.vision_capabilities.cost_per_image or 0.0
            image_cost = image_count * cost_per_image

        return token_cost + image_cost
