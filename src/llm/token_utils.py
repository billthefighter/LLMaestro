"""Unified token estimation utilities for LLM interfaces."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import tiktoken
import yaml

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    import anthropic
except ImportError:
    anthropic = None

from anthropic.types import MessageParam, TextBlockParam

from src.llm.models import ModelFamily, ModelRegistry

# Initialize and load model registry
_model_registry = ModelRegistry()
_models_dir = Path(__file__).parent / "models"
if _models_dir.exists():
    for model_file in _models_dir.glob("*.yaml"):
        try:
            loaded_registry = ModelRegistry.from_yaml(model_file)
            for model in loaded_registry._models.values():
                _model_registry.register(model)
        except Exception as e:
            print(f"Error loading model file {model_file}: {e}")


class BaseTokenizer(ABC):
    """Abstract base class for model-specific tokenizers."""

    @abstractmethod
    def __init__(self, model_name: Optional[str] = None):
        """Initialize tokenizer, optionally with a specific model name."""
        pass

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
    """Tokenizer for Anthropic models."""

    def __init__(self, model_name: Optional[str] = None):
        super().__init__(model_name)
        if not anthropic:
            raise ImportError("anthropic package is required for Anthropic models")
        # Load API key from config
        config_path = Path("config/config.yaml")
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        api_key = config_data["llm"]["api_key"]
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name or "claude-3-sonnet-20240229"

    def count_tokens(self, text: str) -> int:
        # Convert text to a properly typed message format
        message = MessageParam(role="user", content=[TextBlockParam(type="text", text=text)])
        result = self.client.messages.count_tokens(model=self.model_name, messages=[message])
        return result.input_tokens

    def encode(self, text: str) -> List[int]:
        # Anthropic doesn't expose token IDs directly
        raise NotImplementedError("Anthropic doesn't support token ID encoding")

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


class TokenizerRegistry:
    """Registry for model-specific tokenizers."""

    _tokenizers: Dict[ModelFamily, Type[BaseTokenizer]] = {
        ModelFamily.GPT: TiktokenTokenizer,
        ModelFamily.CLAUDE: AnthropicTokenizer,
        ModelFamily.HUGGINGFACE: HuggingFaceTokenizer,
    }

    @classmethod
    def register(cls, model_family: ModelFamily, tokenizer_cls: Type[BaseTokenizer]) -> None:
        """Register a new tokenizer for a model family."""
        if not issubclass(tokenizer_cls, BaseTokenizer):
            raise ValueError(f"{tokenizer_cls.__name__} must inherit from BaseTokenizer")
        if not tokenizer_cls.supports_model(model_family):
            raise ValueError(f"{tokenizer_cls.__name__} does not support {model_family}")
        cls._tokenizers[model_family] = tokenizer_cls

    @classmethod
    def get_tokenizer(cls, model_family: ModelFamily, model_name: str) -> BaseTokenizer:
        """Get the appropriate tokenizer for a model family."""
        tokenizer_cls = cls._tokenizers.get(model_family)
        if not tokenizer_cls:
            raise ValueError(f"No tokenizer registered for {model_family}")

        # Validate model exists
        if not _model_registry.get_model(model_name):
            raise ValueError(f"Unknown model {model_name} in family {model_family.name}")

        return tokenizer_cls(model_name)


class TokenCounter:
    """Unified token counting and estimation."""

    def __init__(self):
        self._tokenizers: Dict[str, BaseTokenizer] = {}

    def get_tokenizer(self, model_family: ModelFamily, model_name: str) -> BaseTokenizer:
        """Get or create a tokenizer for the specified model."""
        cache_key = f"{model_family.name}:{model_name}"
        if cache_key not in self._tokenizers:
            self._tokenizers[cache_key] = TokenizerRegistry.get_tokenizer(model_family, model_name)
        return self._tokenizers[cache_key]

    def count_tokens(self, text: str, model_family: ModelFamily, model_name: str) -> int:
        """Count tokens in text for a specific model."""
        return self.get_tokenizer(model_family, model_name).count_tokens(text)

    def estimate_messages(
        self, messages: List[Dict[str, str]], model_family: ModelFamily, model_name: str
    ) -> Dict[str, int]:
        """Estimate tokens for a list of messages."""
        # Validate model exists
        descriptor = _model_registry.get_model(model_name)
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
        descriptor = _model_registry.get_model(model_name)
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
        descriptor = _model_registry.get_model(model_name)
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
        descriptor = _model_registry.get_model(model_name)
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
