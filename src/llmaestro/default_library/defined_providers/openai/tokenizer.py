from typing import Any, Dict, List

from tiktoken import get_encoding

from llmaestro.llm.interfaces.tokenizers import BaseTokenizer


class TiktokenTokenizer(BaseTokenizer):
    """Tokenizer for OpenAI models using tiktoken."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        encoding_name = {
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-4": "cl100k_base",
        }.get(model_name, "cl100k_base")
        self.encoding = get_encoding(encoding_name)

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
