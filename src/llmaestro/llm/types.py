"""Type definitions for LLM interfaces."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmaestro.llm.models import LLMState
    from llmaestro.llm.interfaces.base import BaseLLMInterface
else:
    # These are just type hints, not actual imports
    LLMState = "LLMState"
    BaseLLMInterface = "BaseLLMInterface"
