"""In-memory prompt implementation."""
from typing import Optional

from llmaestro.prompts.base import BasePrompt


class MemoryPrompt(BasePrompt):
    """A non-persistent prompt implementation that exists only in memory.

    This implementation is useful for temporary prompts or testing scenarios
    where persistence is not needed. The save() and load() methods are implemented
    as no-ops to satisfy the BasePrompt interface.
    """

    async def save(self) -> bool:
        """No-op save implementation.

        Returns:
            bool: Always returns True since no persistence is needed
        """
        return True

    @classmethod
    async def load(cls, identifier: str) -> Optional["MemoryPrompt"]:
        """No-op load implementation.

        Args:
            identifier: Unused identifier string

        Returns:
            Optional[MemoryPrompt]: Always returns None since this is a memory-only implementation
        """
        return None
