from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.types import PromptMetadata, ResponseFormat, VersionInfo
from llmaestro.session.session import Session
import pytest
from datetime import datetime


class TestPrompt(BasePrompt):
    """Test prompt class that implements abstract methods."""

    def __init__(self, **data):
        super().__init__(**data)

    def render(self, **kwargs) -> tuple[str, str, list]:
        """Mock render implementation."""
        return self.system_prompt, self.user_prompt, []

    async def load(self) -> None:
        """Mock load implementation."""
        pass

    async def save(self) -> None:
        """Mock save implementation."""
        pass

    @pytest.fixture
    def session(self):
        """Create a fresh session for each test."""
        return Session()

    @pytest.fixture
    def test_prompt(self):
        """Create a test prompt."""
        return TestPrompt(
            name="test_prompt",
            description="Test prompt",
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello!",
            metadata=PromptMetadata(
                type="test",
                expected_response=ResponseFormat(
                    format="json",
                    schema='{"type": "object"}'
                )
            ),
            current_version=VersionInfo(
                number="1.0.0",
                author="test",
                timestamp=datetime.now(),
                description="Initial version",
                change_type="major"
            )
        )
