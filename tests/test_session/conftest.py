"""Fixtures for session tests."""
import pytest
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from llmaestro.session.session import Session
from llmaestro.core.models import BaseResponse, LLMResponse, TokenUsage, ContextMetrics
from llmaestro.core.storage import Artifact, FileSystemArtifactStorage
from llmaestro.prompts.base import BasePrompt, FileAttachment
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.llm.models import LLMProfile, LLMCapabilities
from llmaestro.prompts.types import PromptMetadata, ResponseFormat, VersionInfo
from llmaestro.llm.rate_limiter import RateLimitConfig
from llmaestro.config import ConfigurationManager

@pytest.fixture
def mock_storage_path(tmp_path) -> Path:
    """Create a temporary storage path for testing."""
    storage_dir = tmp_path / "test_session_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir

@pytest.fixture
def mock_prompt() -> MemoryPrompt:
    """Create a mock prompt for testing."""
    version_info = VersionInfo(
        number="1.0.0",
        timestamp=datetime.now(),
        author="test",
        description="Test version",
        change_type="initial"
    )

    return MemoryPrompt(
        name="test_prompt",
        description="Test prompt",
        system_prompt="You are a test assistant",
        user_prompt="Hello {name}",
        metadata=PromptMetadata(
            type="test",
            expected_response=ResponseFormat(format="text", schema=None),
            tags=["test"],
            is_active=True
        ),
        current_version=version_info,
        version_history=[version_info]
    )

@pytest.fixture
def mock_llm_response(mock_LLMProfile: LLMProfile) -> LLMResponse:
    """Create a mock LLM response for testing."""
    return LLMResponse(
        content="Test response",
        model=mock_LLMProfile,
        success=True,
        token_usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        ),
        context_metrics=ContextMetrics(
            max_context_tokens=4096,
            current_context_tokens=100,
            available_tokens=3996,
            context_utilization=0.024
        )
    )

@pytest.fixture
async def basic_session(config_manager: ConfigurationManager, mock_storage_path: Path) -> Session:
    """Create a basic session for testing using the recommended factory method.

    This fixture creates a fully initialized session using create_default,
    which handles both sync and async initialization automatically.
    """
    return await Session.create_default(
        config=config_manager,
        storage_path=mock_storage_path,
        api_key="test-api-key"
    )
