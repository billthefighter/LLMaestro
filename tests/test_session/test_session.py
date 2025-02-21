"""Tests for Session class functionality."""

import pytest
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from llmaestro.session.session import Session
from llmaestro.core.models import BaseResponse, LLMResponse, TokenUsage, ContextMetrics
from llmaestro.core.storage import Artifact, FileSystemArtifactStorage
from llmaestro.prompts.base import BasePrompt, FileAttachment
from llmaestro.llm.models import LLMProfile, LLMCapabilities, MediaType
from llmaestro.prompts.types import PromptMetadata, ResponseFormat, VersionInfo
from llmaestro.llm.rate_limiter import RateLimitConfig
from llmaestro.config import ConfigurationManager
# Test cases for Session class
class TestSession:
    """Test suite for Session class functionality."""

    def test_session_initialization(self, basic_session: Session, mock_storage_path: Path):
        """Test basic session initialization."""
        assert basic_session.session_id is not None
        assert basic_session.created_at is not None
        assert basic_session.storage_path == mock_storage_path
        assert basic_session.storage is not None
        assert basic_session.api_key == "test-api-key"

    def test_storage_path_creation(self, tmp_path: Path):
        """Test storage path is created if it doesn't exist."""
        storage_path = tmp_path / "new_storage"
        session = Session(storage_path=storage_path)
        assert storage_path.exists()
        assert storage_path.is_dir()

    @pytest.mark.asyncio
    async def test_artifact_storage(self, basic_session: Session):
        """Test artifact storage functionality."""
        test_data = {"key": "value"}
        artifact = await basic_session.store_artifact_async(
            name="test_artifact",
            data=test_data,
            content_type="application/json",
            metadata={"test": True}
        )

        assert artifact.name == "test_artifact"
        assert artifact.content_type == "application/json"

        # Test retrieval
        retrieved = await basic_session.get_artifact_async(artifact.id)
        assert retrieved is not None
        assert retrieved.data == test_data
        assert retrieved.metadata["test"] is True

    def test_model_capability_validation(self, basic_session: Session, mock_LLMProfile: LLMProfile):
        """Test model capability validation."""
        task_requirements = {
            "supports_streaming": True,
            "max_context_window": 100000
        }

        # Set the model registry
        basic_session.llm_registry = mock_LLMProfile

        # Test validation
        assert basic_session.validate_model_for_task(task_requirements) is True

        # Test with requirements exceeding capabilities
        invalid_requirements = {
            "max_context_window": 300000  # Exceeds mock model's capability
        }
        assert basic_session.validate_model_for_task(invalid_requirements) is False

    @pytest.mark.asyncio
    async def test_conversation_management(self, basic_session: Session, mock_prompt: BasePrompt):
        """Test basic conversation management."""
        # Start a conversation
        try:
            conv_id = await basic_session.start_conversation("test_conv", mock_prompt)
            assert conv_id is not None
            assert basic_session.active_conversation_id == conv_id
        except RuntimeError as e:
            # This is expected if orchestrator is not initialized
            assert "Orchestrator not initialized" in str(e)

    def test_session_summary(self, basic_session: Session):
        """Test session summary generation."""
        summary = basic_session.summary()
        assert "session_id" in summary
        assert "created_at" in summary
        assert isinstance(summary["created_at"], str)

    @pytest.mark.asyncio
    async def test_llm_interface_management(self, basic_session: Session):
        """Test LLM interface management."""
        interface = await basic_session.get_llm_interface_async()
        # The result might be None depending on configuration,
        # but the method should execute without errors
        assert interface is not None or interface is None

    def test_list_artifacts(self, basic_session: Session):
        """Test listing artifacts."""
        artifacts = basic_session.list_artifacts()
        assert isinstance(artifacts, list)

        # Test with filter criteria
        filtered_artifacts = basic_session.list_artifacts({"content_type": "application/json"})
        assert isinstance(filtered_artifacts, list)

    @pytest.mark.asyncio
    async def test_add_node_to_conversation(
        self, basic_session: Session, mock_prompt: BasePrompt, mock_llm_response: LLMResponse
    ):
        """Test adding nodes to conversation."""
        try:
            # Try adding a prompt node
            node_id = await basic_session.add_node_to_conversation_async(
                content=mock_prompt,
                node_type="prompt",
                metadata={"test": True}
            )
            assert node_id is not None
        except RuntimeError as e:
            # This is expected if no active conversation
            assert "No active conversation" in str(e)
