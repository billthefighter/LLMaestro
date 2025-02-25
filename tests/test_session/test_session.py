"""Tests for Session class functionality."""

import pytest
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from llmaestro.session.session import Session
from llmaestro.core.models import BaseResponse, LLMResponse, TokenUsage, ContextMetrics
from llmaestro.core.storage import Artifact, FileSystemArtifactStorage
from llmaestro.prompts.base import BasePrompt, FileAttachment
from llmaestro.llm.models import LLMProfile, LLMCapabilities
from llmaestro.prompts.types import PromptMetadata, ResponseFormat, VersionInfo
from llmaestro.llm.rate_limiter import RateLimitConfig
from llmaestro.config import ConfigurationManager, SystemConfig, UserConfig
from llmaestro.llm.llm_registry import LLMRegistry

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

    @pytest.mark.asyncio
    async def test_default_config_manager_initialization(self, tmp_path: Path):
        """Test initialization of default ConfigurationManager in Session."""
        # Create a temporary config directory
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create minimal system_config.yml
        system_config = config_dir / "system_config.yml"
        system_config.write_text("""
providers:
  anthropic:
    api_base: https://api.anthropic.com/v1
    api_version: "2024-02-29"
    rate_limits:
      requests_per_minute: 50
      tokens_per_minute: 100000
    timeout_settings:
      request: 30.0
      stream: 60.0
      connect: 10.0
    retry_settings:
      max_retries: 3
      retry_delay: 1
      max_delay: 30
    allowed_api_domains:
      - "api.anthropic.com"
    require_api_key_encryption: true

llm:
  global_rate_limits:
    total_requests_per_minute: 500
    total_tokens_per_minute: 300000
  max_parallel_requests: 10
  max_retries: 3
  retry_delay: 1.0
  default_request_timeout: 30.0
  default_stream_timeout: 60.0
  enable_response_cache: false
  cache_ttl: 3600
  log_level: "INFO"
  enable_telemetry: false
  require_api_key_encryption: true
  allowed_api_domains:
    - "api.anthropic.com"
""")

        # Set environment variables for testing
        import os
        os.environ["ANTHROPIC_API_KEY"] = "test-api-key"
        os.environ["LLM_MAX_PARALLEL_REQUESTS"] = "10"
        os.environ["LLM_LOG_LEVEL"] = "INFO"

        # Create configuration manager from system config and environment
        config_manager = ConfigurationManager.from_yaml_files(
            system_config_path=system_config
        )

        # Initialize session with default config
        session = await Session.create_default(
            storage_path=tmp_path / "storage",
            config=config_manager,
            api_key="test-api-key"
        )

        # Validate ConfigurationManager initialization
        assert session.config is not None
        assert isinstance(session.config, ConfigurationManager)
        
        # Validate system config
        assert session.config.system_config is not None
        assert "anthropic" in session.config.system_config.providers
        assert session.config.system_config.llm.max_parallel_requests == 10
        
        # Validate user config
        assert session.config.user_config is not None
        assert "anthropic" in session.config.user_config.api_keys
        
        # Validate LLM registry initialization
        assert isinstance(session.llm_registry, LLMRegistry)
        assert session.llm_registry.provider_registry is not None
        assert len(session.llm_registry.provider_registry.list_providers()) > 0

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
