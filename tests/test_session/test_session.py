import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta

from src.session.session import Session
from src.core.config import get_config
from src.core.models import BaseResponse
from src.llm.models import ModelConfig

class TestSession:
    @pytest.fixture
    def session(self):
        """Create a fresh session for each test."""
        return Session()

    def test_session_initialization(self, session):
        """Test basic session initialization."""
        assert session.session_id is not None
        assert session.created_at <= datetime.now()
        assert isinstance(session.storage_path, Path)
        assert session.storage_path.exists()

    def test_get_llm_interface(self, session):
        """Test LLM interface retrieval."""
        llm_interface = session.get_llm_interface()

        # Verify interface creation
        assert llm_interface is not None

        # Verify configuration
        assert llm_interface.config.provider is not None
        assert llm_interface.config.model_name is not None

    def test_store_and_retrieve_artifact(self, session):
        """Test artifact storage and retrieval."""
        # Store an artifact
        test_data = {"key": "value", "number": 42}
        artifact = session.store_artifact(
            "test_artifact",
            test_data,
            metadata={"source": "test"}
        )

        # Verify artifact storage
        assert artifact.name == "test_artifact"
        assert artifact.path.exists()
        assert "test_artifact" in session.artifacts

        # Verify artifact contents
        with open(artifact.path, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data

    def test_model_capabilities(self, session):
        """Test model capabilities retrieval."""
        model_config = session.get_model_capabilities()

        # Verify model config
        assert isinstance(model_config, ModelConfig)
        assert model_config.max_context_tokens > 0
        assert 'input_cost_per_1k_tokens' in model_config.model_dump()

    def test_validate_model_for_task(self, session):
        """Test model validation for task requirements."""
        # Get current model capabilities
        model_config = session.get_model_capabilities()

        # Test with basic requirements
        task_requirements = {
            'max_context_tokens': model_config.max_context_tokens - 1000,
            'input_cost_per_1k_tokens': model_config.input_cost_per_1k_tokens + 0.001
        }

        # Should pass if requirements are less than or equal to model capabilities
        assert session.validate_model_for_task(task_requirements) is True

        # Test with impossible requirements
        impossible_requirements = {
            'max_context_tokens': model_config.max_context_tokens * 2,
            'non_existent_feature': True
        }

        assert session.validate_model_for_task(impossible_requirements) is False

    def test_session_summary(self, session):
        """Test session summary generation."""
        # Store some artifacts and responses to populate summary
        session.store_artifact("test_artifact", {"key": "value"})

        mock_response = BaseResponse(
            success=True,
            metadata={"test": "data"}
        )
        session.responses["test_response"] = mock_response

        # Generate summary
        summary = session.summary()

        # Verify summary contents
        assert 'session_id' in summary
        assert 'created_at' in summary
        assert summary['artifact_count'] == 1
        assert summary['response_count'] == 1
        assert 'config' in summary
        assert 'model_capabilities' in summary['config']

    def test_custom_model_selection(self, session):
        """Test selecting a different model."""
        # Get available models from registry
        registry = get_config().model_registry

        # Try to get a different model from the default
        available_models = registry.get_models_by_feature('vision')

        if len(available_models) > 1:
            # Select a different model with vision capabilities
            alternative_model = [m for m in available_models if m != session.config.get_agent_config().get('model')][0]

            # Get LLM interface with alternative model
            alt_interface = session.get_llm_interface(model_name=alternative_model)

            assert alt_interface.config.model_name == alternative_model

    def test_api_key_override(self):
        """Test API key override during session creation."""
        custom_api_key = "test_custom_api_key"
        session = Session(api_key=custom_api_key)

        # Verify API key is set
        assert session.api_key == custom_api_key

        # Get LLM interface and verify it uses the custom API key
        llm_interface = session.get_llm_interface()
        assert llm_interface.config.api_key == custom_api_key

    @pytest.mark.parametrize("storage_path", [
        "./custom_session_storage",
        Path("./another_custom_path")
    ])
    def test_custom_storage_path(self, storage_path):
        """Test custom storage path creation."""
        session = Session(storage_path=storage_path)

        # Verify storage path
        assert isinstance(session.storage_path, Path)
        assert session.storage_path.exists()
        assert str(session.storage_path) == str(Path(storage_path).resolve())

    def test_error_handling(self, session):
        """Test error scenarios."""
        # Try to get capabilities for non-existent model
        with pytest.raises(ValueError, match="Model .* not found in registry"):
            session.get_model_capabilities("non_existent_model")

        # Validate with invalid requirements
        with pytest.raises(AttributeError):
            session.validate_model_for_task({"invalid_feature": None})

@pytest.mark.integration
class TestSessionIntegration:
    def test_full_workflow(self):
        """
        Integration test demonstrating a full session workflow.
        This test shows how different components interact.
        """
        # Create session
        session = Session()

        # Store input data
        input_artifact = session.store_artifact(
            "input_data",
            {"task": "summarize document"},
            metadata={"source": "user_request"}
        )

        # Get LLM interface
        llm_interface = session.get_llm_interface()

        # Validate model capabilities
        model_capabilities = session.get_model_capabilities()
        assert model_capabilities is not None

        # Optional: Add more complex workflow steps
        # This is a placeholder for actual LLM interaction
        mock_response = BaseResponse(
            success=True,
            metadata={
                "input_artifact": input_artifact.name,
                "processing_time": 2.5
            }
        )
        session.responses["workflow_response"] = mock_response

        # Generate summary
        summary = session.summary()
        assert summary['artifact_count'] == 1
        assert summary['response_count'] == 1
