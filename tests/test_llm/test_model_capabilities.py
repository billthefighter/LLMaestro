"""Tests for model capabilities detection and registry functions."""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from anthropic import Anthropic
from openai import AsyncOpenAI

from src.llm.models import (
    ModelCapabilitiesDetector,
    ModelRegistry,
    ModelFamily,
    ModelCapabilities,
    ModelDescriptor,
    RangeConfig,
    register_claude_3_5_sonnet_latest,
)

# Test Data
TEST_API_KEY = "test_api_key"
TEST_MODEL_NAME = "claude-3-5-sonnet-latest"


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    with patch("anthropic.Anthropic", autospec=True) as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch("openai.AsyncOpenAI", autospec=True) as mock:
        mock_instance = AsyncMock()
        mock_instance.models.retrieve = AsyncMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.mark.asyncio
class TestModelCapabilitiesDetector:
    """Test the ModelCapabilitiesDetector class."""

    async def test_detect_anthropic_capabilities(self, mock_anthropic_client):
        """Test detecting capabilities for Anthropic models."""
        capabilities = await ModelCapabilitiesDetector.detect_capabilities(
            provider="anthropic",
            model_name=TEST_MODEL_NAME,
            api_key=TEST_API_KEY
        )

        assert isinstance(capabilities, ModelCapabilities)
        assert capabilities.supports_streaming is True
        assert capabilities.supports_function_calling is True
        assert capabilities.supports_vision is True
        assert capabilities.max_context_window == 200000
        assert capabilities.input_cost_per_1k_tokens == 0.015
        assert capabilities.output_cost_per_1k_tokens == 0.015
        assert capabilities.supports_json_mode is True
        assert capabilities.supports_tools is True
        assert "image/jpeg" in capabilities.supported_mime_types
        assert "image/png" in capabilities.supported_mime_types

    async def test_detect_openai_capabilities(self, mock_openai_client):
        """Test detecting capabilities for OpenAI models."""
        # Mock the model info response
        mock_model_info = MagicMock()
        mock_model_info.context_window = 8192
        mock_openai_client.models.retrieve.return_value = mock_model_info

        capabilities = await ModelCapabilitiesDetector.detect_capabilities(
            provider="openai",
            model_name="gpt-4",
            api_key=TEST_API_KEY
        )

        assert isinstance(capabilities, ModelCapabilities)
        assert capabilities.supports_streaming is True
        assert capabilities.supports_function_calling is True
        assert capabilities.max_context_window == 8192
        assert capabilities.input_cost_per_1k_tokens == 0.01
        assert capabilities.output_cost_per_1k_tokens == 0.03
        assert capabilities.supports_json_mode is True
        assert capabilities.supports_tools is True

    async def test_unsupported_provider(self):
        """Test detecting capabilities for an unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            await ModelCapabilitiesDetector.detect_capabilities(
                provider="unsupported",
                model_name="test-model",
                api_key=TEST_API_KEY
            )

    async def test_anthropic_api_error(self, mock_anthropic_client):
        """Test handling Anthropic API errors."""
        mock_anthropic_client.side_effect = Exception("API Error")

        with pytest.raises(RuntimeError, match="Failed to detect Anthropic capabilities"):
            await ModelCapabilitiesDetector.detect_capabilities(
                provider="anthropic",
                model_name=TEST_MODEL_NAME,
                api_key=TEST_API_KEY
            )

    async def test_openai_api_error(self, mock_openai_client):
        """Test handling OpenAI API errors."""
        mock_openai_client.models.retrieve.side_effect = Exception("API Error")

        with pytest.raises(RuntimeError, match="Failed to detect OpenAI capabilities"):
            await ModelCapabilitiesDetector.detect_capabilities(
                provider="openai",
                model_name="gpt-4",
                api_key=TEST_API_KEY
            )


@pytest.mark.asyncio
class TestModelRegistry:
    """Test the ModelRegistry class with dynamic capability detection."""

    async def test_detect_and_register_model(self, mock_anthropic_client):
        """Test detecting and registering a new model."""
        registry = ModelRegistry()

        descriptor = await registry.detect_and_register_model(
            provider="anthropic",
            model_name=TEST_MODEL_NAME,
            api_key=TEST_API_KEY
        )

        assert isinstance(descriptor, ModelDescriptor)
        assert descriptor.name == TEST_MODEL_NAME
        assert descriptor.family == ModelFamily.CLAUDE
        assert isinstance(descriptor.capabilities, ModelCapabilities)
        assert descriptor.min_api_version == "2024-02-29"
        assert isinstance(descriptor.release_date, datetime)

        # Verify model was registered
        registered_model = registry.get_model(TEST_MODEL_NAME)
        assert registered_model is not None
        assert registered_model.name == TEST_MODEL_NAME

    async def test_detect_and_register_model_error(self, mock_anthropic_client):
        """Test error handling when detecting and registering a model."""
        mock_anthropic_client.side_effect = Exception("API Error")
        registry = ModelRegistry()

        with pytest.raises(RuntimeError, match="Failed to detect Anthropic capabilities"):
            await registry.detect_and_register_model(
                provider="anthropic",
                model_name=TEST_MODEL_NAME,
                api_key=TEST_API_KEY
            )

    async def test_register_claude_3_5_sonnet_latest(self, mock_anthropic_client):
        """Test the helper function for registering claude-3-5-sonnet-latest."""
        descriptor = await register_claude_3_5_sonnet_latest(TEST_API_KEY)

        assert isinstance(descriptor, ModelDescriptor)
        assert descriptor.name == "claude-3-5-sonnet-latest"
        assert descriptor.family == ModelFamily.CLAUDE
        assert isinstance(descriptor.capabilities, ModelCapabilities)
        assert descriptor.capabilities.supports_function_calling is True
        assert descriptor.capabilities.supports_vision is True
        assert descriptor.capabilities.max_context_window == 200000
