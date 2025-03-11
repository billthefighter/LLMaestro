"""Test configuration and fixtures."""

import pytest
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, Mock
from io import BytesIO

import base64
import numpy as np
from PIL import Image
from anthropic import Anthropic
from anthropic.types import Message, MessageParam, TextBlock
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.core.models import LLMResponse, TokenUsage, ContextMetrics
from llmaestro.prompts.base import BasePrompt
from llmaestro.llm.models import LLMState, LLMProfile, LLMRuntimeConfig, Provider
from llmaestro.llm.capabilities import LLMCapabilities, ProviderCapabilities
from llmaestro.config.base import RateLimitConfig
import json
from llmaestro.llm.llm_registry import LLMRegistry
from pydantic import BaseModel, Field


# Test constants
TEST_API_KEY = "test-api-key"
TEST_RESPONSE = {
    "id": "test_response_id",
    "content": [{"type": "text", "text": "Test response"}],
    "usage": {"input_tokens": 10, "output_tokens": 20}
}


@pytest.fixture
def test_response():
    """Test response data."""
    return TEST_RESPONSE


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img_array = np.zeros((64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

    return {
        "content": img_base64,
        "media_type": "image/png"
    }


class MockLLMInterface(BaseLLMInterface):
    """Mock LLM interface for testing."""

    @property
    def model_family(self) -> str:
        """Get the model family."""
        return "mock"

    async def initialize(self) -> None:
        """Initialize the interface."""
        pass

    async def process(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Process the prompt and return a mock response."""
        # For invoice extraction test, return the expected values
        if isinstance(prompt, BasePrompt) and prompt.name == "invoice_extractor":
            content = {
                "total": "381.12 euro",
                "vat_percentage": "19%",
                "vat_total": "72.41 euro",
                "gross_total": "453.52 euro"
            }
            return LLMResponse(
                content=json.dumps(content),
                success=True,
                token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                context_metrics=ContextMetrics(
                    max_context_tokens=4096,
                    current_context_tokens=150,
                    available_tokens=3946,
                    context_utilization=0.037
                ),
                metadata={"model": "mock-model"}
            )

        return LLMResponse(
            content="mock response",
            success=True,
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            context_metrics=ContextMetrics(
                max_context_tokens=0,
                current_context_tokens=0,
                available_tokens=0,
                context_utilization=0.0
            ),
            metadata={"model": "mock-model"}
        )

    async def stream(
        self,
        prompt: Union[BasePrompt, str],
        variables: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[LLMResponse]:
        """Stream responses (not implemented for mock)."""
        raise NotImplementedError("Streaming not implemented in mock")

    async def batch_process(
        self,
        prompts: list[Union[BasePrompt, str]],
        variables: Optional[list[Optional[Dict[str, Any]]]] = None,
        batch_size: Optional[int] = None,
    ) -> list[LLMResponse]:
        """Batch process prompts (not implemented for mock)."""
        raise NotImplementedError("Batch processing not implemented in mock")


@pytest.fixture
def llm_interface() -> BaseLLMInterface:
    """Create a mock LLM interface for testing."""
    state = LLMState(
        profile=LLMProfile(
            name="mock-model",
            capabilities=LLMCapabilities(
                max_context_window=4096,
                max_output_tokens=1000,
                supports_streaming=True,
                supports_function_calling=False,
                supports_vision=True
            )
        ),
        provider=Provider(
            family="mock",
            api_base="http://mock",
            capabilities=ProviderCapabilities(
                supports_batch_requests=False,
                supports_async_requests=True,
                supports_streaming=True,
                supports_model_selection=True,
                supports_custom_models=False,
                supports_api_key_auth=True,
                supports_oauth=False,
                supports_organization_ids=False,
                supports_custom_endpoints=False,
                supports_concurrent_requests=True,
                max_concurrent_requests=10,
                requests_per_minute=100,
                tokens_per_minute=10000,
                supports_usage_tracking=True,
                supports_cost_tracking=True,
                supports_quotas=True,
                supports_fine_tuning=False,
                supports_model_deployment=False,
                supports_custom_domains=False,
                supports_audit_logs=False
            ),
            rate_limits=RateLimitConfig(
                requests_per_minute=100,
                max_daily_tokens=100000,
                alert_threshold=0.8
            )
        ),
        runtime_config=LLMRuntimeConfig(
            max_tokens=1000,
            temperature=0.7,
            max_context_tokens=4096
        )
    )
    return MockLLMInterface(state=state)

def find_cheapest_model_with_capabilities(llm_registry: LLMRegistry, required_capabilities: Set[str]) -> str:
    """Utility function to find the cheapest model with required capabilities.

    Args:
        llm_registry: The LLM registry to search in
        required_capabilities: Set of capability flags that the model must support

    Returns:
        The name of the cheapest model that supports all required capabilities

    Raises:
        pytest.skip: If no model with the required capabilities is available
    """
    model_name = llm_registry.find_cheapest_model_with_capabilities(required_capabilities)

    if not model_name:
        pytest.skip(f"No model with required capabilities {required_capabilities} is available")

    return model_name  # This will always be a string since we skip if it's None
