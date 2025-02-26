"""Test OpenAI interface implementation."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
from typing import List, Optional

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.completion_usage import CompletionUsage

from llmaestro.llm.models import (
    LLMState,
    LLMProfile,
    LLMCapabilities,
    LLMMetadata,
    Provider,
    LLMRuntimeConfig,
)
from llmaestro.llm.capabilities import ProviderCapabilities
from llmaestro.llm.rate_limiter import RateLimitConfig
from llmaestro.llm.credentials import APIKey
from llmaestro.default_library.defined_profiders.openai.interface import OpenAIInterface
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.types import PromptMetadata, ResponseFormat, ResponseFormatType, VersionInfo
from llmaestro.llm.enums import MediaType

# Create a concrete test prompt class
class TestPrompt(BasePrompt):
    """Test prompt class that implements abstract methods."""

    name: str = "test_prompt"
    description: str = "Test prompt"
    system_prompt: str = "Test system prompt"
    user_prompt: str = "Test user prompt"
    metadata: PromptMetadata = PromptMetadata(
        type="test",
        expected_response=ResponseFormat(
            format=ResponseFormatType.TEXT,
            schema=None,
        ),
        tags=[],
    )

    async def load(self):
        """Implement abstract load method."""
        pass

    async def save(self):
        """Implement abstract save method."""
        pass

@pytest.fixture
def mock_openai_response() -> ChatCompletion:
    """Create a mock OpenAI response."""
    return ChatCompletion(
        id='mock-completion-id',
        choices=[
            Choice(
                finish_reason='stop',
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content='Test response',
                    role='assistant',
                    function_call=None,
                    tool_calls=None,
                ),
            )
        ],
        created=1234567890,
        model='gpt-4',
        object='chat.completion',
        system_fingerprint=None,
        usage=CompletionUsage(
            completion_tokens=10,
            prompt_tokens=20,
            total_tokens=30,
        ),
    )

@pytest.fixture
def mock_openai_stream_response() -> List[ChatCompletionChunk]:
    """Create a mock OpenAI streaming response."""
    return [
        ChatCompletionChunk(
            id='mock-chunk-1',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        content='This ',
                        role='assistant',
                        function_call=None,
                        tool_calls=None,
                    ),
                    finish_reason=None,
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='gpt-4',
            object='chat.completion.chunk',
            system_fingerprint=None,
            usage=None,
        ),
        ChatCompletionChunk(
            id='mock-chunk-2',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        content='is ',
                        role='assistant',
                        function_call=None,
                        tool_calls=None,
                    ),
                    finish_reason=None,
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='gpt-4',
            object='chat.completion.chunk',
            system_fingerprint=None,
            usage=None,
        ),
        ChatCompletionChunk(
            id='mock-chunk-3',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        content='a test',
                        role='assistant',
                        function_call=None,
                        tool_calls=None,
                    ),
                    finish_reason='stop',
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='gpt-4',
            object='chat.completion.chunk',
            system_fingerprint=None,
            usage=CompletionUsage(
                completion_tokens=10,
                prompt_tokens=20,
                total_tokens=30,
            ),
        ),
    ]

@pytest.fixture
def openai_interface() -> OpenAIInterface:
    """Create an OpenAI interface instance."""
    interface = OpenAIInterface(
        SUPPORTED_MEDIA_TYPES={MediaType.PDF, MediaType.JPEG, MediaType.PNG},
        state=LLMState(
            profile=LLMProfile(
                name="gpt-4",
                version="1.0",
                description="Test model",
                capabilities=LLMCapabilities(
                    max_context_window=8192,
                    max_output_tokens=2048,
                    supports_streaming=True,
                    supports_function_calling=True,
                    supports_vision=False,
                    supports_embeddings=False,
                    supports_json_mode=True,
                    supports_system_prompt=True,
                    supports_tools=True,
                    supports_parallel_requests=True,
                    supports_frequency_penalty=True,
                    supports_presence_penalty=True,
                    supports_stop_sequences=True,
                    supports_message_role=True,
                    typical_speed=200.0,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.01,
                    output_cost_per_1k_tokens=0.03,
                ),
                metadata=LLMMetadata(
                    release_date=datetime.now(),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2024-01-01",
                ),
            ),
            provider=Provider(
                family="openai",
                description="OpenAI API Provider",
                capabilities=ProviderCapabilities(
                    supports_batch_requests=True,
                    supports_async_requests=True,
                    supports_streaming=True,
                    supports_model_selection=True,
                    supports_custom_models=False,
                    supports_api_key_auth=True,
                    supports_oauth=False,
                    supports_organization_ids=True,
                    supports_custom_endpoints=False,
                    supports_concurrent_requests=True,
                    max_concurrent_requests=50,
                    requests_per_minute=3500,
                    tokens_per_minute=180000,
                    supports_usage_tracking=True,
                    supports_cost_tracking=True,
                    supports_quotas=True,
                    supports_fine_tuning=True,
                    supports_model_deployment=False,
                    supports_custom_domains=False,
                    supports_audit_logs=True,
                ),
                api_base="https://api.openai.com/v1",
                rate_limits=RateLimitConfig(
                    requests_per_minute=3500,
                    max_daily_tokens=1000000,
                    alert_threshold=0.8,
                ),
            ),
            runtime_config=LLMRuntimeConfig(
                max_tokens=2048,
                temperature=0.7,
                max_context_tokens=8192,
                stream=True,
            ),
        ),
        credentials=APIKey(key="test-api-key"),
    )
    interface.client = AsyncMock()
    interface.client.chat.completions.create = AsyncMock()
    return interface

@pytest.mark.asyncio
async def test_process_string_prompt(openai_interface: OpenAIInterface, mock_openai_response: ChatCompletion):
    """Test processing a simple string prompt."""
    openai_interface.client.chat.completions.create.return_value = mock_openai_response
    response = await openai_interface.process("Test prompt")
    assert response.success
    assert response.content == "Test response"
    assert response.token_usage.total_tokens == 30

@pytest.mark.asyncio
async def test_process_base_prompt(openai_interface: OpenAIInterface, mock_openai_response: ChatCompletion):
    """Test processing a BasePrompt instance."""
    prompt = TestPrompt()
    openai_interface.client.chat.completions.create.return_value = mock_openai_response
    response = await openai_interface.process(prompt)
    assert response.success
    assert response.content == "Test response"
    assert response.token_usage.total_tokens == 30

@pytest.mark.asyncio
async def test_stream_response(openai_interface: OpenAIInterface, mock_openai_stream_response: List[ChatCompletionChunk]):
    """Test streaming response processing."""
    async def mock_aiter():
        for chunk in mock_openai_stream_response:
            yield chunk

    openai_interface.client.chat.completions.create.return_value = mock_aiter()
    responses = []
    accumulated_content = ""
    async for response in openai_interface.stream("Test prompt"):
        responses.append(response)
        accumulated_content += response.content

    assert len(responses) == 3
    assert all(r.success for r in responses)
    assert accumulated_content == "This is a test"  # Check accumulated content
    # Check individual chunks
    assert responses[0].content == "This "
    assert responses[1].content == "is "
    assert responses[2].content == "a test"

@pytest.mark.asyncio
async def test_error_handling(openai_interface: OpenAIInterface):
    """Test error handling in the interface."""
    openai_interface.client.chat.completions.create.side_effect = Exception("Test error")
    response = await openai_interface.process("Test prompt")
    assert not response.success
    assert response.error is not None and "Test error" in response.error

@pytest.mark.asyncio
async def test_batch_process(openai_interface: OpenAIInterface, mock_openai_response: ChatCompletion):
    """Test batch processing of prompts."""
    prompts = [
        TestPrompt(),
        "Test prompt 2",
    ]
    openai_interface.client.chat.completions.create.return_value = mock_openai_response
    responses = await openai_interface.batch_process(prompts)
    assert len(responses) == 2
    assert all(r.success for r in responses)
    assert all(r.content == "Test response" for r in responses)
    assert all(r.token_usage.total_tokens == 30 for r in responses)
