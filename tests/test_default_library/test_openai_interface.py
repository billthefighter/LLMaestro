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
from openai.types.file_object import FileObject

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
from llmaestro.default_library.defined_providers.openai.interface import OpenAIInterface
from llmaestro.prompts.base import BasePrompt, PromptVariable, SerializableType
from llmaestro.prompts.types import (
    PromptMetadata,
    ResponseFormat,
)
from llmaestro.llm.enums import MediaType
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.core.attachments import FileAttachment
from llmaestro.llm.responses import ResponseFormatType

# Create a concrete test prompt class
@pytest.fixture
def test_prompt() -> BasePrompt:
    """Create a test prompt instance with variables."""
    return MemoryPrompt(
        name="test_prompt",
        description="Test prompt",
        system_prompt="You are helping {role} with {task}",
        user_prompt="Please {action} the following {input_type}: {content}",
        variables=[
            PromptVariable(
                name="role",
                description="The role of the assistant",
                expected_input_type=SerializableType.STRING,
            ),
            PromptVariable(
                name="task",
                description="The task to perform",
                expected_input_type=SerializableType.STRING,
            ),
            PromptVariable(
                name="action",
                description="The action to perform",
                expected_input_type=SerializableType.STRING,
            ),
            PromptVariable(
                name="input_type",
                description="The type of input",
                expected_input_type=SerializableType.STRING,
            ),
            PromptVariable(
                name="content",
                description="The content to process",
                expected_input_type=SerializableType.STRING,
            ),
        ],
        metadata=PromptMetadata(
            type="test",
            expected_response=ResponseFormat(
                format=ResponseFormatType.TEXT,
                schema=None,
            ),
        ),
    )

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
async def test_process_base_prompt(openai_interface: OpenAIInterface, test_prompt: BasePrompt, mock_openai_response: ChatCompletion):
    """Test processing a BasePrompt instance."""
    openai_interface.client.chat.completions.create.return_value = mock_openai_response
    response = await openai_interface.process(test_prompt, variables={
        "role": "a tester",
        "task": "testing",
        "action": "validate",
        "input_type": "test case",
        "content": "test content",
    })
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
async def test_prompt_variable_rendering(openai_interface: OpenAIInterface, test_prompt: BasePrompt, mock_openai_response: ChatCompletion):
    """Test that prompt variables are correctly rendered."""
    variables = {
        "role": "a developer",
        "task": "code review",
        "action": "analyze",
        "input_type": "code snippet",
        "content": "def hello(): return 'world'",
    }

    openai_interface.client.chat.completions.create.return_value = mock_openai_response
    response = await openai_interface.process(test_prompt, variables=variables)

    # Verify the API was called with correctly rendered prompts
    call_args = openai_interface.client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]

    # Find system and user messages
    system_message = next(msg for msg in messages if msg["role"] == "system")
    user_message = next(msg for msg in messages if msg["role"] == "user")

    # Verify rendered content
    assert system_message["content"] == "You are helping a developer with code review"
    assert user_message["content"] == "Please analyze the following code snippet: def hello(): return 'world'"

    # Verify response
    assert response.success
    assert response.content == "Test response"

@pytest.mark.asyncio
async def test_batch_process(openai_interface: OpenAIInterface, test_prompt: BasePrompt, mock_openai_response: ChatCompletion):
    """Test batch processing of prompts."""
    prompts = [
        test_prompt,
        "Test prompt 2",
    ]
    variables = [
        {
            "role": "a tester",
            "task": "testing",
            "action": "validate",
            "input_type": "test case",
            "content": "test content",
        },
        None,
    ]
    openai_interface.client.chat.completions.create.return_value = mock_openai_response
    responses = await openai_interface.batch_process(prompts, variables=variables)
    assert len(responses) == 2
    assert all(r.success for r in responses)
    assert all(r.content == "Test response" for r in responses)
    assert all(r.token_usage.total_tokens == 30 for r in responses)

@pytest.mark.asyncio
async def test_file_handling(openai_interface: OpenAIInterface, mock_openai_response: ChatCompletion):
    """Test handling of file attachments."""
    # Mock file upload response
    mock_file_object = FileObject(
        id="file-123",
        bytes=1000,
        created_at=1234567890,
        filename="test.pdf",
        object="file",
        purpose="assistants",
        status="processed",
        status_details=None
    )
    openai_interface.client.files.create.return_value = mock_file_object

    # Create a test file attachment
    file_content = b"Test PDF content"
    file_name = "test.pdf"

    # Upload the file
    file_obj = await openai_interface.upload_file(file_content)
    assert file_obj.id == "file-123"
    assert file_obj.filename == "test.pdf"

    # Create a file attachment
    file_attachment = FileAttachment(
        content=file_content,
        file_name=file_name,
        media_type=MediaType.PDF
    )
    # Set the file ID directly
    setattr(file_attachment, "file_id", file_obj.id)

    # Create a prompt with the file attachment
    test_prompt = MemoryPrompt(
        name="test_prompt",
        description="Test prompt with file",
        system_prompt="Process this file",
        user_prompt="Please analyze the attached file",
        variables=[],
        metadata=PromptMetadata(
            type="test",
            expected_response=ResponseFormat(
                format=ResponseFormatType.TEXT,
                schema=None,
            ),
        ),
        attachments=[file_attachment]
    )

    # Process the prompt
    openai_interface.client.chat.completions.create.return_value = mock_openai_response
    response = await openai_interface.process(test_prompt)

    # Verify the response
    assert response.success
    assert response.content == "Test response"

    # Verify that the file was included in the messages
    call_args = openai_interface.client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]

    # Find the file message
    file_message = next((msg for msg in messages if msg["role"] == "assistant" and "file_ids" in msg), None)
    assert file_message is not None
    assert file_message["file_ids"] == ["file-123"]
    assert file_message["content"] is None
