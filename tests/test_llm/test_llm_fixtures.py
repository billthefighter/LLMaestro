"""Tests to verify that all fixtures can be instantiated without errors."""

import pytest


@pytest.mark.asyncio
async def test_test_response(test_response):
    """Verify test_response fixture."""
    assert test_response is not None


def test_mock_llm_registry_simple(mock_llm_registry_simple):
    """Verify mock_llm_registry_simple fixture."""
    assert mock_llm_registry_simple is not None


def test_mock_llm_registry_full(mock_llm_registry_full):
    """Verify mock_llm_registry_full fixture."""
    assert mock_llm_registry_full is not None


def test_mock_tokenizer(mock_tokenizer):
    """Verify mock_tokenizer fixture."""
    assert mock_tokenizer is not None


@pytest.mark.asyncio
async def test_mock_anthropic_client(mock_anthropic_client):
    """Verify mock_anthropic_client fixture."""
    assert mock_anthropic_client is not None
    # Test async create method
    response = await mock_anthropic_client.messages.create(
        messages=[{"role": "user", "content": "test"}]
    )
    assert response is not None


def test_test_config(test_config):
    """Verify test_config fixture."""
    assert test_config is not None


@pytest.mark.asyncio
async def test_anthropic_llm(anthropic_llm):
    """Verify anthropic_llm fixture."""
    assert anthropic_llm is not None


def test_sample_image(sample_image):
    """Verify sample_image fixture."""
    assert sample_image is not None
    assert "content" in sample_image
    assert "media_type" in sample_image


def test_mock_token_counter(mock_token_counter):
    """Verify mock_token_counter fixture."""
    # This fixture returns None but should set up the mock
    pass


def test_sample_text(sample_text):
    """Verify sample_text fixture."""
    assert sample_text is not None
    assert isinstance(sample_text, str)


def test_sample_messages(sample_messages):
    """Verify sample_messages fixture."""
    assert sample_messages is not None
    assert isinstance(sample_messages, list)
    assert len(sample_messages) > 0


def test_mock_tiktoken(mock_tiktoken):
    """Verify mock_tiktoken fixture."""
    assert mock_tiktoken is not None
    # Test the encode method
    tokens = mock_tiktoken.encode("test")
    assert tokens is not None


def test_mock_hf_tokenizer(mock_hf_tokenizer):
    """Verify mock_hf_tokenizer fixture."""
    # This fixture returns None but should set up the mock
    pass


def test_sample_image_data(sample_image_data):
    """Verify sample_image_data fixture."""
    assert sample_image_data is not None
    assert isinstance(sample_image_data, list)
    assert len(sample_image_data) > 0
