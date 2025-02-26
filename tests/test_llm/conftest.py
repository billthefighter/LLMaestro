"""Test fixtures for LLM-specific functionality."""
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Set
from unittest.mock import AsyncMock, MagicMock, Mock

import base64
import numpy as np
import pytest
import tiktoken
from PIL import Image
from anthropic import Anthropic
from anthropic.types import Message, MessageParam, TextBlock
from openai import AsyncOpenAI
from transformers import AutoTokenizer


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
