"""Tests to validate proper initialization of session test fixtures."""
import pytest
from pathlib import Path

from llmaestro.session.session import Session
from llmaestro.core.models import LLMResponse
from llmaestro.prompts.base import BasePrompt


class TestFixtures:
    """Test suite for session test fixtures."""

    def test_mock_storage_path_fixture(self, mock_storage_path: Path):
        """Test that mock_storage_path fixture initializes."""
        assert mock_storage_path is not None
        assert isinstance(mock_storage_path, Path)

    def test_mock_prompt_fixture(self, mock_prompt: BasePrompt):
        """Test that mock_prompt fixture initializes."""
        assert mock_prompt is not None
        assert isinstance(mock_prompt, BasePrompt)

    def test_mock_llm_response_fixture(self, mock_llm_response: LLMResponse):
        """Test that mock_llm_response fixture initializes."""
        assert mock_llm_response is not None
        assert isinstance(mock_llm_response, LLMResponse)

    def test_basic_session_fixture(self, basic_session: Session):
        """Test that basic_session fixture initializes."""
        assert basic_session is not None
        assert isinstance(basic_session, Session)
