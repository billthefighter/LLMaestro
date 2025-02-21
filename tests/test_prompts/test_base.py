"""Tests for the base prompt functionality."""
import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError
import base64

from llmaestro.prompts.base import BasePrompt, PromptMetadata, FileAttachment
from llmaestro.prompts.types import VersionInfo, ResponseFormat
from llmaestro.llm.enums import MediaType

# Test Data Fixtures
@pytest.fixture
def sample_version_info():
    return VersionInfo(
        number="1.0.0",
        author="test_author",
        timestamp=datetime.now(),
        description="Initial version",
        change_type="major"
    )

@pytest.fixture
def sample_response_format():
    return ResponseFormat(
        format="json",
        schema="""
        {
            "type": "object",
            "properties": {
                "response": {"type": "string"}
            }
        }
        """
    )

@pytest.fixture
def sample_metadata(sample_response_format):
    return PromptMetadata(
        type="test",
        expected_response=sample_response_format,
        model_requirements={"model": "test-model"},
        tags=["test"],
        is_active=True
    )

@pytest.fixture
def valid_prompt_data(sample_version_info, sample_metadata):
    return {
        "name": "test_prompt",
        "description": "A test prompt",
        "system_prompt": "You are a test assistant. Context: {context}",
        "user_prompt": "Hello {name}, {query}",
        "metadata": sample_metadata,
        "examples": [
            {
                "input": {"name": "Test", "query": "help"},
                "output": "Test response"
            }
        ],
        "current_version": sample_version_info,
        "version_history": []
    }

@pytest.fixture
def base_prompt(valid_prompt_data):
    class TestPrompt(BasePrompt):
        async def save(self) -> bool:
            return True

        @classmethod
        async def load(cls, identifier: str):
            return None

    return TestPrompt(**valid_prompt_data)

@pytest.fixture
def sample_image_content():
    """Sample image content for testing."""
    return base64.b64encode(b"fake_image_data").decode()

@pytest.fixture
def sample_file_attachment(sample_image_content):
    """Sample file attachment for testing."""
    return FileAttachment(
        content=sample_image_content,
        media_type=MediaType.JPEG,
        file_name="test.jpg",
        description="Test image"
    )

# Initialization Tests
class TestBasePromptInitialization:
    def test_valid_initialization(self, valid_prompt_data):
        """Test that a prompt can be initialized with valid data."""
        class TestPrompt(BasePrompt):
            async def save(self) -> bool:
                return True

            @classmethod
            async def load(cls, identifier: str):
                return None

        prompt = TestPrompt(**valid_prompt_data)
        assert prompt.name == "test_prompt"
        assert prompt.description == "A test prompt"
        assert prompt.system_prompt == "You are a test assistant. Context: {context}"
        assert prompt.user_prompt == "Hello {name}, {query}"

    def test_invalid_initialization_missing_fields(self, valid_prompt_data):
        """Test that initialization fails with missing required fields."""
        del valid_prompt_data["name"]

        with pytest.raises(ValidationError):
            class TestPrompt(BasePrompt):
                async def save(self) -> bool:
                    return True

                @classmethod
                async def load(cls, identifier: str):
                    return None

            TestPrompt(**valid_prompt_data)

# Template Validation Tests
class TestTemplateValidation:
    def test_valid_template(self, base_prompt):
        """Test that valid templates pass validation."""
        base_prompt._validate_template()  # Should not raise

    def test_unbalanced_braces(self, valid_prompt_data):
        """Test that unbalanced braces raise ValueError."""
        valid_prompt_data["user_prompt"] = "Hello {name"

        with pytest.raises(ValueError, match="Unbalanced braces"):
            class TestPrompt(BasePrompt):
                async def save(self) -> bool:
                    return True

                @classmethod
                async def load(cls, identifier: str):
                    return None

            prompt = TestPrompt(**valid_prompt_data)
            prompt._validate_template()

    def test_invalid_variable_names(self, valid_prompt_data):
        """Test that invalid variable names raise ValueError."""
        valid_prompt_data["user_prompt"] = "Hello {1invalid}"

        with pytest.raises(ValueError, match="Invalid variable name"):
            class TestPrompt(BasePrompt):
                async def save(self) -> bool:
                    return True

                @classmethod
                async def load(cls, identifier: str):
                    return None

            prompt = TestPrompt(**valid_prompt_data)
            prompt._validate_template()

# Template Variable Extraction Tests
class TestTemplateVariableExtraction:
    def test_extract_template_vars(self, base_prompt):
        """Test extraction of template variables."""
        variables = base_prompt._extract_template_vars()
        assert "name" in variables
        assert "query" in variables
        assert "context" in variables

# Prompt Rendering Tests
class TestPromptRendering:
    def test_successful_render(self, base_prompt):
        """Test successful prompt rendering with all variables."""
        system, user = base_prompt.render(
            context="test context",
            name="John",
            query="help me"
        )
        # Both system and user prompts are formatted
        assert system == "You are a test assistant. Context: test context"
        assert user == "Hello John, help me"

    def test_render_missing_variables(self, base_prompt):
        """Test that rendering fails with missing variables."""
        with pytest.raises(ValueError, match="Missing required variables"):
            base_prompt.render(name="John")  # Missing context and query

# Version Information Tests
class TestVersionInformation:
    def test_version_properties(self, base_prompt, sample_version_info):
        """Test version-related properties."""
        assert base_prompt.version == sample_version_info.number
        assert base_prompt.author == sample_version_info.author
        assert isinstance(base_prompt.created_at, datetime)
        assert isinstance(base_prompt.updated_at, datetime)
        assert isinstance(base_prompt.age, timedelta)

# Serialization Tests
class TestSerialization:
    def test_to_dict(self, base_prompt):
        """Test conversion to dictionary."""
        data = base_prompt.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == "test_prompt"
        assert "version_history" in data

        # Test without history
        data_no_history = base_prompt.to_dict(include_history=False)
        assert "version_history" not in data_no_history

    def test_to_json(self, base_prompt):
        """Test conversion to JSON."""
        json_data = base_prompt.to_json()
        assert isinstance(json_data, str)
        assert "test_prompt" in json_data

        # Test pretty printing
        pretty_json = base_prompt.to_json(pretty=True)
        assert pretty_json.count("\n") > json_data.count("\n")

# Response Format Tests
class TestResponseFormat:
    def test_response_format_validation(self):
        """Test response format validation."""
        # Valid format
        format1 = ResponseFormat(format="json")
        assert format1.format == "json"

        # Valid format with schema
        format2 = ResponseFormat(
            format="json",
            schema='{"type": "object"}'
        )
        assert format2.schema == '{"type": "object"}'

# Metadata Tests
class TestPromptMetadata:
    def test_metadata_validation(self, sample_response_format):
        """Test metadata validation."""
        # Valid metadata
        metadata = PromptMetadata(
            type="test",
            expected_response=sample_response_format,
            model_requirements={"model": "test-model"},
            tags=["test"],
            is_active=True
        )
        assert metadata.type == "test"
        assert metadata.is_active is True
        assert metadata.tags == ["test"]

    def test_metadata_defaults(self, sample_response_format):
        """Test metadata default values."""
        metadata = PromptMetadata(
            type="test",
            expected_response=sample_response_format
        )
        assert metadata.is_active is True
        assert metadata.tags == []
        assert metadata.model_requirements is None

class TestFileAttachments:
    """Tests for file attachment functionality."""

    def test_add_attachment(self, base_prompt, sample_image_content):
        """Test adding a file attachment."""
        base_prompt.add_attachment(
            content=sample_image_content,
            media_type=MediaType.JPEG,
            file_name="test.jpg",
            description="Test image"
        )

        assert len(base_prompt.attachments) == 1
        attachment = base_prompt.attachments[0]
        assert attachment.content == sample_image_content
        assert attachment.media_type == MediaType.JPEG
        assert attachment.file_name == "test.jpg"
        assert attachment.description == "Test image"

    def test_add_attachment_with_string_media_type(self, base_prompt, sample_image_content):
        """Test adding attachment with string media type."""
        base_prompt.add_attachment(
            content=sample_image_content,
            media_type="image/jpeg",
            file_name="test.jpg"
        )

        assert len(base_prompt.attachments) == 1
        assert base_prompt.attachments[0].media_type == MediaType.JPEG

    def test_clear_attachments(self, base_prompt, sample_file_attachment):
        """Test clearing attachments."""
        # Add two attachments
        base_prompt.add_attachment(
            content=sample_file_attachment.content,
            media_type=sample_file_attachment.media_type,
            file_name=sample_file_attachment.file_name
        )
        base_prompt.add_attachment(
            content=sample_file_attachment.content,
            media_type=MediaType.PNG,
            file_name="test2.png"
        )

        assert len(base_prompt.attachments) == 2

        # Clear attachments
        base_prompt.clear_attachments()
        assert len(base_prompt.attachments) == 0

    def test_render_with_attachments(self, base_prompt, sample_file_attachment):
        """Test rendering prompt with attachments."""
        base_prompt.add_attachment(
            content=sample_file_attachment.content,
            media_type=sample_file_attachment.media_type,
            file_name=sample_file_attachment.file_name
        )

        system, user, attachments = base_prompt.render(
            context="test context",
            name="John",
            query="help me"
        )

        assert system == "You are a test assistant. Context: test context"
        assert user == "Hello John, help me"
        assert len(attachments) == 1

        attachment = attachments[0]
        assert attachment["content"] == sample_file_attachment.content
        assert attachment["mime_type"] == str(sample_file_attachment.media_type)
        assert attachment["file_name"] == sample_file_attachment.file_name

    def test_render_with_multiple_attachments(self, base_prompt, sample_file_attachment):
        """Test rendering with multiple attachments."""
        # Add JPEG attachment
        base_prompt.add_attachment(
            content=sample_file_attachment.content,
            media_type=MediaType.JPEG,
            file_name="test1.jpg"
        )

        # Add PNG attachment
        base_prompt.add_attachment(
            content=sample_file_attachment.content,
            media_type=MediaType.PNG,
            file_name="test2.png"
        )

        _, _, attachments = base_prompt.render(
            context="test context",
            name="John",
            query="help me"
        )

        assert len(attachments) == 2
        assert attachments[0]["mime_type"] == str(MediaType.JPEG)
        assert attachments[1]["mime_type"] == str(MediaType.PNG)

    def test_attachment_validation(self, base_prompt):
        """Test validation of attachment fields."""
        with pytest.raises(ValidationError):
            # Missing required fields
            FileAttachment()

        with pytest.raises(ValidationError):
            # Invalid media type
            FileAttachment(
                content="test",
                media_type="invalid_type",
                file_name="test.txt"
            )
