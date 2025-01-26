import pytest
from jsonschema.exceptions import ValidationError
from scripts.update_prompt_metadata import validate_prompt

# Valid test prompt with all required fields
VALID_PROMPT = {
    "name": "test_prompt",
    "version": "1.0.0",
    "description": "A test prompt",
    "metadata": {
        "type": "pdf_analysis",
        "expected_response": {
            "format": "json"
        }
    },
    "system_prompt": "You are a test assistant",
    "user_prompt": "This is a test prompt with {variable}"
}

# Test cases for invalid prompts
INVALID_PROMPTS = [
    (
        {  # Missing required field 'name'
            "version": "1.0.0",
            "description": "Test prompt",
            "metadata": {
                "type": "pdf_analysis",
                "expected_response": {"format": "json"}
            },
            "system_prompt": "Test",
            "user_prompt": "Test"
        },
        "'name' is a required property"
    ),
    (
        {  # Invalid version format
            "name": "test_prompt",
            "version": "1.0",  # Missing patch version
            "description": "Test prompt",
            "metadata": {
                "type": "pdf_analysis",
                "expected_response": {"format": "json"}
            },
            "system_prompt": "Test",
            "user_prompt": "Test"
        },
        "'1.0' does not match"
    ),
    (
        {  # Invalid task type
            "name": "test_prompt",
            "version": "1.0.0",
            "description": "Test prompt",
            "metadata": {
                "type": "invalid_type",
                "expected_response": {"format": "json"}
            },
            "system_prompt": "Test",
            "user_prompt": "Test"
        },
        "is not one of"
    ),
    (
        {  # Invalid response format
            "name": "test_prompt",
            "version": "1.0.0",
            "description": "Test prompt",
            "metadata": {
                "type": "pdf_analysis",
                "expected_response": {"format": "invalid_format"}
            },
            "system_prompt": "Test",
            "user_prompt": "Test"
        },
        "is not one of"
    )
]

def test_valid_prompt():
    """Test that a valid prompt passes validation."""
    try:
        validate_prompt(VALID_PROMPT)
    except ValidationError as e:
        pytest.fail(f"Valid prompt failed validation: {e}")

@pytest.mark.parametrize("invalid_prompt,expected_error", INVALID_PROMPTS)
def test_invalid_prompts(invalid_prompt, expected_error):
    """Test that invalid prompts fail validation with expected errors."""
    with pytest.raises(ValidationError) as exc_info:
        validate_prompt(invalid_prompt)
    assert expected_error in str(exc_info.value)

def test_valid_prompt_with_optional_fields():
    """Test that a valid prompt with optional fields passes validation."""
    prompt = VALID_PROMPT.copy()
    prompt.update({
        "author": "Test Author",
        "git_metadata": {
            "created": {
                "commit": "abc123",
                "author": "Test Author"
            },
            "last_modified": {
                "commit": "def456",
                "author": "Test Author"
            }
        },
        "metadata": {
            "type": "pdf_analysis",
            "expected_response": {
                "format": "json",
                "schema": '{"type": "object"}'
            },
            "model_requirements": {
                "min_tokens": 1000,
                "preferred_models": ["gpt-4", "claude-2"]
            }
        },
        "examples": [
            {
                "input": {"variable": "test"},
                "expected_output": '{"result": "test"}'
            }
        ]
    })
    try:
        validate_prompt(prompt)
    except ValidationError as e:
        pytest.fail(f"Valid prompt with optional fields failed validation: {e}")

def test_invalid_name_format():
    """Test that invalid name formats fail validation."""
    prompt = VALID_PROMPT.copy()
    prompt["name"] = "Invalid Name!"  # Contains space and special character
    with pytest.raises(ValidationError) as exc_info:
        validate_prompt(prompt)
    assert "'Invalid Name!' does not match" in str(exc_info.value)

def test_invalid_git_metadata():
    """Test that invalid git metadata structure fails validation."""
    prompt = VALID_PROMPT.copy()
    prompt["git_metadata"] = {
        "created": {
            "commit": "abc123"
            # Missing required 'author' field
        },
        "last_modified": {
            "commit": "def456",
            "author": "Test Author"
        }
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_prompt(prompt)
    assert "'author' is a required property" in str(exc_info.value) 