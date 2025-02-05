"""Tests for the update_readme_badges.py script."""
import pytest
from pathlib import Path
import json
import tempfile
import shutil
import os
from scripts.update_readme_badges import (
    load_test_results,
    group_models_by_provider,
    generate_badge_section,
    update_readme
)

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create test-results directory and sample results
        results_dir = workspace / "test-results"
        results_dir.mkdir()

        test_results = {
            "claude-3-sonnet": True,
            "gpt-4-turbo-preview": True,
            "gpt-3.5-turbo": False
        }

        with open(results_dir / "model_connectivity.json", "w") as f:
            json.dump(test_results, f)

        # Create sample README.md
        readme_content = """# LLMaestro

Some description here.

## Model Status

### Anthropic Models
![Old Claude Model](https://example.com/old-badge.json)

### OpenAI Models
![Old GPT Model](https://example.com/old-badge.json)

## Other Sections
Other content here.
"""

        with open(workspace / "README.md", "w") as f:
            f.write(readme_content)

        # Change to workspace directory
        original_cwd = os.getcwd()
        try:
            os.chdir(str(workspace))
            yield workspace
        finally:
            os.chdir(original_cwd)

def test_load_test_results(temp_workspace):
    """Test loading test results from JSON file."""
    results = load_test_results()
    assert isinstance(results, dict)
    assert "claude-3-sonnet" in results
    assert "gpt-4-turbo-preview" in results
    assert "gpt-3.5-turbo" in results

def test_group_models_by_provider():
    """Test grouping models by provider."""
    test_results = {
        "claude-3-sonnet": True,
        "gpt-4-turbo-preview": True,
        "gpt-3.5-turbo": False
    }

    grouped = group_models_by_provider(test_results)

    assert "anthropic" in grouped
    assert "openai" in grouped
    assert len(grouped["anthropic"]) == 1
    assert len(grouped["openai"]) == 2
    assert grouped["anthropic"][0][0] == "claude-3-sonnet"
    assert "gpt-4-turbo-preview" in [model[0] for model in grouped["openai"]]

def test_generate_badge_section():
    """Test generating markdown for badge section."""
    models = [
        ("claude-3-sonnet", True),
        ("claude-2", False)
    ]

    section = generate_badge_section(models)
    assert "![claude-3-sonnet]" in section
    assert "![claude-2]" in section
    assert "https://img.shields.io/endpoint" in section
    assert "main/docs/badges/" in section

def test_update_readme(temp_workspace):
    """Test updating README.md with new badge sections."""
    # Run the update
    results = load_test_results()
    update_readme(results)

    # Read updated README
    with open("README.md") as f:
        updated_content = f.read()

    # Check that new badges are present
    assert "![claude-3-sonnet]" in updated_content
    assert "![gpt-4-turbo-preview]" in updated_content
    assert "![gpt-3.5-turbo]" in updated_content

    # Check that old badges are removed
    assert "![Old Claude Model]" not in updated_content
    assert "![Old GPT Model]" not in updated_content

    # Check that other sections are preserved
    assert "# LLMaestro" in updated_content
    assert "## Other Sections" in updated_content
    assert "Other content here." in updated_content

def test_update_readme_no_existing_sections(temp_workspace):
    """Test updating README.md when sections don't exist."""
    # Create README without model sections
    with open("README.md", "w") as f:
        f.write("""# LLMaestro

## Model Status

## Other Sections
Other content here.
""")

    # Run the update
    results = load_test_results()
    update_readme(results)

    # Read updated README
    with open("README.md") as f:
        updated_content = f.read()

    # Check that new sections were added
    assert "### Anthropic Models" in updated_content
    assert "### OpenAI Models" in updated_content
    assert "![claude-3-sonnet]" in updated_content
    assert "![gpt-4-turbo-preview]" in updated_content

    # Check that other content is preserved
    assert "# LLMaestro" in updated_content
    assert "## Other Sections" in updated_content
    assert "Other content here." in updated_content
