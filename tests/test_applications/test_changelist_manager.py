"""Tests for the Changelist Manager application."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest
from pydantic import BaseModel

from src.applications.changelistmanager.app import (
    ChangelistEntry,
    ChangelistManager,
    ChangelistResponse,
)
from src.llm.interfaces import LLMResponse


# Test Data Fixtures
@pytest.fixture
def sample_diff_content() -> str:
    """Sample git diff content for testing."""
    return """
diff --git a/src/core/config.py b/src/core/config.py
index 123..456 789
--- a/src/core/config.py
+++ b/src/core/config.py
@@ -1,3 +1,4 @@
+from typing import Dict, Any
 from pydantic import BaseModel

-class Config:
+class Config(BaseModel):
     \"\"\"Configuration class.\"\"\"
-    def __init__(self):
-        self.value = 42
+    values: Dict[str, Any] = {}
"""


@pytest.fixture
def sample_changed_files() -> List[str]:
    """Sample list of changed files."""
    return [
        "src/core/config.py",
        "README.md",
        "docs/api/config.md",
        "tests/test_core/test_config.py",
    ]


@pytest.fixture
def mock_llm_response() -> Dict[str, Union[str, List[str], bool, Dict[str, str]]]:
    """Mock LLM response for testing."""
    return {
        "summary": "Updated Config class to use Pydantic BaseModel",
        "affected_readmes": ["README.md", "docs/api/config.md"],
        "needs_readme_updates": True,
        "suggested_updates": {
            "README.md": "Update configuration section to reflect new Pydantic model usage",
            "docs/api/config.md": "Add documentation for new Dict-based configuration",
        },
    }


@pytest.fixture
def mock_llm_interface(mock_llm_response: Dict[str, Any]):
    """Mock LLM interface for testing."""
    class MockLLM:
        async def process(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
            return LLMResponse(
                content=json.dumps(mock_llm_response),
                metadata={"tokens": 100, "model": "test-model"},
            )
    return MockLLM()


@pytest.fixture
def temp_changelist(tmp_path: Path) -> Path:
    """Create a temporary changelist file."""
    changelist = tmp_path / "changelist.md"
    changelist.write_text("# Changelist\n\n")
    return changelist


# Unit Tests
@pytest.mark.unit
def test_get_readme_files():
    """Test README file detection."""
    files = [
        "src/README.md",
        "docs/readme.md",
        "test.py",
        "path/to/README.md",
    ]
    manager = ChangelistManager()
    readmes = manager.get_readme_files(files)

    assert len(readmes) == 3
    assert "src/README.md" in readmes
    assert "docs/readme.md" in readmes
    assert "path/to/README.md" in readmes


@pytest.mark.unit
def test_changelist_entry_model():
    """Test ChangelistEntry model validation."""
    entry = ChangelistEntry(
        summary="Test changes",
        files_changed=["file1.py", "file2.py"],
        timestamp="2024-02-05 12:34:56",
    )
    assert entry.summary == "Test changes"
    assert len(entry.files_changed) == 2
    assert entry.timestamp == "2024-02-05 12:34:56"


@pytest.mark.unit
def test_changelist_response_model():
    """Test ChangelistResponse model validation."""
    response = ChangelistResponse(
        summary="Test summary",
        affected_readmes=["README.md"],
        needs_readme_updates=True,
        suggested_updates={"README.md": "Update required"},
    )
    assert response.summary == "Test summary"
    assert response.needs_readme_updates is True
    assert response.suggested_updates is not None
    assert "README.md" in response.suggested_updates


# Integration Tests
@pytest.mark.asyncio
@pytest.mark.integration
async def test_process_changes(
    mock_llm_interface,
    sample_diff_content: str,
    mock_llm_response: Dict[str, Any],
    monkeypatch,
):
    """Test processing changes with mock LLM."""
    manager = ChangelistManager()
    monkeypatch.setattr(manager, "llm", mock_llm_interface)

    result = await manager.process_changes(sample_diff_content)

    assert isinstance(result, ChangelistResponse)
    assert result.summary == mock_llm_response["summary"]
    assert result.needs_readme_updates == mock_llm_response["needs_readme_updates"]
    assert result.affected_readmes == mock_llm_response["affected_readmes"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_update_changelist_file(temp_changelist: Path, monkeypatch):
    """Test updating changelist file."""
    # Patch the changelist path
    monkeypatch.setattr(Path, "cwd", lambda: temp_changelist.parent)

    manager = ChangelistManager()
    entry = ChangelistEntry(
        summary="Test changes",
        files_changed=["file1.py", "file2.py"],
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    await manager.update_changelist_file(entry)

    content = temp_changelist.read_text()
    assert "# Changelist" in content
    assert "Test changes" in content
    assert "file1.py" in content
    assert "file2.py" in content


@pytest.mark.asyncio
@pytest.mark.integration
async def test_validate_readmes(mock_llm_interface, monkeypatch):
    """Test README validation."""
    manager = ChangelistManager()
    monkeypatch.setattr(manager, "llm", mock_llm_interface)

    readmes = ["README.md", "docs/api/config.md"]
    changes = "Updated configuration system"

    try:
        result = await manager.validate_readmes(readmes, changes)
        assert isinstance(result, dict)
        # Check if any key in the result contains README.md
        assert any("README.md" in str(key) for key in result.keys())
    except Exception as e:
        pytest.fail(f"Validate readmes failed: {str(e)}")


# Error Cases
@pytest.mark.asyncio
@pytest.mark.unit
async def test_process_changes_invalid_diff(mock_llm_interface, monkeypatch):
    """Test processing invalid diff content."""
    manager = ChangelistManager()
    monkeypatch.setattr(manager, "llm", mock_llm_interface)

    with pytest.raises(Exception):
        await manager.process_changes("")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_update_changelist_invalid_entry():
    """Test updating changelist with invalid entry."""
    manager = ChangelistManager()

    with pytest.raises(ValueError):
        invalid_entry = {"summary": "Test"}  # type: ignore
        await manager.update_changelist_file(invalid_entry)  # type: ignore


# Performance Tests
@pytest.mark.slow
@pytest.mark.parametrize(
    "diff_size",
    [
        pytest.param(100, id="small_diff"),
        pytest.param(1000, id="medium_diff"),
        pytest.param(5000, id="large_diff"),
    ],
)
@pytest.mark.asyncio
async def test_process_changes_performance(
    diff_size: int,
    mock_llm_interface,
    monkeypatch,
):
    """Test performance with different diff sizes."""
    manager = ChangelistManager()
    monkeypatch.setattr(manager, "llm", mock_llm_interface)

    # Generate diff content of specified size
    diff_content = "+" * diff_size

    result = await manager.process_changes(diff_content)
    assert isinstance(result, ChangelistResponse)
