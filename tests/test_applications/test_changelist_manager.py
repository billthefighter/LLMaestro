"""Tests for the Changelist Manager application."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest
from pydantic import BaseModel

from llmaestro.applications.changelistmanager.app import (
    ChangelistEntry,
    ChangelistManager,
    ChangelistResponse,
)
from llmaestro.llm.interfaces import LLMResponse
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.llm.models import LLMProfile, ModelFamily, LLMCapabilities
from llmaestro.llm.llm_registry import LLMRegistry
from tests.test_applications.test_utils import MockLLM


# Test Data Fixtures
@pytest.fixture
def mock_llm_registry() -> LLMRegistry:
    """Create a mock model registry with test models."""
    registry = LLMRegistry()
    registry.register(
        LLMProfile(
            name="claude-3-sonnet-20240229",
            family=ModelFamily.CLAUDE,
            capabilities=LLMCapabilities(
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                max_context_window=200000,
                input_cost_per_1k_tokens=0.015,
                output_cost_per_1k_tokens=0.015,
            ),
        )
    )
    return registry


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
def mock_llm_interface(mock_llm_response: Dict[str, Any], mock_llm_registry: LLMRegistry) -> BaseLLMInterface:
    """Create a mock LLM interface for testing."""
    return MockLLM(mock_llm_registry, mock_llm_response)


@pytest.fixture
def temp_changelist(tmp_path: Path) -> Path:
    """Create a temporary changelist file."""
    changelist = tmp_path / "changelist.md"
    changelist.write_text("# Changelist\n\n")
    return changelist


# Unit Tests
@pytest.mark.unit
def test_get_readme_files(mock_llm_interface):
    """Test README file detection."""
    files = [
        "src/README.md",
        "docs/readme.md",
        "test.py",
        "path/to/README.md",
    ]
    manager = ChangelistManager(llm_interface=mock_llm_interface)
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
    manager = ChangelistManager(llm_interface=mock_llm_interface)
    response = await manager.process_changes(sample_diff_content)

    assert isinstance(response, ChangelistResponse)
    assert response.summary == mock_llm_response["summary"]
    assert response.affected_readmes == mock_llm_response["affected_readmes"]
    assert response.needs_readme_updates == mock_llm_response["needs_readme_updates"]
    assert response.suggested_updates == mock_llm_response["suggested_updates"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_update_changelist_file(temp_changelist: Path, monkeypatch, mock_llm_interface):
    """Test updating changelist file."""
    # Patch the changelist path
    monkeypatch.setattr(Path, "cwd", lambda: temp_changelist.parent)

    manager = ChangelistManager(llm_interface=mock_llm_interface)
    entry = ChangelistEntry(
        summary="Test change",
        files_changed=["test.py"],
        timestamp=datetime.now().isoformat(),
    )

    await manager.update_changelist_file(entry)

    content = temp_changelist.read_text()
    assert "Test change" in content
    assert "test.py" in content


@pytest.mark.asyncio
@pytest.mark.integration
async def test_validate_readmes(mock_llm_interface, monkeypatch):
    """Test README validation."""
    manager = ChangelistManager(llm_interface=mock_llm_interface)
    readmes = ["README.md", "docs/api/README.md"]
    changes = "Updated configuration system"

    validation_results = await manager.validate_readmes(readmes, changes)

    assert isinstance(validation_results, dict)
    assert all(isinstance(k, str) and isinstance(v, bool) for k, v in validation_results.items())


# Error Cases
@pytest.mark.asyncio
@pytest.mark.unit
async def test_process_changes_invalid_diff(mock_llm_interface, monkeypatch):
    """Test processing invalid diff content."""
    manager = ChangelistManager(llm_interface=mock_llm_interface)
    response = await manager.process_changes("Invalid diff content")

    assert isinstance(response, ChangelistResponse)
    assert response.summary
    assert isinstance(response.affected_readmes, list)
    assert isinstance(response.needs_readme_updates, bool)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_update_changelist_invalid_entry():
    """Test updating changelist with invalid entry."""
    manager = ChangelistManager(llm_interface=mock_llm_interface)  # type: ignore
    with pytest.raises(ValueError):
        await manager.update_changelist_file(None)  # type: ignore


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
    manager = ChangelistManager(llm_interface=mock_llm_interface)
    diff_content = "+" * diff_size + "\n-" * diff_size

    response = await manager.process_changes(diff_content)
    assert isinstance(response, ChangelistResponse)
    assert response.summary
