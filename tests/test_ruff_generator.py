import pytest
from pathlib import Path
import tempfile
import responses
import json
from unittest.mock import patch, Mock
from bs4 import BeautifulSoup
from tests.test_fixtures.ruff_examples import RuffExampleBuilder

# Sample HTML content for testing
SAMPLE_HTML = """
<div>
    <p>Bad:</p>
    <div class="highlight">
        <pre>
import sys
if sys.version == None:  # E711
    pass
        </pre>
    </div>
    <p>Good:</p>
    <div class="highlight">
        <pre>
import sys
if sys.version is None:
    pass
        </pre>
    </div>
</div>
"""

# Sample Ruff JSON output
SAMPLE_RUFF_OUTPUT = [
    {
        "code": "E711",
        "message": "comparison to None should be 'if cond is None:'",
        "fix": None,
        "filename": "test.py",
        "location": {
            "row": 2,
            "column": 13
        }
    }
]

@pytest.fixture
def example_builder():
    """Create a RuffExampleBuilder instance with a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = RuffExampleBuilder()
        builder.cache_dir = Path(tmpdir)
        yield builder

@responses.activate
def test_fetch_rule_page(example_builder):
    """Test fetching rule documentation page."""
    # Mock the HTTP request
    responses.add(
        responses.GET,
        "https://docs.astral.sh/ruff/rules/e711",
        body=SAMPLE_HTML,
        status=200
    )
    
    content = example_builder._fetch_rule_page("E711")
    assert content == SAMPLE_HTML
    assert len(responses.calls) == 1

@responses.activate
def test_fetch_rule_page_error(example_builder):
    """Test handling of HTTP errors when fetching rule page."""
    # Mock a failed request
    responses.add(
        responses.GET,
        "https://docs.astral.sh/ruff/rules/e999",
        status=404
    )
    
    with pytest.raises(requests.exceptions.HTTPError):
        example_builder._fetch_rule_page("E999")

def test_extract_examples(example_builder):
    """Test extracting code examples from HTML content."""
    examples = example_builder._extract_examples(SAMPLE_HTML)
    
    assert 'bad' in examples
    assert 'good' in examples
    assert "if sys.version == None:" in examples['bad']
    assert "if sys.version is None:" in examples['good']

def test_write_example_file(example_builder, tmp_path):
    """Test writing code examples to files."""
    code = "print('test')"
    file_path = example_builder._write_example_file(tmp_path, "test", code)
    
    assert file_path.exists()
    assert file_path.read_text() == code

@patch('subprocess.run')
def test_run_ruff(mock_run, example_builder, tmp_path):
    """Test running Ruff on example files."""
    # Mock successful Ruff execution
    mock_run.return_value = Mock(
        stdout=json.dumps(SAMPLE_RUFF_OUTPUT),
        stderr="",
        returncode=0
    )
    
    # Create a test file
    file_path = tmp_path / "test.py"
    file_path.write_text("if x == None: pass")
    
    has_error, violations = example_builder._run_ruff(file_path, "E711")
    assert has_error is True
    assert len(violations) == 1
    assert violations[0]["code"] == "E711"

@patch('subprocess.run')
def test_run_ruff_no_violations(mock_run, example_builder, tmp_path):
    """Test running Ruff on code with no violations."""
    # Mock Ruff execution with no violations
    mock_run.return_value = Mock(
        stdout="[]",
        stderr="",
        returncode=0
    )
    
    # Create a test file
    file_path = tmp_path / "test.py"
    file_path.write_text("if x is None: pass")
    
    has_error, violations = example_builder._run_ruff(file_path, "E711")
    assert has_error is False
    assert len(violations) == 0

@patch('subprocess.run')
def test_validate_examples(mock_run, example_builder, tmp_path):
    """Test validating examples against Ruff."""
    # Mock Ruff execution for bad example (should have violation)
    mock_run.side_effect = [
        Mock(stdout=json.dumps(SAMPLE_RUFF_OUTPUT), stderr="", returncode=0),  # Bad example
        Mock(stdout="[]", stderr="", returncode=0)  # Good example
    ]
    
    # Create example files
    examples = {
        'bad': tmp_path / "bad.py",
        'good': tmp_path / "good.py"
    }
    examples['bad'].write_text("if x == None: pass")
    examples['good'].write_text("if x is None: pass")
    
    results = example_builder.validate_examples(examples, "E711")
    assert results['bad'] is True  # Should have violation
    assert results['good'] is True  # Should not have violation

def test_build_example_with_cache(example_builder, tmp_path):
    """Test building examples with caching."""
    # First, create a cached response
    cache_file = example_builder.cache_dir / "e711.html"
    cache_file.write_text(SAMPLE_HTML)
    
    examples = example_builder.build_example(tmp_path, "E711")
    assert 'bad' in examples
    assert 'good' in examples
    assert examples['bad'].exists()
    assert examples['good'].exists()

@responses.activate
def test_build_example_without_cache(example_builder, tmp_path):
    """Test building examples without cache."""
    # Mock the HTTP request
    responses.add(
        responses.GET,
        "https://docs.astral.sh/ruff/rules/e711",
        body=SAMPLE_HTML,
        status=200
    )
    
    examples = example_builder.build_example(tmp_path, "E711")
    assert 'bad' in examples
    assert 'good' in examples
    assert examples['bad'].exists()
    assert examples['good'].exists()
    
    # Verify cache was created
    cache_file = example_builder.cache_dir / "e711.html"
    assert cache_file.exists()
    assert cache_file.read_text() == SAMPLE_HTML

@pytest.mark.parametrize('rule_codes', [
    ['E711'],  # Comparison with None
    ['F401'],  # Unused import
    ['E501'],  # Line too long
])
def test_end_to_end(ruff_examples):
    """Test end-to-end functionality with real rules."""
    for rule_code in ruff_examples:
        examples = ruff_examples[rule_code]
        assert 'bad' in examples
        assert 'good' in examples
        assert examples['bad'].exists()
        assert examples['good'].exists()
        
        # Validate file contents
        bad_content = examples['bad'].read_text()
        good_content = examples['good'].read_text()
        assert bad_content != good_content  # Examples should be different 