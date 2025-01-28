import pytest
from unittest.mock import patch, Mock, ANY
from pathlib import Path
from src.prompts.loader import PromptLoader
import subprocess
import json
from datetime import datetime, timedelta
import tempfile
from src.prompts.loader import (
    PromptSource,
    AuthConfig,
    CacheEntry
)

MOCK_REMOTE_YAML = """
name: "remote_test_prompt"
version: "1.0.0"
description: "A test prompt loaded from remote URL"
metadata:
  type: "remote_test"
  expected_response:
    format: "json"
system_prompt: "You are a remote test assistant"
user_prompt: "This is a test prompt with {variable}"
"""

class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def test_load_remote_prompt():
    """Test loading a prompt from a remote URL."""
    with patch('subprocess.run') as mock_run:
        # Mock successful curl response
        mock_run.return_value.stdout = MOCK_REMOTE_YAML
        mock_run.return_value.stderr = ""
        mock_run.return_value.returncode = 0
        
        loader = PromptLoader(remote_sources=["https://example.com/prompts/test.yaml"])
        prompt = loader.get_prompt("remote_test")
        
        assert prompt is not None
        assert prompt.name == "remote_test_prompt"
        assert prompt.metadata.type == "remote_test"
        
        # Verify curl was called correctly
        mock_run.assert_called_once_with(
            ["curl", "-s", "-L", "https://example.com/prompts/test.yaml"],
            capture_output=True,
            text=True,
            check=True
        )

def test_add_remote_source():
    """Test adding a new remote source."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stdout = MOCK_REMOTE_YAML
        mock_run.return_value.stderr = ""
        mock_run.return_value.returncode = 0
        
        loader = PromptLoader()
        success = loader.add_remote_source("https://example.com/prompts/test.yaml")
        
        assert success is True
        assert any(str(s.url) == "https://example.com/prompts/test.yaml" 
                  for s in loader.remote_sources)
        assert loader.get_prompt("remote_test") is not None

def test_refresh_remote_sources():
    """Test refreshing remote sources."""
    with patch('subprocess.run') as mock_run:
        # Set up mock for initial load
        mock_run.return_value = Mock(
            stdout=MOCK_REMOTE_YAML,
            stderr="",
            returncode=0
        )
        
        loader = PromptLoader(remote_sources=[
            "https://example.com/prompts/test1.yaml",
            "https://example.com/prompts/test2.yaml"
        ])
        
        # Reset mock and set up for refresh
        mock_run.reset_mock()
        mock_run.return_value = Mock(
            stdout=MOCK_REMOTE_YAML,
            stderr="",
            returncode=0
        )
        
        # Clear cache to force refresh
        loader.refresh_remote_sources(force=True)
        
        assert mock_run.call_count == 2  # Just the refresh calls
        assert loader.get_prompt("remote_test") is not None

def test_failed_remote_load():
    """Test handling of failed remote prompt loads."""
    with patch('subprocess.run') as mock_run:
        # Mock failed curl response
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "curl", output=b"", stderr=b"Error"
        )
        
        loader = PromptLoader()  # Start with empty loader
        loader.add_remote_source("https://example.com/prompts/test.yaml")
        prompt = loader.get_prompt("remote_test")
        
        assert prompt is None  # Should not have loaded the prompt
        
        # Verify curl was attempted
        mock_run.assert_called_once()

def test_invalid_remote_yaml():
    """Test handling of invalid YAML from remote source."""
    with patch('subprocess.run') as mock_run:
        # Mock invalid YAML response
        mock_run.return_value = Mock(
            stdout="not a yaml file",
            stderr="",
            returncode=0
        )
        
        loader = PromptLoader()  # Start with empty loader
        loader.add_remote_source("https://example.com/prompts/test.yaml")
        prompt = loader.get_prompt("remote_test")
        
        assert prompt is None  # Should not have loaded the prompt

def test_url_validation():
    """Test URL validation for remote sources."""
    # Invalid scheme
    with pytest.raises(ValueError) as exc_info:
        PromptSource(url="ftp://example.com/test.yaml")
    assert "URL scheme should be 'http' or 'https'" in str(exc_info.value)
    
    # Invalid file extension
    with pytest.raises(ValueError) as exc_info:
        PromptSource(url="https://example.com/test.txt")
    assert "URL must point to a YAML file" in str(exc_info.value)
    
    # Valid URLs
    source = PromptSource(url="https://example.com/test.yaml")
    assert source.url is not None
    assert source.is_remote is True

def test_auth_config():
    """Test authentication configuration."""
    # Token auth
    auth = AuthConfig(token="test_token")
    args = auth.get_curl_args()
    assert "-H" in args
    assert "Authorization: Bearer test_token" in args
    
    # Basic auth
    auth = AuthConfig(username="user", password="pass")
    args = auth.get_curl_args()
    assert "-u" in args
    assert "user:pass" in args
    
    # Custom headers
    auth = AuthConfig(headers={"X-Custom": "value"})
    args = auth.get_curl_args()
    assert "-H" in args
    assert "X-Custom: value" in args

@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

def test_caching(temp_cache_dir):
    """Test prompt caching functionality."""
    url = "https://example.com/test.yaml"
    
    with patch('subprocess.run') as mock_run:
        # First request - should download
        mock_run.return_value.stdout = MOCK_REMOTE_YAML
        mock_run.return_value.stderr = "ETag: abc123"
        mock_run.return_value.returncode = 0
        
        loader = PromptLoader(
            cache_dir=temp_cache_dir,
            remote_sources=[url]
        )
        
        # Verify cache was created
        cache_path = loader._get_cache_path(url)
        assert cache_path.exists()
        
        # Verify cache content
        with open(cache_path) as f:
            cache_data = json.load(f)
            assert cache_data["content"] == MOCK_REMOTE_YAML
            assert cache_data["etag"] == "abc123"
            assert datetime.fromisoformat(cache_data["last_fetched"])
        
        # Second request - should use cache
        mock_run.reset_mock()
        loader.refresh_remote_sources()
        mock_run.assert_not_called()
        
        # Force refresh - should download again
        loader.refresh_remote_sources(force=True)
        mock_run.assert_called_once()

def test_stale_cache(temp_cache_dir):
    """Test handling of stale cache entries."""
    url = "https://example.com/test.yaml"
    
    # Create a stale cache entry
    cache_path = Path(temp_cache_dir) / f"{hash(url)}.json"
    stale_entry = CacheEntry(
        content=MOCK_REMOTE_YAML,
        last_fetched=datetime.now() - timedelta(hours=2),
        etag="old_etag"
    )
    with open(cache_path, "w") as f:
        json.dump(stale_entry.model_dump(), f, cls=DateTimeEncoder)
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stdout = MOCK_REMOTE_YAML
        mock_run.return_value.stderr = "ETag: new_etag"
        mock_run.return_value.returncode = 0
        
        loader = PromptLoader(
            cache_dir=temp_cache_dir,
            remote_sources=[url]
        )
        
        # Should download new content due to stale cache
        mock_run.assert_called_once()

def test_authenticated_request():
    """Test remote prompt loading with authentication."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stdout = MOCK_REMOTE_YAML
        mock_run.return_value.stderr = ""
        mock_run.return_value.returncode = 0
        
        auth = AuthConfig(
            token="test_token",
            headers={"X-Custom": "value"}
        )
        source = PromptSource(
            url="https://example.com/test.yaml",
            auth=auth
        )
        
        loader = PromptLoader(remote_sources=[source])
        
        # Verify curl command included auth headers
        mock_run.assert_called_once_with(
            ["curl", "-s", "-L", "-H", "Authorization: Bearer test_token",
             "-H", "X-Custom: value", str(source.url)],
            capture_output=True,
            text=True,
            check=True
        ) 