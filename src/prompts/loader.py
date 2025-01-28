from pathlib import Path
from typing import Dict, Optional, Union
import yaml
import subprocess
import tempfile
import json
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse
from pydantic import BaseModel, HttpUrl, Field, validator, field_validator

class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class CacheEntry(BaseModel):
    """Cache entry for a remote prompt."""
    content: str
    last_fetched: datetime
    etag: Optional[str] = None
    
    @property
    def is_stale(self) -> bool:
        """Check if the cache entry is older than 1 hour."""
        return datetime.now() - self.last_fetched > timedelta(hours=1)

class AuthConfig(BaseModel):
    """Authentication configuration for remote sources."""
    token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    
    def get_curl_args(self) -> list[str]:
        """Get curl arguments for authentication."""
        args = []
        if self.token:
            args.extend(["-H", f"Authorization: Bearer {self.token}"])
        elif self.username and self.password:
            args.extend(["-u", f"{self.username}:{self.password}"])
        for key, value in self.headers.items():
            args.extend(["-H", f"{key}: {value}"])
        return args

class ResponseFormat(BaseModel):
    format: str
    schema: Optional[str] = None

class PromptMetadata(BaseModel):
    type: str
    expected_response: ResponseFormat
    model_requirements: Optional[Dict] = None

class GitCommitInfo(BaseModel):
    """Git commit information."""
    commit: str
    author: str

class GitMetadata(BaseModel):
    """Git metadata for tracking prompt changes."""
    created: GitCommitInfo
    last_modified: GitCommitInfo

class PromptTemplate(BaseModel):
    """A prompt template with metadata."""
    name: str
    version: str
    description: str
    author: Optional[str] = None
    git_metadata: Optional[GitMetadata] = None
    metadata: PromptMetadata
    system_prompt: str
    user_prompt: str
    examples: Optional[list] = None

class PromptSource(BaseModel):
    """Source configuration for a prompt template."""
    url: Optional[HttpUrl] = None
    path: Optional[Path] = None
    auth: Optional[AuthConfig] = None
    
    @property
    def is_remote(self) -> bool:
        return self.url is not None
    
    @field_validator("url")
    @classmethod
    def validate_url(cls, v: Optional[HttpUrl]) -> Optional[HttpUrl]:
        """Validate URL format and scheme."""
        if v:
            scheme = urlparse(str(v)).scheme
            if scheme not in ("http", "https"):
                raise ValueError("URL must use http or https scheme")
            if not str(v).endswith((".yaml", ".yml")):
                raise ValueError("URL must point to a YAML file")
        return v

class PromptLoader:
    """Loads and manages prompt templates from YAML files and URLs."""
    
    def __init__(
        self,
        prompts_dir: Optional[str] = None,
        remote_sources: Optional[list[Union[str, PromptSource]]] = None,
        cache_dir: Optional[str] = None
    ):
        if prompts_dir is None:
            # Default to the prompts directory in the package
            package_dir = Path(__file__).parent
            prompts_dir = str(package_dir / "tasks")
        self.prompts_dir = Path(prompts_dir)
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = str(Path.home() / ".llm_orchestrator" / "prompt_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize remote sources with proper configuration
        self.remote_sources: list[PromptSource] = []
        if remote_sources:
            for source in remote_sources:
                if isinstance(source, str):
                    self.remote_sources.append(PromptSource(url=source))
                else:
                    self.remote_sources.append(source)
        
        self.prompts: Dict[str, PromptTemplate] = {}
        self._load_prompts()
    
    def _get_cache_path(self, url: str) -> Path:
        """Get the cache file path for a URL."""
        url_hash = str(hash(url))
        return self.cache_dir / f"{url_hash}.json"
    
    def _load_from_cache(self, url: str) -> Optional[str]:
        """Load prompt content from cache if available and not stale."""
        cache_path = self._get_cache_path(url)
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path) as f:
                cache_entry = CacheEntry(**json.load(f))
                if not cache_entry.is_stale:
                    return cache_entry.content
        except Exception:
            return None
        return None
    
    def _save_to_cache(self, url: str, content: str, etag: Optional[str] = None) -> None:
        """Save prompt content to cache."""
        cache_path = self._get_cache_path(url)
        cache_entry = CacheEntry(
            content=content,
            last_fetched=datetime.now(),
            etag=etag
        )
        with open(cache_path, "w") as f:
            json.dump(cache_entry.model_dump(), f, cls=DateTimeEncoder)
    
    def _load_prompts(self) -> None:
        """Load all prompt templates from local files and remote sources."""
        # Load local prompts
        for yaml_file in self.prompts_dir.glob("*.yaml"):
            self._load_local_prompt(yaml_file)
        
        # Load remote prompts
        for source in self.remote_sources:
            self._load_remote_prompt(source)
    
    def _load_local_prompt(self, file_path: Path) -> None:
        """Load a prompt template from a local file."""
        try:
            with open(file_path) as f:
                data = yaml.safe_load(f)
                prompt = PromptTemplate(**data)
                self.prompts[prompt.metadata.type] = prompt
        except Exception as e:
            print(f"Error loading prompt from {file_path}: {e}")
    
    def _load_remote_prompt(self, source: PromptSource) -> None:
        """Load a prompt template from a remote URL using curl."""
        if not source.url:
            return
            
        url = str(source.url)
        prompt_type = None
        
        # Try loading from cache first
        cached_content = self._load_from_cache(url)
        if cached_content:
            try:
                data = yaml.safe_load(cached_content)
                prompt = PromptTemplate(**data)
                self.prompts[prompt.metadata.type] = prompt
                return
            except Exception:
                # If cache is corrupted, continue with fresh download
                pass
        
        try:
            # Build curl command with authentication if provided
            curl_args = ["curl", "-s", "-L"]  # -L to follow redirects
            if source.auth:
                curl_args.extend(source.auth.get_curl_args())
            
            # Add URL
            curl_args.append(url)
            
            # Download the YAML file
            result = subprocess.run(
                curl_args,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Get ETag if available
            etag = None
            for line in result.stderr.splitlines():
                if line.startswith("ETag:"):
                    etag = line.split(":", 1)[1].strip()
            
            # Parse and validate the YAML content
            try:
                data = yaml.safe_load(result.stdout)
                if not data or not isinstance(data, dict):
                    raise ValueError("Invalid YAML content")
                    
                # Extract type before creating prompt
                if "metadata" in data:
                    metadata = data.get("metadata", {})
                    if isinstance(metadata, dict):
                        prompt_type = metadata.get("type")
                
                # Validate required fields
                required_fields = ["name", "version", "description", "metadata", 
                                 "system_prompt", "user_prompt"]
                missing_fields = [f for f in required_fields if f not in data]
                if missing_fields:
                    raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
                
                prompt = PromptTemplate(**data)
                self.prompts[prompt.metadata.type] = prompt
                
                # Cache the result
                self._save_to_cache(url, result.stdout, etag)
            except Exception as e:
                print(f"Error parsing prompt from {url}: {e}")
                if prompt_type and prompt_type in self.prompts:
                    del self.prompts[prompt_type]
                
        except subprocess.CalledProcessError as e:
            print(f"Error downloading prompt from {url}: {e}")
            if prompt_type and prompt_type in self.prompts:
                del self.prompts[prompt_type]
    
    def add_remote_source(
        self,
        url: str,
        auth: Optional[AuthConfig] = None
    ) -> bool:
        """Add and load a new remote prompt source with optional authentication."""
        source = PromptSource(url=url, auth=auth)
        if not any(s.url == source.url for s in self.remote_sources):
            self.remote_sources.append(source)
            self._load_remote_prompt(source)
            return True
        return False
    
    def refresh_remote_sources(self, force: bool = False) -> None:
        """Refresh all remote sources, optionally forcing cache invalidation."""
        if force:
            # Clear the cache if force refresh is requested
            for file in self.cache_dir.glob("*.json"):
                file.unlink()
            # Clear the prompts dictionary
            self.prompts.clear()
        
        # Store current prompts to detect changes
        old_prompts = self.prompts.copy()
        
        # Attempt to load each source
        for source in self.remote_sources:
            self._load_remote_prompt(source)
            
        # Remove prompts that weren't refreshed
        for type_, prompt in old_prompts.items():
            if type_ not in self.prompts:
                self.prompts[type_] = prompt
    
    def get_prompt(self, task_type: str) -> Optional[PromptTemplate]:
        """Get a prompt template by task type."""
        return self.prompts.get(task_type)
    
    def format_prompt(self, task_type: str, **kwargs) -> tuple[Optional[str], Optional[str]]:
        """Format a prompt template with the given variables."""
        prompt = self.get_prompt(task_type)
        if not prompt:
            return None, None
            
        try:
            system_prompt = prompt.system_prompt
            user_prompt = prompt.user_prompt.format(**kwargs)
            return system_prompt, user_prompt
        except KeyError as e:
            raise ValueError(f"Missing required variable for prompt: {e}")
    
    def validate_response(self, task_type: str, response: Dict) -> bool:
        """Validate that a response matches the expected schema."""
        prompt = self.get_prompt(task_type)
        if not prompt or not prompt.metadata.expected_response.schema:
            return True
            
        # TODO: Implement JSON schema validation
        return True 