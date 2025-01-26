from pathlib import Path
from typing import Dict, Optional
import yaml
from pydantic import BaseModel

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

class PromptLoader:
    """Loads and manages prompt templates from YAML files."""
    
    def __init__(self, prompts_dir: Optional[str] = None):
        if prompts_dir is None:
            # Default to the prompts directory in the package
            package_dir = Path(__file__).parent
            prompts_dir = str(package_dir / "tasks")
        self.prompts_dir = Path(prompts_dir)
        self.prompts: Dict[str, PromptTemplate] = {}
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """Load all prompt templates from the prompts directory."""
        for yaml_file in self.prompts_dir.glob("*.yaml"):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
                prompt = PromptTemplate(**data)
                self.prompts[prompt.metadata.type] = prompt
    
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