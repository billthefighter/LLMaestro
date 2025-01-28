from pathlib import Path
import os
import yaml
from typing import Dict, Optional
from pydantic import BaseModel, Field

class APICredentials(BaseModel):
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    azure_api_key: Optional[str] = Field(None, description="Azure OpenAI API key")
    azure_endpoint: Optional[str] = Field(None, description="Azure OpenAI endpoint")

class Config(BaseModel):
    api_credentials: APICredentials = Field(default_factory=lambda: APICredentials(
        openai_api_key=None,
        anthropic_api_key=None,
        azure_api_key=None,
        azure_endpoint=None
    ))
    default_model: str = Field("gpt-4", description="Default LLM model to use")
    storage_path: str = Field("chain_storage", description="Path to store chain execution data")
    
    @classmethod
    def load_from_file(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from a YAML file."""
        if config_path is None:
            config_path = os.path.expanduser("~/.llm_orchestrator/config.yaml")
        
        path = Path(config_path)
        if not path.exists():
            return cls(
                api_credentials=APICredentials(
                    openai_api_key=None,
                    anthropic_api_key=None,
                    azure_api_key=None,
                    azure_endpoint=None
                ),
                default_model="gpt-4",
                storage_path="chain_storage"
            )
        
        with path.open('r') as f:
            config_data = yaml.safe_load(f)
            return cls.model_validate(config_data)
    
    def save_to_file(self, config_path: Optional[str] = None) -> None:
        """Save configuration to a YAML file."""
        if config_path is None:
            config_path = os.path.expanduser("~/.llm_orchestrator/config.yaml")
        
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with path.open('w') as f:
            yaml.safe_dump(self.model_dump(), f)

# Global configuration instance
_config: Optional[Config] = None

def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.load_from_file(config_path)
    return _config

def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config 