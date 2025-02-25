"""Secure credential management for API keys and other sensitive data."""

from typing import Dict, Optional
from datetime import datetime

from pydantic import BaseModel, Field

from llmaestro.llm.models import ModelFamily

class APIKeyFormat(BaseModel):
    """Defines the format and validation rules for an API key."""
    
    pattern: str = Field(..., description="Regex pattern for key validation")
    prefix: Optional[str] = Field(default=None, description="Expected key prefix")
    min_length: int = Field(default=32, description="Minimum key length")
    max_length: Optional[int] = Field(default=None, description="Maximum key length")
    
    def validate(self, key: str) -> bool:
        """Validate a key against this format."""
        import re
        if self.prefix and not key.startswith(self.prefix):
            return False
        if len(key) < self.min_length:
            return False
        if self.max_length and len(key) > self.max_length:
            return False
        return bool(re.match(self.pattern, key))

class APIKey(BaseModel):
    """Represents a single API key with metadata."""
    
    family: ModelFamily = Field(..., description="Model family this key belongs to")
    key: str = Field(..., description="The actual API key value")
    is_encrypted: bool = Field(default=False, description="Whether the key is encrypted")
    created_at: datetime = Field(default_factory=datetime.now, description="When this key was first added")
    last_used: Optional[datetime] = Field(default=None, description="When this key was last used")
    last_rotated: Optional[datetime] = Field(default=None, description="When this key was last rotated")
    expiration: Optional[datetime] = Field(default=None, description="When this key expires")
    description: Optional[str] = Field(default=None, description="Optional description or notes")
    
    # Define key formats for known providers
    KEY_FORMATS: Dict[ModelFamily, APIKeyFormat] = {
        ModelFamily.ANTHROPIC: APIKeyFormat(
            pattern=r"^sk-ant-[A-Za-z0-9]{32,}$",
            prefix="sk-ant-",
            min_length=40
        ),
        ModelFamily.OPENAI: APIKeyFormat(
            pattern=r"^sk-[A-Za-z0-9]{32,}$",
            prefix="sk-",
            min_length=40
        ),
        ModelFamily.GOOGLE: APIKeyFormat(
            pattern=r"^[A-Za-z0-9\-_]{32,}$",
            min_length=32
        )
    }
    
    @validator('key')
    def validate_key_format(cls, v: str, values: Dict) -> str:
        """Validate the key format based on model family."""
        if not v.strip():
            raise ValueError("API key cannot be empty")
            
        # Get family from values
        family = values.get('family')
        if not family:
            return v  # Skip validation if no family specified
            
        # Get format rules for this family
        key_format = cls.KEY_FORMATS.get(family)
        if key_format and not key_format.validate(v):
            raise ValueError(f"Invalid API key format for model family {family.value}")
            
        return v
    
    def mark_used(self) -> None:
        """Mark this key as being used now."""
        self.last_used = datetime.now()

    
    def is_expired(self) -> bool:
        """Check if this key has expired."""
        if not self.expiration:
            return False
        return datetime.now() > self.expiration

