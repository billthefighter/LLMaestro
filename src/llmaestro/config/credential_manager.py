"""Secure credential management for API keys and other sensitive data."""

from typing import Dict, Optional, List
from pathlib import Path
import base64
from datetime import datetime

from cryptography.fernet import Fernet
from pydantic import BaseModel, Field, validator

from llmaestro.llm.provider_registry import Provider
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
        ModelFamily.CLAUDE: APIKeyFormat(
            pattern=r"^sk-ant-[A-Za-z0-9]{32,}$",
            prefix="sk-ant-",
            min_length=40
        ),
        ModelFamily.GPT: APIKeyFormat(
            pattern=r"^sk-[A-Za-z0-9]{32,}$",
            prefix="sk-",
            min_length=40
        ),
        ModelFamily.GEMINI: APIKeyFormat(
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
    
    @classmethod
    def from_provider(cls, provider: Provider, key: str, **kwargs) -> "APIKey":
        """Create an APIKey instance from a Provider."""
        return cls(
            family=ModelFamily.from_provider(provider.name),
            key=key,
            **kwargs
        )
    
    def mark_used(self) -> None:
        """Mark this key as being used now."""
        self.last_used = datetime.now()
    
    def rotate(self, new_key: str) -> None:
        """Rotate this key with a new value."""
        # Validate new key format
        self.validate_key_format(new_key, {"family": self.family})
        self.key = new_key
        self.last_rotated = datetime.now()
    
    def is_expired(self) -> bool:
        """Check if this key has expired."""
        if not self.expiration:
            return False
        return datetime.now() > self.expiration

class CredentialManager(BaseModel):
    """Manages API keys and credentials securely."""
    
    _api_keys: List[APIKey] = Field(default_factory=list)
    _encryption_key: Optional[bytes] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        # Initialize encryption key if encryption is needed
        if not self._encryption_key:
            self._encryption_key = self._get_or_create_key()
    
    def _get_or_create_key(self) -> bytes:
        """Get existing or create new encryption key."""
        key_file = Path.home() / ".llmaestro" / "credentials.key"
        if key_file.exists():
            with open(key_file, "rb") as f:
                return base64.urlsafe_b64decode(f.read())
        
        # Create new key
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key = Fernet.generate_key()
        with open(key_file, "wb") as f:
            f.write(base64.urlsafe_b64encode(key))
        return key
    
    def _encrypt_key(self, api_key: str) -> str:
        """Encrypt an API key."""
        f = Fernet(self._encryption_key)
        return f.encrypt(api_key.encode()).decode()
    
    def _decrypt_key(self, encrypted_key: str) -> str:
        """Decrypt an API key."""
        f = Fernet(self._encryption_key)
        return f.decrypt(encrypted_key.encode()).decode()
    
    def add_credential(
        self, 
        provider: Provider, 
        api_key: str, 
        encrypt: bool = True,
        description: Optional[str] = None,
        expiration: Optional[datetime] = None
    ) -> None:
        """Add a credential with optional encryption."""
        # Remove any existing key for this family
        family = ModelFamily.from_provider(provider.name)
        self._api_keys = [k for k in self._api_keys if k.family != family]
        
        # Encrypt if needed
        key_value = self._encrypt_key(api_key) if encrypt else api_key
        
        # Create new API key
        api_key = APIKey.from_provider(
            provider=provider,
            key=key_value,
            is_encrypted=encrypt,
            description=description,
            expiration=expiration
        )
        
        self._api_keys.append(api_key)
    
    def get_credential(self, provider: Provider) -> Optional[str]:
        """Get a credential, decrypting if necessary."""
        family = ModelFamily.from_provider(provider.name)
        api_key = next((k for k in self._api_keys if k.family == family), None)
        if not api_key:
            return None
            
        if api_key.is_expired():
            return None
            
        api_key.mark_used()
        return self._decrypt_key(api_key.key) if api_key.is_encrypted else api_key.key
    
    def has_credential(self, provider: Provider) -> bool:
        """Check if valid credentials exist for a provider."""
        family = ModelFamily.from_provider(provider.name)
        api_key = next((k for k in self._api_keys if k.family == family), None)
        return api_key is not None and not api_key.is_expired()
    
    def list_families(self) -> List[ModelFamily]:
        """List all model families with valid credentials."""
        return [k.family for k in self._api_keys if not k.is_expired()]
    
    def rotate_credential(
        self, 
        provider: Provider, 
        new_key: str, 
        encrypt: bool = True,
        description: Optional[str] = None
    ) -> None:
        """Rotate an existing credential with a new key."""
        family = ModelFamily.from_provider(provider.name)
        api_key = next((k for k in self._api_keys if k.family == family), None)
        if not api_key:
            raise ValueError(f"No existing credential found for provider {provider.name}")
            
        # Encrypt if needed
        key_value = self._encrypt_key(new_key) if encrypt else new_key
        
        # Update the key
        api_key.rotate(key_value)
        api_key.is_encrypted = encrypt
        if description:
            api_key.description = description
    
    def get_key_info(self, provider: Provider) -> Optional[APIKey]:
        """Get metadata about a credential without exposing the key."""
        family = ModelFamily.from_provider(provider.name)
        api_key = next((k for k in self._api_keys if k.family == family), None)
        if api_key:
            # Create a copy with redacted key
            info = api_key.copy()
            info.key = "********"
            return info
        return None 