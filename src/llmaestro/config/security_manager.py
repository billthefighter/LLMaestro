"""Centralized security policy management."""

from typing import Dict, Any, Set, Optional
from pydantic import BaseModel, Field
import re
from urllib.parse import urlparse

class SecurityPolicy(BaseModel):
    """Security policy configuration for a provider."""
    
    require_encryption: bool = True
    allowed_domains: Set[str] = Field(default_factory=set)
    allowed_endpoints: Set[str] = Field(default_factory=set)
    max_token_limit: Optional[int] = None
    require_ssl: bool = True

class SecurityManager(BaseModel):
    """Centralizes security policies and validation."""
    
    allowed_api_domains: Set[str] = Field(default_factory=set)
    require_api_key_encryption: bool = True
    allowed_providers: Set[str] = Field(default_factory=set)
    provider_policies: Dict[str, SecurityPolicy] = Field(default_factory=dict)
    
    def validate_api_domain(self, domain: str) -> bool:
        """Validate if an API domain is allowed."""
        if not self.allowed_api_domains:
            return True
            
        parsed = urlparse(domain)
        domain_to_check = parsed.netloc or parsed.path
        
        return any(
            domain_to_check.endswith(allowed) 
            for allowed in self.allowed_api_domains
        )
    
    def validate_provider(self, provider: str) -> bool:
        """Validate if a provider is allowed."""
        if not self.allowed_providers:
            return True
        return provider in self.allowed_providers
    
    def get_provider_security_policy(self, provider: str) -> SecurityPolicy:
        """Get security policies for a specific provider."""
        if provider in self.provider_policies:
            return self.provider_policies[provider]
            
        # Return default policy if none specified
        return SecurityPolicy(
            require_encryption=self.require_api_key_encryption,
            allowed_domains=set(d for d in self.allowed_api_domains if provider in d)
        )
    
    def set_provider_policy(self, provider: str, policy: SecurityPolicy) -> None:
        """Set security policy for a provider."""
        self.provider_policies[provider] = policy
        
    def validate_api_key_format(self, provider: str, api_key: str) -> bool:
        """Validate API key format for a provider."""
        # Add provider-specific API key format validation
        key_formats = {
            "openai": r"^sk-[A-Za-z0-9]{32,}$",
            "anthropic": r"^sk-ant-[A-Za-z0-9]{32,}$",
            "cohere": r"^[A-Za-z0-9]{32,}$",
        }
        
        if provider in key_formats:
            return bool(re.match(key_formats[provider], api_key))
        return True  # No specific format validation for unknown providers 