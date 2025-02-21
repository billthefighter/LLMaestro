"""System-wide configuration models."""

from pathlib import Path
from typing import Dict, Optional, Set, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ProviderSystemConfig(BaseModel):
    """System configuration specific to a provider."""

    # Provider identification
    name: str
    api_base: str
    api_version: Optional[str] = None

    # Core settings
    capabilities_detector: str

    # Rate limiting and quotas
    rate_limits: Dict[str, int]

    # Request handling
    timeout_settings: Dict[str, float] = Field(
        default_factory=lambda: {"request": 30.0, "stream": 60.0, "connect": 10.0}
    )
    retry_settings: Dict[str, int] = Field(
        default_factory=lambda: {"max_retries": 3, "retry_delay": 1, "max_delay": 30}
    )

    # Security
    allowed_api_domains: Optional[Set[str]] = None
    require_api_key_encryption: bool = True

    model_config = ConfigDict(validate_assignment=True)


class LLMSystemConfig(BaseModel):
    """Global system configuration for LLM functionality."""

    # Global rate limiting and quotas
    global_rate_limits: Dict[str, int] = Field(
        default_factory=dict, description="Global rate limits that apply across all providers"
    )
    max_parallel_requests: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0

    # Default timeouts
    default_request_timeout: float = 30.0
    default_stream_timeout: float = 60.0

    # Cache settings
    enable_response_cache: bool = False
    cache_ttl: int = 3600  # 1 hour

    # Logging and monitoring
    log_level: str = "INFO"
    enable_telemetry: bool = False

    # Security settings
    require_api_key_encryption: bool = True
    allowed_api_domains: Set[str] = Field(default_factory=set)

    model_config = ConfigDict(validate_assignment=True)


class SystemConfig(BaseModel):
    """Root system configuration combining global and provider-specific settings."""

    # Global LLM settings
    llm: LLMSystemConfig = Field(default_factory=LLMSystemConfig, description="Global LLM system settings")

    # Provider configurations
    providers: Dict[str, ProviderSystemConfig] = Field(
        default_factory=dict, description="Provider-specific configurations"
    )

    model_config = ConfigDict(validate_assignment=True)

    def get_provider_config(self, provider_name: str) -> Optional[ProviderSystemConfig]:
        """Get configuration for a specific provider, applying global defaults where appropriate."""
        provider_config = self.providers.get(provider_name)
        if not provider_config:
            return None

        # Apply global settings if provider-specific ones aren't set
        if not provider_config.allowed_api_domains:
            provider_config.allowed_api_domains = self.llm.allowed_api_domains

        return provider_config

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SystemConfig":
        """Create configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, sort_keys=False)
