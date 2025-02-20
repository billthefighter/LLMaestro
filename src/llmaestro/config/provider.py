"""Provider-specific configuration models."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProviderAPIConfig(BaseModel):
    """Configuration for provider API access."""

    provider: str
    name: str
    api_base: str
    api_key: Optional[str] = None
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    rate_limits: Dict[str, int] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)
