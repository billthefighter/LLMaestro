"""System-wide configuration models."""

from pathlib import Path
from typing import Dict, Union

import yaml
from pydantic import BaseModel, ConfigDict

from llmaestro.llm.provider_registry import ProviderConfig


class SystemConfig(BaseModel):
    """System-wide configuration for all providers and models."""

    providers: Dict[str, ProviderConfig]

    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SystemConfig":
        """Create configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
