"""Shared type definitions for prompts."""
import json
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import Field
from llmaestro.core.persistence import PersistentModel


class VersionInfo(PersistentModel):
    """Version information for a prompt."""

    number: str
    timestamp: datetime
    author: str
    description: str
    change_type: str
    git_commit: Optional[str] = None

    def model_dump_json(self, **kwargs) -> str:
        data = self.model_dump(**kwargs)
        if "timestamp" in data:
            data["timestamp"] = data["timestamp"].isoformat()
        return json.dumps(data, **kwargs)


class PromptMetadata(PersistentModel):
    """Enhanced metadata for prompts."""

    type: str
    model_requirements: Optional[Dict] = None
    decomposition: Optional[Dict] = None
    tags: List[str] = Field(default_factory=list)
    is_active: bool = True
