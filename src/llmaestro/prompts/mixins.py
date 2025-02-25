import json
from datetime import datetime, timedelta
from typing import List

from llmaestro.prompts.types import VersionInfo
from pydantic import BaseModel, Field


class VersionMixin(BaseModel):
    """Mixin for version control functionality in prompts."""

    current_version: VersionInfo
    version_history: List[VersionInfo] = Field(default_factory=list)

    @property
    def version(self) -> str:
        """Get current version number."""
        return self.current_version.number

    @property
    def author(self) -> str:
        """Get original author."""
        return self.version_history[0].author if self.version_history else self.current_version.author

    @property
    def created_at(self) -> datetime:
        """Get creation timestamp."""
        return self.version_history[0].timestamp if self.version_history else self.current_version.timestamp

    @property
    def updated_at(self) -> datetime:
        """Get last update timestamp."""
        return self.current_version.timestamp

    @property
    def age(self) -> timedelta:
        """Get the age of this prompt."""
        return datetime.now() - self.created_at

    def model_dump_json(self, **kwargs) -> str:
        """Override to handle datetime serialization in versions."""
        data = self.model_dump(**kwargs)
        if "current_version" in data and "timestamp" in data["current_version"]:
            data["current_version"]["timestamp"] = data["current_version"]["timestamp"].isoformat()
        for version in data.get("version_history", []):
            if "timestamp" in version:
                version["timestamp"] = version["timestamp"].isoformat()
        return json.dumps(data, **kwargs)
