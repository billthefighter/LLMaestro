import json
from pathlib import Path
from typing import Optional

from ..core.models import StorageConfig, Task


class StorageManager:
    def __init__(self, base_path: str):
        self.config = StorageConfig(base_path=base_path)
        self.base_path = Path(base_path)
        self._ensure_storage_path()

    def _ensure_storage_path(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_task_path(self, task_id: str) -> Path:
        """Get the file path for a task's data."""
        return self.base_path / f"{task_id}.json"

    def save_task(self, task: Task) -> None:
        """Save a task's data to disk."""
        task_path = self._get_task_path(task.id)
        with open(task_path, "w") as f:
            json.dump(task.model_dump(), f, indent=2)

    def load_task(self, task_id: str) -> Optional[Task]:
        """Load a task's data from disk."""
        task_path = self._get_task_path(task_id)
        if not task_path.exists():
            return None

        with open(task_path) as f:
            data = json.load(f)
            return Task.model_validate(data)

    def delete_task(self, task_id: str) -> bool:
        """Delete a task's data from disk."""
        task_path = self._get_task_path(task_id)
        if task_path.exists():
            task_path.unlink()
            return True
        return False
