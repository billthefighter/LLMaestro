from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class SubTask(BaseModel):
    """A subtask to be processed by an agent."""
    id: str
    type: str  # Task type from prompt loader
    input_data: Union[str, Dict[str, Any]]
    parent_task_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None

class Task(BaseModel):
    """A task that can be decomposed into subtasks."""
    id: str
    type: str  # Task type from prompt loader
    input_data: Any
    config: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    subtasks: List[SubTask] = Field(default_factory=list)
    result: Optional[Any] = None

class AgentConfig(BaseModel):
    """Configuration for an LLM agent."""
    model_name: str
    max_tokens: int
    temperature: float = 0.7
    top_p: float = 1.0

class StorageConfig(BaseModel):
    """Configuration for storage manager."""
    base_path: str
    max_cache_size: int = 1024 * 1024 * 1024  # 1GB default 