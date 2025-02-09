from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


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

    model_config = ConfigDict(validate_assignment=True)


class Task(BaseModel):
    """A task that can be decomposed into subtasks."""

    id: str
    type: str  # Task type from prompt loader
    input_data: Any
    config: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    subtasks: List[SubTask] = Field(default_factory=list)
    result: Optional[Any] = None

    model_config = ConfigDict(validate_assignment=True)


class TokenUsage(BaseModel):
    """Tracks token usage for a single LLM request."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: Optional[float] = None

    model_config = ConfigDict(validate_assignment=True)


class ContextMetrics(BaseModel):
    """Tracks context window usage and limits."""

    max_context_tokens: int
    current_context_tokens: int
    available_tokens: int
    context_utilization: float  # percentage of context window used

    model_config = ConfigDict(validate_assignment=True)


class SummarizationConfig(BaseModel):
    """Configuration for context summarization."""

    enabled: bool = Field(default=True, description="Whether to enable automatic context summarization")
    target_utilization: float = Field(
        default=0.8, description="Target context window utilization before summarizing (0.0-1.0)"
    )
    min_tokens_for_summary: int = Field(
        default=1000, description="Minimum number of tokens before considering summarization"
    )
    preserve_last_n_messages: int = Field(
        default=3, description="Number of most recent messages to preserve without summarization"
    )
    reminder_frequency: int = Field(
        default=5,
        description="Number of messages between reminders of the initial task (0 to disable)",
    )
    reminder_template: str = Field(
        default="Remember, your initial task was: {task}",
        description="Template for task reminder messages",
    )

    model_config = ConfigDict(validate_assignment=True)


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""

    enabled: bool = Field(default=True, description="Whether to enable rate limiting")
    requests_per_minute: int = Field(default=60, description="Maximum requests per minute")
    requests_per_hour: int = Field(default=3500, description="Maximum requests per hour")
    max_daily_tokens: int = Field(default=1000000, description="Maximum tokens per day")
    alert_threshold: float = Field(default=0.8, description="Alert threshold for quota usage (0.0-1.0)")

    model_config = ConfigDict(validate_assignment=True)


class AgentConfig(BaseModel):
    """Configuration for an LLM agent."""

    provider: str
    model_name: str
    api_key: str
    google_api_key: Optional[str] = None  # For Google models
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    max_context_tokens: int = Field(default=32000, ge=1)

    model_config = ConfigDict(validate_assignment=True)


class StorageConfig(BaseModel):
    """Configuration for storage manager."""

    base_path: str
    max_cache_size: int = 1024 * 1024 * 1024  # 1GB default

    model_config = ConfigDict(validate_assignment=True)
