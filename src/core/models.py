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

class TokenUsage(BaseModel):
    """Tracks token usage for a single LLM request."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: Optional[float] = None

class ContextMetrics(BaseModel):
    """Tracks context window usage and limits."""
    max_context_tokens: int
    current_context_tokens: int
    available_tokens: int
    context_utilization: float  # percentage of context window used

class SummarizationConfig(BaseModel):
    """Configuration for context summarization."""
    enabled: bool = Field(
        default=True,
        description="Whether to enable automatic context summarization"
    )
    target_utilization: float = Field(
        default=0.8,
        description="Target context window utilization before summarizing (0.0-1.0)"
    )
    min_tokens_for_summary: int = Field(
        default=1000,
        description="Minimum number of tokens before considering summarization"
    )
    preserve_last_n_messages: int = Field(
        default=3,
        description="Number of most recent messages to preserve without summarization"
    )
    reminder_frequency: int = Field(
        default=5,
        description="Number of messages between reminders of the initial task (0 to disable)"
    )
    reminder_template: str = Field(
        default="Remember, your initial task was: {task}",
        description="Template for task reminder messages"
    )

class AgentConfig(BaseModel):
    """Configuration for an LLM agent."""
    provider: str = Field(
        description="The LLM provider (e.g., 'openai', 'anthropic')"
    )
    model_name: str = Field(
        description="The specific model to use (e.g., 'gpt-4', 'claude-2')"
    )
    max_tokens: int = Field(
        description="Maximum tokens to generate in responses"
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature for response generation (0.0-1.0)"
    )
    top_p: float = Field(
        default=1.0,
        description="Top-p sampling parameter (0.0-1.0)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the LLM provider"
    )
    max_context_tokens: int = Field(
        default=8192,
        description="Maximum tokens allowed in context window"
    )
    token_tracking: bool = Field(
        default=True,
        description="Whether to track token usage and costs"
    )
    cost_per_1k_tokens: Optional[float] = Field(
        default=None,
        description="Cost per 1000 tokens (if tracking costs)"
    )
    summarization: SummarizationConfig = Field(
        default_factory=SummarizationConfig,
        description="Configuration for context summarization"
    )

class StorageConfig(BaseModel):
    """Configuration for storage manager."""
    base_path: str
    max_cache_size: int = 1024 * 1024 * 1024  # 1GB default 