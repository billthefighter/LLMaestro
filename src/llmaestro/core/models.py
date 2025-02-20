"""Core models for the application."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from llmaestro.llm import ModelDescriptor
from llmaestro.prompts.base import BasePrompt


class DecompositionConfig(TypedDict, total=False):
    """Configuration for task decomposition."""

    strategy: str  # The strategy to use (chunk, file, error, custom)
    chunk_size: int  # Size of chunks when using chunk strategy
    max_parallel: int  # Maximum number of parallel subtasks
    aggregation: str  # How to combine results


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


class BaseResponse(BaseModel):
    """Base class for all response types."""

    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = Field(..., description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")
    execution_time: Optional[float] = Field(default=None, description="Time taken to generate response in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the response")

    model_config = ConfigDict(validate_assignment=True)


class LLMResponse(BaseResponse):
    """Response from an LLM model."""

    content: str = Field(..., description="The content of the response")
    model: ModelDescriptor = Field(..., description="The model used to generate the response")
    token_usage: TokenUsage = Field(..., description="Token usage statistics")
    context_metrics: Optional[ContextMetrics] = Field(default=None, description="Context window metrics")

    @property
    def model_name(self) -> str:
        """Get the name of the model used."""
        return self.model.name

    @property
    def model_family(self) -> str:
        """Get the family of the model used."""
        return self.model.family


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SubTask(BaseModel):
    """A subtask to be processed by an agent. Cannot be decomposed further."""

    id: str
    type: str  # Task type identifier
    input_data: BasePrompt  # Only baseprompt - no decomposition possible
    parent_task_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[BaseResponse] = None
    error: Optional[str] = None

    model_config = ConfigDict(validate_assignment=True)


class Task(BaseModel):
    """A task that can be decomposed into subtasks.

    A task can be either:
    1. An LLM task with a BasePrompt as input
    2. A data processing task with raw data as input (e.g. files, text chunks)
    """

    id: str
    type: str  # Task type identifier
    input_data: Union[BasePrompt, Dict[str, Any], str, List[Any]]  # More explicit typing for input data
    config: Dict[str, Any]  # Configuration for task execution and decomposition
    status: TaskStatus = TaskStatus.PENDING
    subtasks: List[SubTask] = Field(default_factory=list)
    result: Optional[BaseResponse] = None
    decomposition_config: Optional[DecompositionConfig] = None  # Configuration for how to break down the task

    model_config = ConfigDict(validate_assignment=True)

    @property
    def is_llm_task(self) -> bool:
        """Whether this task requires LLM interaction."""
        return isinstance(self.input_data, BasePrompt)

    @property
    def decomposition_strategy(self) -> Optional[str]:
        """Get the decomposition strategy if configured."""
        return self.decomposition_config.get("strategy") if self.decomposition_config else None


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
