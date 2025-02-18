from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


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
    """Standardized response from LLM processing."""

    content: str = Field(..., description="The content returned by the LLM")
    token_usage: Optional[TokenUsage] = Field(default=None, description="Token usage information if applicable")
    context_metrics: Optional[ContextMetrics] = Field(default=None, description="Context window usage metrics")
    provider: str = Field(..., description="The LLM provider (e.g., 'anthropic', 'gemini', 'openai')")
    provider_response_id: Optional[str] = Field(default=None, description="Provider-specific response ID")
    provider_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific metadata (e.g., Anthropic message ID, Gemini safety ratings)",
    )

    @classmethod
    def from_anthropic_response(cls, response: Any, content: str, token_estimates: Dict[str, int]) -> "LLMResponse":
        """Create LLMResponse from Anthropic API response."""
        return cls(
            content=content,
            provider="anthropic",
            provider_response_id=getattr(response, "id", None),
            provider_metadata={"cost": 0.0, "image_tokens": token_estimates.get("image_tokens", 0)},
            token_usage=TokenUsage(
                prompt_tokens=getattr(response.usage, "input_tokens", 0) if hasattr(response, "usage") else 0,
                completion_tokens=getattr(response.usage, "output_tokens", 0) if hasattr(response, "usage") else 0,
                total_tokens=sum(
                    [
                        getattr(response.usage, "input_tokens", 0) if hasattr(response, "usage") else 0,
                        getattr(response.usage, "output_tokens", 0) if hasattr(response, "usage") else 0,
                    ]
                ),
            ),
            success=True,
        )

    @classmethod
    def from_gemini_response(cls, response: Any, content: str, token_estimates: Dict[str, int]) -> "LLMResponse":
        """Create LLMResponse from Gemini API response."""
        completion_tokens = len(content.split()) * 4  # Rough estimate
        return cls(
            content=content,
            provider="gemini",
            provider_response_id=getattr(response, "id", None),
            provider_metadata={
                "cost": 0.0,
                "image_tokens": token_estimates.get("image_tokens", 0),
                "safety_ratings": getattr(response, "safety_ratings", None),
            },
            token_usage=TokenUsage(
                prompt_tokens=token_estimates["prompt_tokens"],
                completion_tokens=completion_tokens,
                total_tokens=token_estimates["total_tokens"] + completion_tokens,
            ),
            success=True,
        )

    @classmethod
    def from_openai_response(cls, response: Any, messages: List[Dict[str, Any]]) -> "LLMResponse":
        """Create LLMResponse from OpenAI API response."""
        return cls(
            content=response.choices[0].message.content,
            provider="openai",
            provider_response_id=response.id,
            provider_metadata={
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
            },
            token_usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
            success=True,
        )

    def model_dump(self) -> Dict[str, Any]:
        base_data = super().model_dump()
        base_data.update(
            {
                "content": self.content,
                "token_usage": self.token_usage.model_dump() if self.token_usage else None,
                "context_metrics": self.context_metrics.model_dump() if self.context_metrics else None,
                "provider": self.provider,
                "provider_response_id": self.provider_response_id,
                "provider_metadata": self.provider_metadata,
            }
        )
        return base_data


class PromptResponsePair(BaseModel):
    """Represents a prompt and its corresponding response."""

    id: str = Field(..., description="Unique identifier for this prompt-response pair")
    prompt: Any = Field(..., description="The prompt object that was used")
    response: LLMResponse = Field(..., description="The LLM response received")
    created_at: datetime = Field(default_factory=datetime.now)
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context or variables used in prompt rendering"
    )
    model_info: Dict[str, str] = Field(..., description="Information about the model used (provider, model name, etc.)")

    model_config = ConfigDict(validate_assignment=True)


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
    result: Optional[BaseResponse] = None
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
    result: Optional[BaseResponse] = None

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


class Artifact(BaseModel):
    """Base model for all artifacts in the system."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    content_type: str
    data: Any
    path: Optional[Path] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def serialize(self) -> Any:
        """Serialize the artifact data for storage.

        Returns:
            The serialized data in a format suitable for storage (dict, list, or primitive type).
        """
        if hasattr(self.data, "model_dump"):
            return self.data.model_dump()
        elif isinstance(self.data, list) and all(hasattr(item, "model_dump") for item in self.data):
            return [item.model_dump() for item in self.data]
        return self.data


class ArtifactStorage(Protocol):
    """Protocol defining the interface for artifact storage implementations."""

    def save_artifact(self, artifact: Artifact) -> bool:
        """Save an artifact to storage."""
        ...

    def load_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Load an artifact from storage."""
        ...

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact from storage."""
        ...

    def list_artifacts(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Artifact]:
        """List artifacts matching the filter criteria."""
        ...
