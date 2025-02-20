"""Core models for the application."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from llmaestro.llm import ModelDescriptor


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


class AgentConfig(BaseModel):
    """Configuration for an LLM agent."""

    provider: str
    model_name: str
    api_key: str
    google_api_key: Optional[str] = None  # For Google models
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    rate_limit: Dict[str, Any] = Field(default_factory=dict)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    max_context_tokens: int = Field(default=32000, ge=1)

    model_config = ConfigDict(validate_assignment=True)
