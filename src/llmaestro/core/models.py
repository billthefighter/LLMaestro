"""Core models for the application."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import ConfigDict, Field

from llmaestro.core.persistence import PersistentModel


class TokenUsage(PersistentModel):
    """Tracks token usage for a single LLM request."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: Optional[float] = None

    model_config = ConfigDict(validate_assignment=True)


class ContextMetrics(PersistentModel):
    """Tracks context window usage and limits."""

    max_context_tokens: int
    current_context_tokens: int
    available_tokens: int
    context_utilization: float  # percentage of context window used

    model_config = ConfigDict(validate_assignment=True)


class BaseResponse(PersistentModel):
    """Base class for all response types."""

    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = Field(description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")
    execution_time: Optional[float] = Field(default=None, description="Time taken to generate response in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the response")

    model_config = ConfigDict(validate_assignment=True)


class LLMResponse(BaseResponse):
    """Response from an LLM model."""

    content: str = Field(..., description="The content of the response")
    token_usage: TokenUsage = Field(..., description="Token usage statistics")
    context_metrics: Optional[ContextMetrics] = Field(default=None, description="Context window metrics")
