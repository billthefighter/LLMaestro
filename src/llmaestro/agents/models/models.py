"""Core models for agent functionality."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Set

from llmaestro.core.models import BaseResponse, ContextMetrics, TokenUsage
from pydantic import BaseModel, ConfigDict, Field


class AgentResponse(BaseResponse):
    """Base response type for agent operations."""

    agent_id: str = Field(..., description="ID of the agent that generated the response")
    agent_type: str = Field(..., description="Type of agent that generated the response")


class AgentMetrics(BaseModel):
    """Metrics for agent performance and usage."""

    token_usage: TokenUsage
    context_metrics: ContextMetrics
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(validate_assignment=True)


class AgentState(str, Enum):
    """States an agent can be in."""

    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentCapability(str, Enum):
    """Capabilities that an agent can have."""

    TEXT = "text"
    CODE = "code"
    VISION = "vision"
    AUDIO = "audio"
    FUNCTION_CALLING = "function_calling"
    TOOL_USE = "tool_use"
    PLANNING = "planning"
    REASONING = "reasoning"


class Agent(BaseModel):
    """Base model for an agent instance."""

    id: str = Field(..., description="Unique identifier for this agent")
    type: str = Field(..., description="Type of agent (e.g., 'general', 'specialist')")
    provider: str = Field(..., description="The LLM provider for this agent")
    model: str = Field(..., description="The model being used by this agent")
    state: AgentState = Field(default=AgentState.IDLE)
    capabilities: Set[AgentCapability] = Field(default_factory=set)
    metrics: Optional[AgentMetrics] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(validate_assignment=True)

    def update_metrics(self, metrics: AgentMetrics) -> None:
        """Update the agent's metrics."""
        self.metrics = metrics
        self.last_active = datetime.now()

    def update_state(self, state: AgentState) -> None:
        """Update the agent's state."""
        self.state = state
        self.last_active = datetime.now()
