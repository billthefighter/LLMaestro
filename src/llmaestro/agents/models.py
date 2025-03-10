"""Core models for agent functionality."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import ConfigDict, Field

from llmaestro.core.models import BaseResponse, ContextMetrics, TokenUsage
from llmaestro.llm.capabilities import LLMCapabilities
from llmaestro.core.persistence import PersistentModel


class AgentResponse(BaseResponse):
    """Base response type for agent operations."""

    agent_id: str = Field(..., description="ID of the agent that generated the response")
    agent_type: str = Field(..., description="Type of agent that generated the response")


class AgentMetrics(PersistentModel):
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


class Agent(PersistentModel):
    """Base model for an agent instance."""

    id: str = Field(description="Unique identifier for this agent")
    type: str = Field(description="Type of agent (e.g., 'general', 'specialist')")
    model: str = Field(description="The model being used by this agent")
    state: AgentState = Field(default=AgentState.IDLE)
    capabilities: LLMCapabilities = Field(description="The model's capabilities")
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
