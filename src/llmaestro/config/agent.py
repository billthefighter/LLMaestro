"""Agent-specific configuration models."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from llmaestro.llm.models import LLMCapabilities


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""

    requests_per_minute: int = Field(default=60, ge=1)
    requests_per_hour: int = Field(default=3600, ge=1)
    max_daily_tokens: int = Field(default=1000000, ge=1)
    alert_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

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


class AgentRuntimeConfig(BaseModel):
    """Runtime configuration for an agent."""

    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    max_context_tokens: int = Field(default=32000, ge=1)
    stream: bool = Field(default=True)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)

    model_config = ConfigDict(validate_assignment=True)


class AgentTypeConfig(BaseModel):
    """Configuration for a specific agent type."""

    provider: str = Field(default="anthropic")
    model: str = Field(default="claude-3-sonnet-20240229")
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.7)
    description: Optional[str] = None
    settings: Dict[str, Any] = Field(default_factory=dict)
    capabilities: Optional[LLMCapabilities] = None
    runtime: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig)

    model_config = ConfigDict(validate_assignment=True)


class AgentPoolConfig(BaseModel):
    """Configuration for the agent pool."""

    max_agents: int = Field(default=10, ge=1, le=100)
    default_agent_type: str = Field(default="general")
    agent_types: Dict[str, AgentTypeConfig] = Field(
        default_factory=lambda: {
            "general": AgentTypeConfig(
                provider="anthropic", model="claude-3-sonnet-20240229", description="General purpose agent"
            ),
            "fast": AgentTypeConfig(
                provider="anthropic",
                model="claude-3-haiku-20240229",
                description="Fast, lightweight agent for simple tasks",
                runtime=AgentRuntimeConfig(max_tokens=4096),
            ),
            "specialist": AgentTypeConfig(
                provider="anthropic",
                model="claude-3-opus-20240229",
                description="Specialist agent for complex tasks",
                runtime=AgentRuntimeConfig(max_tokens=16384, max_context_tokens=48000),
            ),
        }
    )

    model_config = ConfigDict(validate_assignment=True)

    def get_agent_config(self, agent_type: Optional[str] = None) -> AgentTypeConfig:
        """Get configuration for a specific agent type."""
        type_to_use = agent_type or self.default_agent_type
        if type_to_use not in self.agent_types:
            raise ValueError(f"Agent type '{type_to_use}' not found in configuration")
        return self.agent_types[type_to_use]
