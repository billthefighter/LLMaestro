"""Configuration models for agent functionality."""

from typing import Dict, Optional, Set

from llmaestro.agents.models.models import AgentCapability
from llmaestro.core.models import RateLimitConfig, SummarizationConfig
from pydantic import BaseModel, ConfigDict, Field


class AgentRuntimeConfig(BaseModel):
    """Runtime configuration for an agent."""

    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    max_context_tokens: int = Field(default=32000, ge=1)
    stream: bool = Field(default=True)

    model_config = ConfigDict(validate_assignment=True)


class AgentTypeConfig(BaseModel):
    """Configuration for a specific type of agent."""

    provider: str = Field(default="anthropic")
    model: str = Field(default="claude-3-sonnet-20240229")
    runtime: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig)
    description: Optional[str] = None
    capabilities: Set[AgentCapability] = Field(default_factory=set)
    settings: Dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)


class AgentPoolConfig(BaseModel):
    """Configuration for the agent pool."""

    max_agents: int = Field(default=10, ge=1, le=100)
    default_agent_type: str = Field(default="general")
    agent_types: Dict[str, AgentTypeConfig] = Field(
        default_factory=lambda: {
            "general": AgentTypeConfig(
                description="General purpose agent",
                capabilities={
                    AgentCapability.TEXT,
                    AgentCapability.CODE,
                    AgentCapability.FUNCTION_CALLING,
                    AgentCapability.TOOL_USE,
                },
            ),
            "fast": AgentTypeConfig(
                model="claude-3-haiku-20240229",
                description="Fast, lightweight agent for simple tasks",
                runtime=AgentRuntimeConfig(max_tokens=4096),
                capabilities={AgentCapability.TEXT, AgentCapability.CODE},
            ),
            "specialist": AgentTypeConfig(
                model="claude-3-opus-20240229",
                description="Specialist agent for complex tasks",
                runtime=AgentRuntimeConfig(max_tokens=16384, max_context_tokens=48000),
                capabilities={
                    AgentCapability.TEXT,
                    AgentCapability.CODE,
                    AgentCapability.VISION,
                    AgentCapability.FUNCTION_CALLING,
                    AgentCapability.TOOL_USE,
                    AgentCapability.PLANNING,
                    AgentCapability.REASONING,
                },
            ),
        }
    )

    model_config = ConfigDict(validate_assignment=True)

    def get_agent_config(self, agent_type: Optional[str] = None) -> AgentTypeConfig:
        """Get configuration for a specific agent type.

        Args:
            agent_type: The type of agent to get config for. If None, uses default type.

        Returns:
            AgentTypeConfig for the specified or default agent type.

        Raises:
            ValueError: If the agent type is not found in configuration.
        """
        type_to_use = agent_type or self.default_agent_type
        if type_to_use not in self.agent_types:
            raise ValueError(f"Agent type '{type_to_use}' not found in configuration")
        return self.agent_types[type_to_use]
