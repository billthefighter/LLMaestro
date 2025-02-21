"""Agent pool for managing multiple LLM agents for prompt processing."""
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Protocol, Set, TypeVar, cast

from llmaestro.agents.capabilities import AgentCapability
from llmaestro.agents.models import Agent, AgentCapability, AgentMetrics, AgentState
from llmaestro.config.agent import AgentPoolConfig, AgentTypeConfig
from llmaestro.config.manager import ConfigurationManager
from llmaestro.llm.interfaces import LLMResponse
from llmaestro.llm.interfaces.factory import create_llm_interface
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.prompts.base import BasePrompt

T = TypeVar("T")


class JsonOutputTransform(Protocol):
    """Protocol for JSON output transformers."""

    def transform(self, response: LLMResponse) -> Dict[str, Any]:
        ...


class RuntimeAgent:
    """Runtime agent that processes prompts using LLM.

    This agent handles the actual execution of prompts through the LLM interface,
    managing state and metrics for each execution.
    """

    def __init__(self, config: AgentTypeConfig):
        """Initialize a runtime agent.

        Args:
            config: Configuration for this agent type
        """
        self.agent = Agent(
            id=str(uuid.uuid4()),
            type=config.description or "unknown",
            provider=config.provider,
            model=config.model,
            capabilities=set()
            if config.capabilities is None
            else {AgentCapability(cap) for cap in config.capabilities},
        )
        self.config = config
        self.active_prompts: Dict[str, asyncio.Task[Any]] = {}

        # Create LLM interface with runtime config
        runtime = config.runtime.model_dump()
        self.llm = create_llm_interface(runtime)

    async def process_prompt(self, prompt: BasePrompt) -> LLMResponse:
        """Process a prompt using this agent's LLM.

        Args:
            prompt: The prompt to process

        Returns:
            The LLM response
        """
        self.agent.update_state(AgentState.BUSY)

        try:
            start_time = asyncio.get_event_loop().time()
            result = await self.llm.process(prompt)
            execution_time = asyncio.get_event_loop().time() - start_time

            # Update agent metrics
            if result.token_usage and result.context_metrics:
                self.agent.update_metrics(
                    AgentMetrics(
                        token_usage=result.token_usage,
                        context_metrics=result.context_metrics,
                        execution_time=execution_time,
                    )
                )

            return result

        except Exception as e:
            self.agent.update_state(AgentState.ERROR)
            raise e

        finally:
            if self.agent.state != AgentState.ERROR:
                self.agent.update_state(AgentState.IDLE)


class AgentPool:
    """Pool of agents that can be reused for prompt processing.

    Manages a collection of runtime agents, handling agent allocation,
    prompt execution, and resource management.
    """

    def __init__(self, config: Optional[AgentPoolConfig] = None):
        """Initialize the agent pool.

        Args:
            config: Optional agent pool configuration. If None, uses config from global config.
        """
        self._config = config or get_config().agents
        self._llm_registry = LLMRegistry()
        self._active_agents: Dict[str, RuntimeAgent] = {}
        self.executor = ThreadPoolExecutor(max_workers=self._config.max_agents)
        self.prompts: Dict[str, asyncio.Task[Any]] = {}
        self.loop = asyncio.get_event_loop()

    def get_agent(self, agent_type: Optional[str] = None) -> RuntimeAgent:
        """Get an agent suitable for prompt processing.

        Args:
            agent_type: Optional type of agent to get. If None, uses default type.

        Returns:
            A RuntimeAgent instance capable of processing prompts

        Raises:
            ValueError: If no suitable agent is available or pool is full
        """
        agent_config = cast(AgentTypeConfig, self._config.get_agent_config(agent_type))

        # Create a new agent if we haven't reached the limit
        if len(self._active_agents) < self._config.max_agents:
            agent = self._create_agent(agent_config)
            self._active_agents[agent.agent.id] = agent
            return agent

        # Otherwise, find the least busy agent of compatible type
        compatible_agents = [
            agent
            for agent in self._active_agents.values()
            if agent.config.provider == agent_config.provider and agent.config.model == agent_config.model
        ]

        if not compatible_agents:
            raise ValueError(f"No compatible agents available for type: {agent_type}")

        return min(compatible_agents, key=lambda a: len(a.active_prompts))

    def _create_agent(self, agent_config: AgentTypeConfig) -> RuntimeAgent:
        """Create a new agent for prompt processing.

        Args:
            agent_config: Configuration for the new agent

        Returns:
            A new RuntimeAgent instance
        """
        return RuntimeAgent(config=agent_config)

    async def execute_prompt(
        self,
        prompt: BasePrompt,
        agent_type: Optional[str] = None,
        required_capabilities: Optional[Set[AgentCapability]] = None,
    ) -> LLMResponse:
        """Execute a prompt using an appropriate agent.

        This method handles the complete lifecycle of prompt execution:
        1. Agent selection/creation
        2. Prompt submission
        3. Response retrieval
        4. Resource cleanup

        Args:
            prompt: The prompt to execute
            agent_type: Optional type of agent to use
            required_capabilities: Optional set of required capabilities

        Returns:
            The LLM response from processing the prompt

        Raises:
            ValueError: If no suitable agent is available
            RuntimeError: If prompt execution fails
        """
        # Get or create an agent
        agent = self.get_agent(agent_type)

        # Create and store the async task
        prompt_id = str(uuid.uuid4())
        task = self.loop.create_task(agent.process_prompt(prompt))
        self.prompts[prompt_id] = task
        agent.active_prompts[prompt_id] = task

        try:
            return await task
        finally:
            # Cleanup
            if prompt_id in self.prompts:
                del self.prompts[prompt_id]
            if prompt_id in agent.active_prompts:
                del agent.active_prompts[prompt_id]

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent pool.

        Returns a dictionary containing current pool statistics including:
        - Total number of agents
        - Maximum allowed agents
        - Number of active prompts
        - Per-agent statistics
        """
        return {
            "total_agents": len(self._active_agents),
            "max_agents": self._config.max_agents,
            "active_prompts": sum(len(agent.active_prompts) for agent in self._active_agents.values()),
            "agents": [
                {
                    "id": agent.agent.id,
                    "type": agent.agent.type,
                    "state": agent.agent.state,
                    "active_prompts": len(agent.active_prompts),
                    "metrics": agent.agent.metrics.model_dump() if agent.agent.metrics else None,
                }
                for agent in self._active_agents.values()
            ],
        }


def get_config() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    return ConfigurationManager.get_instance()
