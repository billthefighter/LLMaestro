"""Agent pool for managing multiple LLM agents."""
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Protocol, TypeVar, cast

from llmaestro.agents.models.config import AgentPoolConfig, AgentTypeConfig
from llmaestro.agents.models.models import Agent, AgentMetrics, AgentState
from llmaestro.core.config import get_config
from llmaestro.core.models import AgentConfig, SubTask
from llmaestro.core.task_manager import TaskManager
from llmaestro.llm.interfaces import LLMResponse
from llmaestro.llm.interfaces.factory import create_llm_interface
from llmaestro.llm.llm_registry import ModelRegistry
from llmaestro.prompts.base import BasePrompt

T = TypeVar("T")


class JsonOutputTransform(Protocol):
    """Protocol for JSON output transformers."""

    def transform(self, response: LLMResponse) -> Dict[str, Any]:
        ...


def _convert_to_agent_config(type_config: AgentTypeConfig) -> AgentConfig:
    """Convert AgentTypeConfig to AgentConfig for LLM interface."""
    runtime = type_config.runtime.model_dump()
    return AgentConfig(
        provider=type_config.provider,
        model_name=type_config.model,
        api_key=get_config().user_config.api_keys.get(type_config.provider, ""),
        max_tokens=runtime.get("max_tokens", 1024),
        temperature=runtime.get("temperature", 0.7),
        rate_limit=runtime.get("rate_limit", {}),
        summarization=runtime.get("summarization", {}),
        max_context_tokens=runtime.get("max_context_tokens", 32000),
    )


class RuntimeAgent:
    """Runtime agent that processes tasks using LLM."""

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
            capabilities=config.capabilities,
        )
        self.config = config
        self.active_tasks: Dict[str, asyncio.Task[Any]] = {}

        # Convert config and create LLM interface
        llm_config = _convert_to_agent_config(config)
        self.llm = create_llm_interface(llm_config)

    async def process_task(self, task: SubTask) -> Any:
        """Process a task using this agent's LLM.

        Args:
            task: The subtask to process

        Returns:
            The result from the LLM
        """
        self.agent.update_state(AgentState.BUSY)

        try:
            start_time = asyncio.get_event_loop().time()

            # Convert input data to BasePrompt if needed
            input_data = task.input_data
            if not isinstance(input_data, BasePrompt):
                raise ValueError(f"Unsupported input type: {type(input_data)}")

            result = await self.llm.process(input_data)
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
    """Pool of agents that can be reused across tasks."""

    def __init__(self, config: Optional[AgentPoolConfig] = None):
        """Initialize the agent pool.

        Args:
            config: Optional agent pool configuration. If None, uses config from global config.
        """
        self._config = config or get_config().agents
        self._task_manager = TaskManager()
        self._model_registry = ModelRegistry(providers={})
        self._active_agents: Dict[str, RuntimeAgent] = {}
        self.executor = ThreadPoolExecutor(max_workers=self._config.max_agents)
        self.tasks: Dict[str, asyncio.Task[Any]] = {}
        self.loop = asyncio.get_event_loop()

    def get_agent(self, agent_type: Optional[str] = None) -> RuntimeAgent:
        """Get an agent of the specified type from the pool.

        Args:
            agent_type: Optional type of agent to get. If None, uses default type.

        Returns:
            A RuntimeAgent instance

        Raises:
            ValueError: If agent type not found or pool is full
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

        return min(compatible_agents, key=lambda a: len(a.active_tasks))

    def _create_agent(self, agent_config: AgentTypeConfig) -> RuntimeAgent:
        """Create a new agent with the specified configuration."""
        return RuntimeAgent(config=agent_config)

    async def submit_task(self, subtask: SubTask) -> None:
        """Submit a subtask to be processed by an available agent.

        Args:
            subtask: The subtask to process
        """
        # Get or create an agent for the task
        agent = self.get_agent(subtask.type)

        # Create and store the async task
        task = self.loop.create_task(agent.process_task(subtask))
        self.tasks[subtask.id] = task
        agent.active_tasks[subtask.id] = task

    async def wait_for_result(self, task_id: str) -> Any:
        """Wait for a task to complete and return its result.

        Args:
            task_id: ID of the task to wait for

        Returns:
            The task result

        Raises:
            ValueError: If task not found
        """
        if task_id not in self.tasks:
            raise ValueError(f"No task found with id: {task_id}")

        task = self.tasks[task_id]
        try:
            result = await task

            # Cleanup task references
            del self.tasks[task_id]
            for agent in self._active_agents.values():
                if task_id in agent.active_tasks:
                    del agent.active_tasks[task_id]
                    break

            return result

        except Exception as e:
            # Ensure cleanup even on error
            if task_id in self.tasks:
                del self.tasks[task_id]
            for agent in self._active_agents.values():
                if task_id in agent.active_tasks:
                    del agent.active_tasks[task_id]
                    break
            raise e

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent pool."""
        return {
            "total_agents": len(self._active_agents),
            "max_agents": self._config.max_agents,
            "active_tasks": sum(len(agent.active_tasks) for agent in self._active_agents.values()),
            "agents": [
                {
                    "id": agent.agent.id,
                    "type": agent.agent.type,
                    "state": agent.agent.state,
                    "active_tasks": len(agent.active_tasks),
                    "metrics": agent.agent.metrics.model_dump() if agent.agent.metrics else None,
                }
                for agent in self._active_agents.values()
            ],
        }
