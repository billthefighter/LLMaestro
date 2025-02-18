"""Agent pool for managing multiple LLM agents."""
from typing import Any, Dict, Optional, Protocol, TypeVar, cast
from concurrent.futures import ThreadPoolExecutor
import asyncio

from llmaestro.core.config import AgentPoolConfig, AgentTypeConfig, get_config
from llmaestro.core.models import AgentConfig, SubTask
from llmaestro.core.task_manager import TaskManager
from llmaestro.llm.interfaces import LLMResponse
from llmaestro.llm.llm_registry import ModelRegistry
from llmaestro.llm.interfaces.factory import create_llm_interface

T = TypeVar("T")


class JsonOutputTransform(Protocol):
    """Protocol for JSON output transformers."""

    def transform(self, response: LLMResponse) -> Dict[str, Any]:
        ...


class Agent:
    """Agent that can process tasks using LLM."""

    def __init__(self, config: AgentTypeConfig):
        self.id = str(id(self))  # Use object id as unique identifier
        self.config = config
        self.active_tasks: Dict[str, SubTask] = {}
        self.llm = create_llm_interface(config)

    async def process_task(self, task: SubTask) -> Any:
        self.active_tasks[task.id] = task
        try:
            result = await self.llm.generate(task.input_data)
            return result
        finally:
            del self.active_tasks[task.id]


class AgentPool:
    """Pool of agents that can be reused across tasks."""

    def __init__(self, config: Optional[AgentPoolConfig] = None):
        """Initialize the agent pool.

        Args:
            config: Optional agent pool configuration. If None, uses config from global config.
        """
        self._config = config or get_config().agents
        self._task_manager = TaskManager()
        self._model_registry = ModelRegistry(providers={})  # Initialize with empty providers, will be populated by config
        self._active_agents: Dict[str, Agent] = {}
        self.executor = ThreadPoolExecutor(max_workers=self._config.max_agents)
        self.tasks: Dict[str, asyncio.Task[Any]] = {}
        self.loop = asyncio.get_event_loop()

    def get_agent(self, agent_type: Optional[str] = None) -> Agent:
        """Get an agent of the specified type from the pool."""
        agent_config = self._config.get_agent_config(agent_type)
        
        # Create a new agent if we haven't reached the limit
        if len(self._active_agents) < self._config.max_agents:
            agent = self._create_agent(agent_config)
            self._active_agents[agent.id] = agent
            return agent
            
        # Otherwise, find the least busy agent
        return min(self._active_agents.values(), key=lambda a: len(a.active_tasks))

    def _create_agent(self, agent_config: AgentTypeConfig) -> Agent:
        """Create a new agent with the specified configuration."""
        return Agent(config=agent_config)

    def submit_task(self, subtask: SubTask) -> None:
        """Submit a subtask to be processed by an available agent."""
        # Get or create an agent for the task
        agent = self.get_agent(subtask.type)

        # Create and store the async task
        task = self.loop.create_task(agent.process_task(subtask))
        self.tasks[subtask.id] = task

    def wait_for_result(self, task_id: str) -> Any:
        """Wait for a task to complete and return its result."""
        if task_id not in self.tasks:
            raise ValueError(f"No task found with id: {task_id}")

        task = self.tasks[task_id]
        result = self.loop.run_until_complete(task)

        # Cleanup
        del self.tasks[task_id]
        if task_id in self._active_agents:
            del self._active_agents[task_id]

        return result
