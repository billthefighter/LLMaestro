"""Agent pool for managing multiple LLM agents."""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Protocol, TypeVar, cast

from src.core.config import AgentPoolConfig, get_config

from ..core.models import AgentConfig, SubTask
from ..llm.chains import ChainStep, OutputTransform, SequentialChain
from ..llm.interfaces import LLMResponse
from ..llm.interfaces.factory import create_llm_interface
from ..prompts.loader import PromptLoader

T = TypeVar("T")


class JsonOutputTransform(Protocol):
    """Protocol for JSON output transformers."""

    def transform(self, response: LLMResponse) -> Dict[str, Any]:
        ...


class Agent:
    """Agent that can process tasks using LLM."""

    def __init__(self, config: AgentConfig, prompt_loader: Optional[PromptLoader] = None):
        self.config = config
        self.busy = False
        self.llm_interface = create_llm_interface(config)
        self.prompt_loader = prompt_loader or PromptLoader()

    def _create_json_parser(self) -> OutputTransform:
        """Create a JSON parser that implements OutputTransform."""

        def transform_json(response: LLMResponse) -> Dict[str, Any]:
            return cast(Dict[str, Any], json.loads(response.content))

        return cast(OutputTransform, transform_json)

    async def process_task(self, subtask: SubTask) -> Any:
        """Process a single subtask using the configured LLM."""
        try:
            if isinstance(subtask.input_data, dict):
                result = await self._process_typed_task(subtask)
            else:
                # Fallback for untyped tasks
                response = await self.llm_interface.process(subtask.input_data)
                result = {
                    "status": "completed",
                    "data": response.content,
                    "metadata": response.metadata,
                }

            return {"status": "completed", "data": result, "task_id": subtask.id}
        except Exception as e:
            return {"status": "failed", "error": str(e), "task_id": subtask.id}

    async def _process_typed_task(self, subtask: SubTask) -> Dict[str, Any]:
        """Process a task using chains and prompt templates."""
        if not isinstance(subtask.input_data, dict):
            raise ValueError("Typed tasks require dictionary input data")

        # Get the prompt template for this task type
        prompt = self.prompt_loader.get_prompt(subtask.type)
        if not prompt:
            raise ValueError(f"No prompt template found for task type: {subtask.type}")

        # Create a chain for this task
        chain = SequentialChain[Dict[str, Any]](
            steps=[
                ChainStep[Dict[str, Any]](
                    task_type=subtask.type,
                    output_transform=self._create_json_parser()
                    if prompt.metadata.expected_response.format == "json"
                    else None,
                )
            ],
            llm=self.llm_interface,
            prompt_loader=self.prompt_loader,
        )

        # Execute the chain
        result = await chain.execute(**subtask.input_data)
        return result


class AgentPool:
    """Pool of agents that can be reused across tasks."""

    def __init__(self, config: Optional[AgentPoolConfig] = None):
        """Initialize the agent pool.

        Args:
            config: Optional agent pool configuration. If None, uses config from global config.
        """
        self.config = config or get_config().agents
        self.agents: Dict[str, Agent] = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_agents)
        self.tasks: Dict[str, asyncio.Task[Any]] = {}
        self.loop = asyncio.get_event_loop()

    def submit_task(self, subtask: SubTask) -> None:
        """Submit a subtask to be processed by an available agent."""
        # Get or create an agent for the task
        agent = self._get_or_create_agent(subtask.id)

        # Create and store the async task
        task = self.loop.create_task(agent.process_task(subtask))
        self.tasks[subtask.id] = task

    def _get_or_create_agent(self, task_id: str, agent_type: Optional[str] = None) -> Agent:
        """Get an existing agent or create a new one if possible.

        Args:
            task_id: Unique identifier for the task
            agent_type: Type of agent to create, must match a key in config.agent_types.
                       If None, uses the default agent type.

        Returns:
            An agent instance

        Raises:
            RuntimeError: If no agents are available and cannot create more
            ValueError: If specified agent_type doesn't exist in configuration
        """
        if task_id in self.agents:
            return self.agents[task_id]

        max_agents = self.config.agents.get("max_agents", 10)
        if len(self.agents) >= max_agents:
            raise RuntimeError("No available agents and cannot create more")

        # Remove the unused assignment and directly use config
        agent_config = AgentConfig(
            provider=self.config.agents.get("default_agent", {}).get("provider", "anthropic"),
            model_name=self.config.agents.get("default_agent", {}).get("model", "claude-3-sonnet-20240229"),
            max_tokens=self.config.agents.get("default_agent", {}).get("settings", {}).get("max_tokens", 1024),
            temperature=self.config.agents.get("default_agent", {}).get("settings", {}).get("temperature", 0.7),
            api_key=self.config.api_keys.get(
                self.config.agents.get("default_agent", {}).get("provider", "anthropic"), ""
            ),
            max_context_tokens=8192,
            token_tracking=True,
            cost_per_1k_tokens=None,
            top_p=1.0,
        )

        agent = Agent(agent_config)
        self.agents[task_id] = agent
        return agent

    def get_agent(self, task_id: str, agent_type: Optional[str] = None) -> Agent:
        """Get an agent for a task, optionally specifying the type of agent needed."""
        return self._get_or_create_agent(task_id, agent_type)

    def wait_for_result(self, task_id: str) -> Any:
        """Wait for a specific task to complete and return its result."""
        if task_id not in self.tasks:
            raise ValueError(f"No task found with id: {task_id}")

        task = self.tasks[task_id]
        result = self.loop.run_until_complete(task)

        # Cleanup
        del self.tasks[task_id]
        if task_id in self.agents:
            del self.agents[task_id]

        return result
