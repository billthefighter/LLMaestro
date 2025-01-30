import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Protocol, TypeVar, cast

from ..core.models import AgentConfig, SubTask
from ..llm.chains import ChainStep, OutputTransform, SequentialChain
from ..llm.interfaces import LLMResponse, create_llm_interface
from ..prompts.loader import PromptLoader

T = TypeVar("T")


class JsonOutputTransform(OutputTransform, Protocol):
    """Protocol for JSON output transformation."""

    def __call__(self, response: LLMResponse) -> Dict[str, Any]:
        ...


class Agent:
    def __init__(self, config: AgentConfig, prompt_loader: Optional[PromptLoader] = None):
        self.config = config
        self.busy = False
        self.llm_interface = create_llm_interface(config)
        self.prompt_loader = prompt_loader or PromptLoader()

    def _create_json_parser(self) -> JsonOutputTransform:
        """Create a JSON parser that conforms to OutputTransform protocol."""

        def parse_json(response: LLMResponse) -> Dict[str, Any]:
            try:
                return response.json()
            except Exception as e:
                return {
                    "error": f"Failed to parse JSON response: {str(e)}",
                    "raw_content": response.content,
                }

        return cast(JsonOutputTransform, parse_json)

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
    def __init__(self, max_agents: int = 5):
        self.max_agents = max_agents
        self.agents: Dict[str, Agent] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_agents)
        self.tasks: Dict[str, asyncio.Task] = {}
        self.loop = asyncio.get_event_loop()

    def submit_task(self, subtask: SubTask) -> None:
        """Submit a subtask to be processed by an available agent."""
        # Get or create an agent for the task
        agent = self._get_or_create_agent(subtask.id)

        # Create and store the async task
        task = self.loop.create_task(agent.process_task(subtask))
        self.tasks[subtask.id] = task

    def _get_or_create_agent(self, task_id: str) -> Agent:
        """Get an existing agent or create a new one if possible."""
        if task_id in self.agents:
            return self.agents[task_id]

        if len(self.agents) < self.max_agents:
            agent = Agent(AgentConfig(model_name="gpt-4", max_tokens=8192))
            self.agents[task_id] = agent
            return agent

        raise RuntimeError("No available agents and cannot create more")

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
