"""Task management for the application."""

import textwrap
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from typing_extensions import TypedDict

from llmaestro.llm.models import ModelCapabilities, ModelDescriptor
from llmaestro.prompts.base import BasePrompt, VersionInfo

from .conversations import ConversationGraph, LLMResponse
from .models import DecompositionConfig, SubTask, Task, TaskStatus, TokenUsage
from .storage import Artifact, ArtifactStorage, FileSystemArtifactStorage

if TYPE_CHECKING:
    from ..agents.agent_pool import AgentPool


class DecompositionConfig(TypedDict, total=False):
    """Configuration for task decomposition."""

    strategy: str
    chunk_size: int
    max_parallel: int
    aggregation: str


class DecompositionStrategy(ABC):
    """Abstract base class for task decomposition strategies."""

    @abstractmethod
    def decompose(self, task: Task, config: DecompositionConfig) -> List[SubTask]:
        """Break down a task into subtasks."""
        pass

    @abstractmethod
    def aggregate(self, results: List[Any]) -> Any:
        """Combine subtask results."""
        pass


class ChunkStrategy(DecompositionStrategy):
    """Strategy for breaking down tasks into chunks."""

    def decompose(self, task: Task, config: DecompositionConfig) -> List[SubTask]:
        chunk_size = config.get("chunk_size", 1000)
        if not isinstance(chunk_size, int):
            chunk_size = 1000

        if isinstance(task.input_data, str):
            chunks = [task.input_data[i : i + chunk_size] for i in range(0, len(task.input_data), chunk_size)]
        elif isinstance(task.input_data, list):
            chunks = [task.input_data[i : i + chunk_size] for i in range(0, len(task.input_data), chunk_size)]
        else:
            raise ValueError("Input data must be string or list for chunk strategy")

        return [
            SubTask(
                id=str(uuid.uuid4()),
                type=task.type,
                input_data=str(chunk) if isinstance(chunk, str) else {"data": chunk},
                parent_task_id=task.id,
            )
            for chunk in chunks
        ]

    def aggregate(self, results: List[Any]) -> str:
        return "\n".join(str(r) for r in results)


class FileStrategy(DecompositionStrategy):
    """Strategy for processing multiple files."""

    def decompose(self, task: Task, config: DecompositionConfig) -> List[SubTask]:
        if not isinstance(task.input_data, (list, dict)):
            raise ValueError("Input data must be list or dict of files for file strategy")

        files = task.input_data if isinstance(task.input_data, list) else list(task.input_data.items())

        return [
            SubTask(id=str(uuid.uuid4()), type=task.type, input_data={"file_data": file_data}, parent_task_id=task.id)
            for file_data in files
        ]

    def aggregate(self, results: List[Any]) -> Dict[str, Any]:
        merged = {}
        for result in results:
            if isinstance(result, dict):
                for key, value in result.items():
                    if key not in merged:
                        merged[key] = []
                    if isinstance(value, list):
                        merged[key].extend(value)
                    else:
                        merged[key].append(value)
        return merged


class ErrorStrategy(DecompositionStrategy):
    """Strategy for processing multiple errors."""

    def decompose(self, task: Task, config: DecompositionConfig) -> List[SubTask]:
        if not isinstance(task.input_data, (list, dict)):
            raise ValueError("Input data must be list or dict of errors for error strategy")

        errors = task.input_data if isinstance(task.input_data, list) else list(task.input_data.items())

        return [
            SubTask(id=str(uuid.uuid4()), type=task.type, input_data={"error_data": error_data}, parent_task_id=task.id)
            for error_data in errors
        ]

    def aggregate(self, results: List[Any]) -> Dict[str, Any]:
        return self._merge_results(results)

    def _merge_results(self, results: List[Any]) -> Dict[str, Any]:
        merged = {}
        for result in results:
            if isinstance(result, dict):
                for key, value in result.items():
                    if key not in merged:
                        merged[key] = []
                    if isinstance(value, list):
                        merged[key].extend(value)
                    else:
                        merged[key].append(value)
        return merged


class DynamicStrategy(DecompositionStrategy):
    """Strategy that generates decomposition logic at runtime."""

    def __init__(self, strategy_code: Dict[str, Any]):
        self.strategy_code = strategy_code
        self._namespace = {
            "Task": Task,
            "SubTask": SubTask,
            "uuid": uuid,
            "List": List,
            "Dict": Dict,
            "Any": Any,
        }

    def decompose(self, task: Task, config: DecompositionConfig) -> List[SubTask]:
        try:
            exec(textwrap.dedent(self.strategy_code["decomposition"]["method"]), self._namespace)
            decompose_func = self._namespace[f"decompose_{self.strategy_code['strategy']['name']}"]
            return decompose_func(task)
        except Exception as e:
            raise ValueError(f"Failed to execute dynamic decomposition: {e}") from e

    def aggregate(self, results: List[Any]) -> Any:
        try:
            exec(textwrap.dedent(self.strategy_code["decomposition"]["aggregation"]), self._namespace)
            aggregate_func = self._namespace[f"aggregate_{self.strategy_code['strategy']['name']}_results"]
            return aggregate_func(results)
        except Exception as e:
            raise ValueError(f"Failed to execute dynamic aggregation: {e}") from e


class TaskManager:
    """Manages task execution and tracking."""

    _strategies = {"chunk": ChunkStrategy(), "file": FileStrategy(), "error": ErrorStrategy()}

    def __init__(self, storage: Optional[ArtifactStorage] = None):
        self.storage = storage if storage is not None else FileSystemArtifactStorage.create(Path("./task_storage"))
        self._dynamic_strategies: Dict[str, DynamicStrategy] = {}
        self.tasks: Dict[str, SubTask] = {}
        self._agent_pool: Optional["AgentPool"] = None

    def set_agent_pool(self, agent_pool: "AgentPool") -> None:
        """Set the agent pool for task execution."""
        self._agent_pool = agent_pool

    async def submit_task(self, task: SubTask) -> None:
        """Submit a task for execution."""
        if not self._agent_pool:
            raise RuntimeError("Agent pool not set")
        self.tasks[task.id] = task
        await self._agent_pool.submit_task(task)

    def get_task(self, task_id: str) -> Optional[SubTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    async def wait_for_result(self, task_id: str) -> Any:
        """Wait for a task to complete and return its result."""
        if not self._agent_pool:
            raise RuntimeError("Agent pool not set")
        return await self._agent_pool.wait_for_result(task_id)

    def create_task(
        self, task_type: str, input_data: Union[BasePrompt, Dict[str, Any], str, List[Any]], config: Dict[str, Any]
    ) -> Task:
        """Create a new task with the given parameters."""
        task = Task(
            id=str(uuid.uuid4()),
            type=task_type,
            input_data=input_data,
            config=config,
            decomposition_config=config.get("decomposition"),
        )

        # Convert task to artifact and save
        artifact = Artifact(
            id=task.id,
            name=f"task_{task.id}",
            content_type="task",
            data=task.model_dump(),
        )
        self.storage.save_artifact(artifact)
        return task

    def save_task(self, task: Task) -> None:
        """Save a task to storage."""
        artifact = Artifact(
            id=task.id,
            name=f"task_{task.id}",
            content_type="task",
            data=task.model_dump(),
        )
        self.storage.save_artifact(artifact)

    def load_task(self, task_id: str) -> Optional[Task]:
        """Load a task from storage."""
        artifact = self.storage.load_artifact(task_id)
        if artifact and artifact.content_type == "task":
            return Task.model_validate(artifact.data)
        return None

    async def decompose_task(self, task: Task) -> List[SubTask]:
        """Break down a large task into smaller subtasks based on task type."""
        if not task.decomposition_config:
            # If no decomposition config, treat as single task
            return [SubTask(id=str(uuid.uuid4()), type=task.type, input_data=task.input_data, parent_task_id=task.id)]

        strategy_name = task.decomposition_strategy
        if not strategy_name:
            raise ValueError("No decomposition strategy specified in task config")

        if strategy_name == "custom":
            strategy = await self._get_dynamic_strategy(task)
            return strategy.decompose(task, task.decomposition_config)

        strategy = self._strategies.get(strategy_name)
        if not strategy:
            raise ValueError(f"Unknown decomposition strategy: {strategy_name}")

        return strategy.decompose(task, task.decomposition_config)

    async def _get_dynamic_strategy(self, task: Task) -> DynamicStrategy:
        """Get or create a dynamic strategy for a task type."""
        if task.type not in self._dynamic_strategies:
            strategy_code = await self._generate_dynamic_strategy(task)
            self._dynamic_strategies[task.type] = DynamicStrategy(strategy_code)
        return self._dynamic_strategies[task.type]

    async def _generate_dynamic_strategy(self, task: Task) -> Dict[str, Any]:
        """Generate a new dynamic strategy using the task decomposer."""
        if not task.is_llm_task:
            raise ValueError("Dynamic strategy generation requires a BasePrompt input")

        decomposer_task = SubTask(
            id=str(uuid.uuid4()),
            type="task_decomposer",
            input_data={"system_prompt": task.input_data.system_prompt, "user_prompt": task.input_data.user_prompt},
            parent_task_id=task.id,
        )

        if not self._agent_pool:
            raise RuntimeError("Agent pool not set")

        await self._agent_pool.submit_task(decomposer_task)
        return await self._agent_pool.wait_for_result(decomposer_task.id)

    async def execute(self, task: Task, conversation: Optional[ConversationGraph] = None) -> Any:
        """Execute a task and optionally integrate with a conversation graph."""
        if not self._agent_pool:
            raise RuntimeError("Agent pool not set")

        # Create conversation if none provided
        if not conversation:
            conversation = ConversationGraph(id=str(uuid.uuid4()))

        # Decompose task into subtasks
        subtasks = await self.decompose_task(task)
        task.subtasks = subtasks
        task.status = TaskStatus.IN_PROGRESS
        self.save_task(task)

        # Process subtasks and track in conversation
        results = []
        for subtask in subtasks:
            # Initialize node IDs
            prompt_node_id = None
            response_node_id = None

            # Add subtask to conversation if it's an LLM task
            if task.is_llm_task:
                # Convert input_data to BasePrompt if needed
                if isinstance(subtask.input_data, dict):
                    system_prompt = str(subtask.input_data.get("system_prompt", ""))
                    user_prompt = str(subtask.input_data.get("user_prompt", ""))
                    prompt_content = BasePrompt(
                        name=f"subtask_{subtask.id}",
                        description="Generated subtask prompt",
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        metadata={},
                        current_version=VersionInfo(
                            number="1.0.0",
                            author="system",
                            timestamp=datetime.now(),
                            description="Generated prompt",
                            change_type=ChangeType.CREATED,
                        ),
                    )
                elif isinstance(subtask.input_data, BasePrompt):
                    prompt_content = subtask.input_data
                else:
                    # For non-dict, non-BasePrompt input, create a simple prompt
                    prompt_content = BasePrompt(
                        name=f"subtask_{subtask.id}",
                        description="Generated subtask prompt",
                        system_prompt="Process the following input",
                        user_prompt=str(subtask.input_data),
                        metadata={},
                        current_version=VersionInfo(
                            number="1.0.0",
                            author="system",
                            timestamp=datetime.now(),
                            description="Generated prompt",
                            change_type=ChangeType.CREATED,
                        ),
                    )

                prompt_node_id = conversation.add_node(
                    content=prompt_content, node_type="prompt", metadata={"task_id": subtask.id}
                )

            # Process subtask
            await self._agent_pool.submit_task(subtask)
            result = await self._agent_pool.wait_for_result(subtask.id)
            results.append(result)

            # Add result to conversation if it's an LLM task
            if task.is_llm_task and prompt_node_id is not None:
                # Convert result to LLMResponse if needed
                if isinstance(result, LLMResponse):
                    response_content = result
                else:
                    # Create a basic LLMResponse for non-LLM results
                    response_content = LLMResponse(
                        content=str(result),
                        success=True,
                        model=ModelDescriptor(name="unknown", family="unknown", capabilities=ModelCapabilities()),
                        token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                    )

                response_node_id = conversation.add_node(
                    content=response_content, node_type="response", metadata={"task_id": subtask.id}
                )
                conversation.add_edge(prompt_node_id, response_node_id, "response_to")

        # Aggregate results
        task.result = await self._aggregate_results(task, results)
        task.status = TaskStatus.COMPLETED
        self.save_task(task)

        return task.result

    async def _aggregate_results(self, task: Task, results: List[Any]) -> Any:
        """Combine subtask results based on aggregation strategy."""
        if not task.decomposition_config:
            # If no decomposition config, return first result
            return results[0] if results else None

        strategy_name = task.decomposition_strategy
        if not strategy_name:
            return results[0] if results else None

        if strategy_name == "custom":
            strategy = await self._get_dynamic_strategy(task)
            return strategy.aggregate(results)

        strategy = self._strategies.get(strategy_name)
        if not strategy:
            raise ValueError(f"Unknown decomposition strategy: {strategy_name}")

        return strategy.aggregate(results)
