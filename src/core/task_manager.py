import textwrap
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypedDict, cast

from ..agents.agent_pool import AgentPool
from ..prompts.loader import PromptLoader
from ..utils.storage import StorageManager
from .models import SubTask, Task, TaskStatus


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
    """Manager for task decomposition and execution."""

    _strategies = {"chunk": ChunkStrategy(), "file": FileStrategy(), "error": ErrorStrategy()}

    def __init__(self, storage_path: str = "./task_storage"):
        self.storage = StorageManager(storage_path)
        self.agent_pool = AgentPool()
        self.prompt_loader = PromptLoader()
        self._dynamic_strategies: Dict[str, DynamicStrategy] = {}

    def create_task(self, task_type: str, input_data: Any, config: Dict[str, Any]) -> Task:
        """Create a new task with the given parameters."""
        # Validate task type exists in prompt loader
        if not self.prompt_loader.get_prompt(task_type):
            raise ValueError(f"Unknown task type: {task_type}")

        task = Task(id=str(uuid.uuid4()), type=task_type, input_data=input_data, config=config)
        self.storage.save_task(task)
        return task

    def decompose_task(self, task: Task, config: DecompositionConfig) -> List[SubTask]:
        """Break down a large task into smaller subtasks based on task type."""
        prompt = self.prompt_loader.get_prompt(task.type)
        if not prompt:
            raise ValueError(f"No prompt template found for task type: {task.type}")

        decomp_config = cast(DecompositionConfig, prompt.metadata.decomposition)
        if not decomp_config or not isinstance(decomp_config, dict):
            raise ValueError("Invalid decomposition configuration")

        strategy_name = decomp_config.get("strategy")
        if not strategy_name:
            raise ValueError("No decomposition strategy specified")

        if strategy_name == "custom":
            return self._get_dynamic_strategy(task).decompose(task, cast(DecompositionConfig, decomp_config))

        strategy = self._strategies.get(strategy_name)
        if not strategy:
            raise ValueError(f"Unknown decomposition strategy: {strategy_name}")

        return strategy.decompose(task, cast(DecompositionConfig, decomp_config))

    def _get_dynamic_strategy(self, task: Task) -> DynamicStrategy:
        """Get or create a dynamic strategy for a task type."""
        if task.type not in self._dynamic_strategies:
            strategy_code = self._generate_dynamic_strategy(task)
            self._dynamic_strategies[task.type] = DynamicStrategy(strategy_code)
        return self._dynamic_strategies[task.type]

    def _generate_dynamic_strategy(self, task: Task) -> Dict[str, Any]:
        """Generate a new dynamic strategy using the task decomposer."""
        decomposer = self.prompt_loader.get_prompt("task_decomposer")
        if not decomposer:
            raise ValueError("Task decomposer prompt not found")

        task_prompt = self.prompt_loader.get_prompt(task.type)
        if not task_prompt:
            raise ValueError(f"No prompt found for task type: {task.type}")

        try:
            system_prompt, user_prompt = self.prompt_loader.format_prompt(
                "task_decomposer",
                task_metadata={
                    "type": task.type,
                    "description": task_prompt.description,
                    "input_format": str(task_prompt.user_prompt),
                    "expected_output": task_prompt.metadata.expected_response.schema,
                    "requirements": task_prompt.metadata.model_requirements,
                },
            )
        except AttributeError:
            system_prompt = "System prompt"
            user_prompt = "User prompt"

        decomposer_task = SubTask(
            id=str(uuid.uuid4()),
            type="task_decomposer",
            input_data={"system_prompt": system_prompt, "user_prompt": user_prompt},
            parent_task_id=task.id,
        )

        self.agent_pool.submit_task(decomposer_task)
        return self.agent_pool.wait_for_result(decomposer_task.id)

    def execute(self, task: Task) -> Any:
        """Execute a task by breaking it down and processing subtasks in parallel."""
        # Get the prompt template for this task type
        prompt = self.prompt_loader.get_prompt(task.type)
        if not prompt:
            raise ValueError(f"No prompt template found for task type: {task.type}")

        # Get decomposition config from metadata
        decomp_config = cast(DecompositionConfig, prompt.metadata.decomposition)

        # Decompose task into subtasks
        subtasks = self.decompose_task(task, decomp_config)
        task.subtasks = subtasks
        task.status = TaskStatus.IN_PROGRESS
        self.storage.save_task(task)

        # Process subtasks in parallel using the agent pool
        for subtask in subtasks:
            self.agent_pool.submit_task(subtask)  # max_parallel handled by agent pool

        # Wait for all subtasks to complete
        results = []
        for subtask in subtasks:
            result = self.agent_pool.wait_for_result(subtask.id)
            results.append(result)

        # Aggregate results
        task.result = self._aggregate_results(task, results)
        task.status = TaskStatus.COMPLETED
        self.storage.save_task(task)

        return task.result

    def _aggregate_results(self, task: Task, results: List[Any]) -> Any:
        """Combine subtask results based on aggregation strategy."""
        prompt = self.prompt_loader.get_prompt(task.type)
        if not prompt:
            raise ValueError(f"No prompt template found for task type: {task.type}")

        decomp_config = cast(DecompositionConfig, prompt.metadata.decomposition)
        if not decomp_config or not isinstance(decomp_config, dict):
            raise ValueError("Invalid decomposition configuration")

        strategy_name = decomp_config.get("strategy")
        if not strategy_name:
            raise ValueError("No decomposition strategy specified")

        if strategy_name == "custom":
            return self._get_dynamic_strategy(task).aggregate(results)

        strategy = self._strategies.get(strategy_name)
        if not strategy:
            raise ValueError(f"Unknown decomposition strategy: {strategy_name}")

        return strategy.aggregate(results)
