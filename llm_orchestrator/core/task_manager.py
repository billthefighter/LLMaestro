import uuid
from typing import Any, Dict, List, Optional, TypedDict, cast
from pathlib import Path
import inspect
import textwrap

from .models import Task, TaskStatus, SubTask
from ..agents.agent_pool import AgentPool
from ..utils.storage import StorageManager
from ..prompts.loader import PromptLoader

class DecompositionConfig(TypedDict, total=False):
    """Configuration for task decomposition."""
    strategy: str
    chunk_size: int
    max_parallel: int
    aggregation: str

class TaskManager:
    def __init__(self, storage_path: str = "./task_storage"):
        self.storage = StorageManager(storage_path)
        self.agent_pool = AgentPool()
        self.prompt_loader = PromptLoader()
        self._dynamic_strategies: Dict[str, Dict[str, Any]] = {}
        
    def create_task(self, task_type: str, input_data: Any, config: Dict[str, Any]) -> Task:
        """Create a new task with the given parameters."""
        # Validate task type exists in prompt loader
        if not self.prompt_loader.get_prompt(task_type):
            raise ValueError(f"Unknown task type: {task_type}")
            
        task = Task(
            id=str(uuid.uuid4()),
            type=task_type,
            input_data=input_data,
            config=config
        )
        self.storage.save_task(task)
        return task
    
    def decompose_task(self, task: Task) -> List[SubTask]:
        """Break down a large task into smaller subtasks based on task type."""
        # Get the prompt template for this task type
        prompt = self.prompt_loader.get_prompt(task.type)
        if not prompt:
            raise ValueError(f"No prompt template found for task type: {task.type}")
            
        # Get decomposition strategy from prompt metadata
        strategy = prompt.metadata.decomposition.strategy
        
        if strategy == "custom":
            return self._decompose_dynamic(task)
        
        # Call the appropriate decomposition method based on strategy
        decompose_method = f"_decompose_{strategy}"
        if hasattr(self, decompose_method):
            return getattr(self, decompose_method)(task, prompt.metadata.decomposition)
        else:
            raise ValueError(f"No decomposition method found for strategy: {strategy}")
    
    def _decompose_dynamic(self, task: Task) -> List[SubTask]:
        """Generate and execute a dynamic decomposition strategy."""
        # Check if we already have a strategy for this task type
        if task.type in self._dynamic_strategies:
            strategy = self._dynamic_strategies[task.type]
        else:
            # Get the task decomposer prompt
            decomposer = self.prompt_loader.get_prompt("task_decomposer")
            if not decomposer:
                raise ValueError("Task decomposer prompt not found")
            
            # Get task metadata
            task_prompt = self.prompt_loader.get_prompt(task.type)
            if not task_prompt:
                raise ValueError(f"No prompt found for task type: {task.type}")
            
            # Format the decomposer prompt
            system_prompt, user_prompt = self.prompt_loader.format_prompt(
                "task_decomposer",
                task_metadata={
                    "type": task.type,
                    "description": task_prompt.description,
                    "input_format": str(task_prompt.user_prompt),
                    "expected_output": task_prompt.metadata.expected_response.schema,
                    "requirements": task_prompt.metadata.model_requirements
                }
            )
            
            # Get decomposition strategy from LLM
            decomposer_task = Task(
                id=str(uuid.uuid4()),
                type="task_decomposer",
                input_data={
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt
                }
            )
            self.agent_pool.submit_task(decomposer_task)
            result = self.agent_pool.wait_for_result(decomposer_task.id)
            
            # Store the strategy
            self._dynamic_strategies[task.type] = result
            strategy = result
        
        # Execute the dynamic decomposition
        try:
            # Create a local namespace with required imports
            namespace = {
                'Task': Task,
                'SubTask': SubTask,
                'uuid': uuid,
                'List': List,
                'Dict': Dict,
                'Any': Any
            }
            
            # Execute the decomposition method
            exec(textwrap.dedent(strategy["decomposition"]["method"]), namespace)
            decompose_func = namespace[f"decompose_{strategy['strategy']['name']}"]
            
            return decompose_func(task)
        except Exception as e:
            raise ValueError(f"Failed to execute dynamic decomposition: {e}")
    
    def execute(self, task: Task) -> Any:
        """Execute a task by breaking it down and processing subtasks in parallel."""
        # Get the prompt template for this task type
        prompt = self.prompt_loader.get_prompt(task.type)
        if not prompt:
            raise ValueError(f"No prompt template found for task type: {task.type}")
            
        # Get max parallel subtasks from metadata
        decomp_config = cast(DecompositionConfig, prompt.metadata.decomposition)
        max_parallel = decomp_config.get('max_parallel')
        
        # Decompose task into subtasks
        subtasks = self.decompose_task(task)
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
    
    def _decompose_chunk(self, task: Task, decomp_config: DecompositionConfig) -> List[SubTask]:
        """Break down task into chunks of specified size."""
        chunk_size = decomp_config.get('chunk_size', 1000)  # Default to 1000 if not specified
        if not isinstance(chunk_size, int):
            chunk_size = 1000  # Fallback if not an integer
            
        if isinstance(task.input_data, str):
            # Split text into chunks
            chunks = [task.input_data[i:i+chunk_size] 
                     for i in range(0, len(task.input_data), chunk_size)]
        elif isinstance(task.input_data, list):
            # Split list into chunks
            chunks = [task.input_data[i:i+chunk_size] 
                     for i in range(0, len(task.input_data), chunk_size)]
        else:
            raise ValueError("Input data must be string or list for chunk strategy")
            
        return [
            SubTask(
                id=str(uuid.uuid4()),
                type=task.type,
                input_data=chunk,
                parent_task_id=task.id
            )
            for chunk in chunks
        ]
    
    def _decompose_file(self, task: Task, decomp_config: DecompositionConfig) -> List[SubTask]:
        """Break down task by files."""
        if not isinstance(task.input_data, (list, dict)):
            raise ValueError("Input data must be list or dict of files for file strategy")
            
        files = task.input_data if isinstance(task.input_data, list) else list(task.input_data.items())
        
        return [
            SubTask(
                id=str(uuid.uuid4()),
                type=task.type,
                input_data=file_data,
                parent_task_id=task.id
            )
            for file_data in files
        ]
    
    def _decompose_error(self, task: Task, decomp_config: DecompositionConfig) -> List[SubTask]:
        """Break down task by individual errors."""
        if not isinstance(task.input_data, (list, dict)):
            raise ValueError("Input data must be list or dict of errors for error strategy")
            
        errors = task.input_data if isinstance(task.input_data, list) else list(task.input_data.items())
        
        return [
            SubTask(
                id=str(uuid.uuid4()),
                type=task.type,
                input_data=error_data,
                parent_task_id=task.id
            )
            for error_data in errors
        ]
    
    def _aggregate_results(self, task: Task, results: List[Any]) -> Any:
        """Combine subtask results based on aggregation strategy."""
        # Get the prompt template for this task type
        prompt = self.prompt_loader.get_prompt(task.type)
        if not prompt:
            raise ValueError(f"No prompt template found for task type: {task.type}")
            
        # Get aggregation strategy from prompt metadata
        strategy = prompt.metadata.decomposition.aggregation
        
        if strategy == "custom" and task.type in self._dynamic_strategies:
            return self._aggregate_dynamic(task.type, results)
            
        # Call the appropriate aggregation method based on strategy
        aggregate_method = f"_aggregate_{strategy}"
        if hasattr(self, aggregate_method):
            return getattr(self, aggregate_method)(results)
        else:
            raise ValueError(f"No aggregation method found for strategy: {strategy}")
    
    def _aggregate_dynamic(self, task_type: str, results: List[Any]) -> Any:
        """Execute dynamic aggregation strategy."""
        strategy = self._dynamic_strategies[task_type]
        try:
            # Create a local namespace with required imports
            namespace = {
                'List': List,
                'Dict': Dict,
                'Any': Any,
                'results': results
            }
            
            # Execute the aggregation method
            exec(textwrap.dedent(strategy["decomposition"]["aggregation"]), namespace)
            aggregate_func = namespace[f"aggregate_{strategy['strategy']['name']}_results"]
            
            return aggregate_func(results)
        except Exception as e:
            raise ValueError(f"Failed to execute dynamic aggregation: {e}")
    
    def _aggregate_concatenate(self, results: List[Any]) -> str:
        """Simple concatenation of results."""
        return "\n".join(str(r) for r in results)
    
    def _aggregate_merge(self, results: List[Any]) -> Dict[str, Any]:
        """Merge dictionaries of results."""
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
    
    def _aggregate_custom(self, results: List[Any]) -> Any:
        """Custom aggregation - should be overridden by subclasses."""
        raise NotImplementedError("Custom aggregation strategy must be implemented in subclass") 