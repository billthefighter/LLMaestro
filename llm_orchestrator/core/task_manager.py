import uuid
from typing import Any, Dict, List, Optional

from .models import Task, TaskType, TaskStatus, SubTask
from ..agents.agent_pool import AgentPool
from ..utils.storage import StorageManager

class TaskManager:
    def __init__(self, storage_path: str = "./task_storage"):
        self.storage = StorageManager(storage_path)
        self.agent_pool = AgentPool()
        
    def create_task(self, task_type: TaskType, input_data: Any, config: Dict[str, Any]) -> Task:
        """Create a new task with the given parameters."""
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
        if task.type == TaskType.PDF_ANALYSIS:
            return self._decompose_pdf_task(task)
        elif task.type == TaskType.CODE_REFACTOR:
            return self._decompose_code_task(task)
        elif task.type == TaskType.LINT_FIX:
            return self._decompose_lint_task(task)
        else:
            raise ValueError(f"Unknown task type: {task.type}")
    
    def execute(self, task: Task) -> Any:
        """Execute a task by breaking it down and processing subtasks in parallel."""
        subtasks = self.decompose_task(task)
        task.subtasks = subtasks
        task.status = TaskStatus.IN_PROGRESS
        self.storage.save_task(task)
        
        # Process subtasks in parallel using the agent pool
        for subtask in subtasks:
            self.agent_pool.submit_task(subtask)
        
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
    
    def _decompose_pdf_task(self, task: Task) -> List[SubTask]:
        """Break down PDF analysis task into chunks."""
        chunk_size = task.config.get("chunk_size", 1000)
        # Implementation details for PDF chunking
        return []
    
    def _decompose_code_task(self, task: Task) -> List[SubTask]:
        """Break down code refactoring task into modules/files."""
        # Implementation details for code chunking
        return []
    
    def _decompose_lint_task(self, task: Task) -> List[SubTask]:
        """Break down lint fixing task into separate files/errors."""
        # Implementation details for lint error chunking
        return []
    
    def _aggregate_results(self, task: Task, results: List[Any]) -> Any:
        """Combine subtask results based on task type."""
        if task.type == TaskType.PDF_ANALYSIS:
            return self._aggregate_pdf_results(results)
        elif task.type == TaskType.CODE_REFACTOR:
            return self._aggregate_code_results(results)
        elif task.type == TaskType.LINT_FIX:
            return self._aggregate_lint_results(results)
        else:
            raise ValueError(f"Unknown task type: {task.type}")
    
    def _aggregate_pdf_results(self, results: List[Any]) -> Dict[str, Any]:
        """Combine PDF analysis results."""
        return {"categories": results}
    
    def _aggregate_code_results(self, results: List[Any]) -> Dict[str, Any]:
        """Combine code refactoring results."""
        return {"refactored_files": results}
    
    def _aggregate_lint_results(self, results: List[Any]) -> Dict[str, Any]:
        """Combine lint fixing results."""
        return {"fixed_files": results} 