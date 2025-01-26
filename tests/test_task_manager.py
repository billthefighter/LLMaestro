import pytest
from unittest.mock import Mock, patch
from typing import Dict, List, Any

from llm_orchestrator.core.models import Task, SubTask, TaskStatus
from llm_orchestrator.core.task_manager import TaskManager
from llm_orchestrator.prompts.loader import PromptLoader

@pytest.fixture
def task_manager():
    return TaskManager("./test_storage")

@pytest.fixture
def mock_prompt_loader():
    loader = Mock(spec=PromptLoader)
    
    # Mock prompt for test_task
    test_prompt = Mock()
    test_prompt.metadata.decomposition.strategy = "custom"
    test_prompt.description = "Test task description"
    test_prompt.user_prompt = "Test user prompt"
    test_prompt.metadata.expected_response.schema = {"type": "object"}
    test_prompt.metadata.model_requirements = {"min_tokens": 1000}
    
    # Mock prompt for task_decomposer
    decomposer_prompt = Mock()
    decomposer_prompt.metadata.decomposition.strategy = "custom"
    
    def get_prompt(task_type: str):
        if task_type == "test_task":
            return test_prompt
        elif task_type == "task_decomposer":
            return decomposer_prompt
        return None
        
    loader.get_prompt.side_effect = get_prompt
    loader.format_prompt.return_value = ("system prompt", "user prompt")
    return loader

def test_dynamic_decomposition(task_manager, mock_prompt_loader):
    """Test dynamic task decomposition using the task_decomposer."""
    task_manager.prompt_loader = mock_prompt_loader
    
    # Mock agent pool responses
    mock_strategy = {
        "strategy": {
            "name": "test_strategy",
            "description": "Test strategy",
            "max_parallel": 2
        },
        "decomposition": {
            "method": """
def decompose_test_strategy(task: Task) -> List[SubTask]:
    # Simple strategy that creates two subtasks
    data = task.input_data
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary")
        
    return [
        SubTask(
            id=str(uuid.uuid4()),
            type=task.type,
            input_data={"part": 1, "data": data.get("first_half")},
            parent_task_id=task.id
        ),
        SubTask(
            id=str(uuid.uuid4()),
            type=task.type,
            input_data={"part": 2, "data": data.get("second_half")},
            parent_task_id=task.id
        )
    ]
""",
            "aggregation": """
def aggregate_test_strategy_results(results: List[Any]) -> Dict[str, Any]:
    # Simple strategy that combines results
    combined = {
        "parts": [],
        "total": 0
    }
    for result in results:
        if isinstance(result, dict):
            combined["parts"].append(result)
            combined["total"] += len(result.get("data", []))
    return combined
"""
        }
    }
    
    def mock_wait_for_result(task_id: str) -> Any:
        return mock_strategy
        
    task_manager.agent_pool.wait_for_result = Mock(side_effect=mock_wait_for_result)
    
    # Create and decompose a test task
    task = task_manager.create_task(
        "test_task",
        {
            "first_half": ["a", "b"],
            "second_half": ["c", "d"]
        },
        {}
    )
    
    subtasks = task_manager.decompose_task(task)
    
    # Verify decomposition
    assert len(subtasks) == 2
    assert all(isinstance(st, SubTask) for st in subtasks)
    assert all(st.type == "test_task" for st in subtasks)
    assert subtasks[0].input_data["part"] == 1
    assert subtasks[1].input_data["part"] == 2

def test_dynamic_decomposition_caching(task_manager, mock_prompt_loader):
    """Test that dynamic decomposition strategies are cached."""
    task_manager.prompt_loader = mock_prompt_loader
    
    # Mock strategy response
    mock_strategy = {
        "strategy": {"name": "cached_strategy"},
        "decomposition": {
            "method": """
def decompose_cached_strategy(task: Task) -> List[SubTask]:
    return [SubTask(id=str(uuid.uuid4()), type=task.type, input_data={}, parent_task_id=task.id)]
""",
            "aggregation": """
def aggregate_cached_strategy_results(results: List[Any]) -> Dict[str, Any]:
    return {"results": results}
"""
        }
    }
    
    task_manager.agent_pool.wait_for_result = Mock(return_value=mock_strategy)
    
    # Create and decompose first task
    task1 = task_manager.create_task("test_task", {}, {})
    task_manager.decompose_task(task1)
    
    # Create and decompose second task
    task2 = task_manager.create_task("test_task", {}, {})
    task_manager.decompose_task(task2)
    
    # Verify strategy was cached
    assert task_manager.agent_pool.wait_for_result.call_count == 1
    assert "test_task" in task_manager._dynamic_strategies

def test_dynamic_decomposition_error_handling(task_manager, mock_prompt_loader):
    """Test error handling in dynamic decomposition."""
    task_manager.prompt_loader = mock_prompt_loader
    
    # Mock invalid strategy response
    mock_strategy = {
        "strategy": {"name": "invalid_strategy"},
        "decomposition": {
            "method": "invalid python code",
            "aggregation": "invalid python code"
        }
    }
    
    task_manager.agent_pool.wait_for_result = Mock(return_value=mock_strategy)
    
    # Create task and attempt decomposition
    task = task_manager.create_task("test_task", {}, {})
    
    with pytest.raises(ValueError) as exc_info:
        task_manager.decompose_task(task)
    assert "Failed to execute dynamic decomposition" in str(exc_info.value)

def test_dynamic_aggregation(task_manager, mock_prompt_loader):
    """Test dynamic result aggregation."""
    task_manager.prompt_loader = mock_prompt_loader
    
    # Set up mock strategy
    mock_strategy = {
        "strategy": {"name": "test_strategy"},
        "decomposition": {
            "method": """
def decompose_test_strategy(task: Task) -> List[SubTask]:
    return [SubTask(id=str(uuid.uuid4()), type=task.type, input_data={}, parent_task_id=task.id)]
""",
            "aggregation": """
def aggregate_test_strategy_results(results: List[Any]) -> Dict[str, Any]:
    return {"combined": sum(r.get("value", 0) for r in results)}
"""
        }
    }
    
    task_manager._dynamic_strategies["test_task"] = mock_strategy
    
    # Test aggregation
    results = [{"value": 1}, {"value": 2}, {"value": 3}]
    task = task_manager.create_task("test_task", {}, {})
    
    aggregated = task_manager._aggregate_dynamic("test_task", results)
    assert aggregated == {"combined": 6} 