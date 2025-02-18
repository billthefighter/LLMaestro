import pytest
import uuid
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock
from datetime import datetime

from llmaestro.core.task_manager import (
    TaskManager, Task, SubTask, TaskStatus,
    DecompositionConfig, DynamicStrategy
)
from llmaestro.prompts.loader import PromptLoader
from llmaestro.agents.agent_pool import AgentPool
from llmaestro.prompts.types import PromptMetadata, ResponseFormat


@pytest.fixture
def mock_prompt_loader():
    class MockPrompt:
        def __init__(self):
            self.metadata = PromptMetadata(
                type="test_task",
                expected_response=ResponseFormat(
                    format="json",
                    schema='{"type": "string"}'
                ),
                model_requirements={"model": "test-model"},
                decomposition={
                    "strategy": "chunk",
                    "chunk_size": 100,
                    "max_parallel": 2,
                    "aggregation": "concatenate"
                }
            )
            self.description = "Test prompt"
            self.user_prompt = "Test user prompt {var}"

        def format_prompt(self, task_type: str, task_metadata: dict) -> Tuple[str, str]:
            return "System prompt", "User prompt"

    mock = Mock(spec=PromptLoader)
    mock.get_prompt.return_value = MockPrompt()
    return mock


@pytest.fixture
def mock_agent_pool():
    mock = Mock(spec=AgentPool)
    mock.submit_task.return_value = None
    mock.wait_for_result.return_value = "test result"
    return mock


@pytest.fixture
def task_manager(tmp_path, mock_prompt_loader, mock_agent_pool):
    manager = TaskManager(storage_path=str(tmp_path / "task_storage"))
    manager.prompt_loader = mock_prompt_loader
    manager.agent_pool = mock_agent_pool
    return manager


class TestTaskManager:
    def test_create_task(self, task_manager):
        # Arrange
        task_type = "test_task"
        input_data = "test input"
        config = {"max_retries": 3}

        # Act
        task = task_manager.create_task(task_type, input_data, config)

        # Assert
        assert isinstance(task, Task)
        assert task.type == task_type
        assert task.input_data == input_data
        assert task.config == config
        assert isinstance(task.id, str)

    def test_create_task_invalid_type(self, task_manager):
        # Arrange
        task_manager.prompt_loader.get_prompt.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Unknown task type"):
            task_manager.create_task("invalid_type", "test input", {})

    def test_decompose_task_chunk_strategy(self, task_manager):
        # Arrange
        task = Task(
            id=str(uuid.uuid4()),
            type="test_task",
            input_data="test" * 60,  # Create string of length 240, which will create multiple chunks with chunk_size=100
            config={}
        )

        # Act
        subtasks = task_manager.decompose_task(task)

        # Assert
        assert isinstance(subtasks, list)
        assert all(isinstance(st, SubTask) for st in subtasks)
        assert all(st.parent_task_id == task.id for st in subtasks)
        assert len(subtasks) == 3  # Should be split into 3 chunks (240/100 rounded up)
        assert len(subtasks[0].input_data) <= 100  # Each chunk should be at most chunk_size
        assert len(subtasks[1].input_data) <= 100
        assert len(subtasks[2].input_data) <= 100

    def test_execute_task(self, task_manager):
        # Arrange
        task = Task(
            id=str(uuid.uuid4()),
            type="test_task",
            input_data="test input",
            config={}
        )

        # Act
        result = task_manager.execute(task)

        # Assert
        assert result == "test result"
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "test result"
        task_manager.agent_pool.submit_task.assert_called()
        task_manager.agent_pool.wait_for_result.assert_called()

    def test_aggregate_results_concatenate(self, task_manager):
        # Arrange
        task = Task(
            id=str(uuid.uuid4()),
            type="test_task",
            input_data="test input",
            config={}
        )
        task_manager.prompt_loader.get_prompt.return_value.metadata.decomposition = {
            "strategy": "chunk",
            "chunk_size": 100,
            "max_parallel": 2,
            "aggregation": "concatenate"
        }
        results = ["result1", "result2", "result3"]

        # Act
        aggregated = task_manager._aggregate_results(task, results)

        # Assert
        assert aggregated == "result1\nresult2\nresult3"

    def test_aggregate_results_merge(self, task_manager):
        # Arrange
        task = Task(
            id=str(uuid.uuid4()),
            type="test_task",
            input_data="test input",
            config={}
        )
        task_manager.prompt_loader.get_prompt.return_value.metadata.decomposition = {
            "strategy": "file",
            "max_parallel": 2,
            "aggregation": "merge"
        }
        results = [
            {"key1": ["value1"]},
            {"key1": ["value2"], "key2": ["value3"]},
        ]

        # Act
        aggregated = task_manager._aggregate_results(task, results)

        # Assert
        assert aggregated == {
            "key1": ["value1", "value2"],
            "key2": ["value3"]
        }

    def test_decompose_file_strategy(self, task_manager):
        # Arrange
        task = Task(
            id=str(uuid.uuid4()),
            type="test_task",
            input_data=[{"file1": "content1"}, {"file2": "content2"}],
            config={}
        )
        task_manager.prompt_loader.get_prompt.return_value.metadata.decomposition = {
            "strategy": "file",
            "max_parallel": 2,
            "aggregation": "merge"
        }

        # Act
        subtasks = task_manager.decompose_task(task)

        # Assert
        assert isinstance(subtasks, list)
        assert len(subtasks) == 2
        assert all(isinstance(st, SubTask) for st in subtasks)

    def test_decompose_error_strategy(self, task_manager):
        # Arrange
        task = Task(
            id=str(uuid.uuid4()),
            type="test_task",
            input_data=[{"error1": "details1"}, {"error2": "details2"}],
            config={}
        )
        task_manager.prompt_loader.get_prompt.return_value.metadata.decomposition = {
            "strategy": "error",
            "max_parallel": 2,
            "aggregation": "merge"
        }

        # Act
        subtasks = task_manager.decompose_task(task)

        # Assert
        assert isinstance(subtasks, list)
        assert len(subtasks) == 2
        assert all(isinstance(st, SubTask) for st in subtasks)

    @pytest.mark.asyncio
    async def test_dynamic_decomposition(self, task_manager):
        # Arrange
        task = Task(
            id=str(uuid.uuid4()),
            type="test_task",
            input_data="test input",
            config={}
        )
        task_manager.prompt_loader.get_prompt.return_value.metadata.decomposition = {
            "strategy": "custom",
            "max_parallel": 1,
            "aggregation": "custom"
        }

        # Mock dynamic strategy
        strategy_code = {
            "strategy": {"name": "test_dynamic"},
            "decomposition": {
                "method": """
def decompose_test_dynamic(task: Task) -> List[SubTask]:
    return [SubTask(
        id=str(uuid.uuid4()),
        type=task.type,
        input_data={"data": task.input_data},
        parent_task_id=task.id
    )]
""",
                "aggregation": """
def aggregate_test_dynamic_results(results: List[Any]) -> Any:
    return "\\n".join(str(r) for r in results)
"""
            }
        }
        task_manager._dynamic_strategies[task.type] = DynamicStrategy(strategy_code)

        # Act
        subtasks = task_manager.decompose_task(task)

        # Assert
        assert isinstance(subtasks, list)
        assert len(subtasks) == 1
        assert isinstance(subtasks[0], SubTask)
        assert subtasks[0].parent_task_id == task.id
