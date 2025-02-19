"""Fixtures for chain testing."""

import pytest
from datetime import datetime
from typing import Dict, Any, Set, Callable, List, Optional, Union
from uuid import uuid4
from enum import Enum

from llmaestro.agents.agent_pool import AgentPool
from llmaestro.agents.models.models import AgentCapability
from llmaestro.chains.chains import (
    NodeType,
    AgentType,
    RetryStrategy,
    ChainMetadata,
    ChainState,
    ChainContext,
    ChainStep,
    ChainNode,
    ChainEdge,
    AgentChainNode,
    ChainGraph,
    AgentAwareChainGraph,
    TaskAwareChainGraph,
)
from llmaestro.core.models import Task, SubTask, Artifact, BaseResponse, AgentConfig
from llmaestro.core.task_manager import TaskManager, DecompositionConfig
from llmaestro.llm.interfaces import BaseLLMInterface, LLMResponse
from llmaestro.llm.models import ModelFamily
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.loader import PromptLoader
from llmaestro.prompts.types import (
    PromptMetadata,
    VersionInfo,
    ResponseFormat,
)


class ChangeType(str, Enum):
    """Change type for version info."""
    NEW = "new"
    UPDATE = "update"
    FIX = "fix"
    REFACTOR = "refactor"


@pytest.fixture
def test_response() -> LLMResponse:
    """Test LLM response."""
    return LLMResponse(
        content="Test response",
        success=True,
        provider="test",
        provider_metadata={"test": True},
    )


@pytest.fixture
def llm(monkeypatch, test_response) -> BaseLLMInterface:
    """LLM interface with monkeypatched process method."""
    async def mock_process(*args, **kwargs) -> LLMResponse:
        return test_response

    class TestLLM(BaseLLMInterface):
        @property
        def model_family(self) -> ModelFamily:
            return ModelFamily.CLAUDE

        async def process(self, prompt: Union[BasePrompt, str], variables: Optional[Dict[str, Any]] = None) -> LLMResponse:
            return await mock_process(prompt, variables)

        async def batch_process(
            self,
            prompts: List[Union[BasePrompt, str]],
            variables: Optional[List[Optional[Dict[str, Any]]]] = None
        ) -> List[LLMResponse]:
            return [await mock_process(p, v) for p, v in zip(prompts, variables or [None] * len(prompts))]

        async def process_async(self, prompt: Union[BasePrompt, str], variables: Optional[Dict[str, Any]] = None) -> LLMResponse:
            return await self.process(prompt, variables)

    return TestLLM(config=AgentConfig(provider="test", model_name="test", api_key="test"))


@pytest.fixture
def prompt() -> BasePrompt:
    """Test prompt."""
    class TestPrompt(BasePrompt):
        def render(self, **kwargs) -> tuple[str, str, list]:
            return "system prompt", "user prompt", []

        async def save(self) -> bool:
            return True

        @classmethod
        async def load(cls, identifier: str) -> Optional[BasePrompt]:
            return cls(
                name="test_prompt",
                description="Test prompt",
                system_prompt="Test system prompt",
                user_prompt="Test user prompt",
                metadata=PromptMetadata(
                    type="test",
                    expected_response=ResponseFormat(
                        format="json",
                        schema='{"type": "object"}'
                    ),
                ),
                current_version=VersionInfo(
                    number="1.0.0",
                    author="test",
                    timestamp=datetime.now(),
                    description="Initial version",
                    change_type=ChangeType.NEW,
                ),
            )

    return TestPrompt(
        name="test_prompt",
        description="Test prompt",
        system_prompt="Test system prompt",
        user_prompt="Test user prompt",
        metadata=PromptMetadata(
            type="test",
            expected_response=ResponseFormat(
                format="json",
                schema='{"type": "object"}'
            ),
        ),
        current_version=VersionInfo(
            number="1.0.0",
            author="test",
            timestamp=datetime.now(),
            description="Initial version",
            change_type=ChangeType.NEW,
        ),
    )


@pytest.fixture
def prompt_loader(monkeypatch, prompt) -> PromptLoader:
    """PromptLoader with monkeypatched load_prompt method."""
    async def mock_load_prompt(*args, **kwargs) -> BasePrompt:
        return prompt

    loader = PromptLoader()
    monkeypatch.setattr(loader, "load_prompt", mock_load_prompt)
    return loader


@pytest.fixture
def retry_strategy() -> RetryStrategy:
    """Basic retry strategy."""
    return RetryStrategy(max_retries=3, delay=0.1)


@pytest.fixture
def chain_metadata() -> ChainMetadata:
    """Basic chain metadata."""
    return ChainMetadata(
        description="Test chain",
        tags={"test"},
        version="1.0.0",
    )


@pytest.fixture
def chain_state() -> ChainState:
    """Basic chain state."""
    return ChainState(
        status="pending",
        current_step=None,
        completed_steps=set(),
        failed_steps=set(),
        step_results={},
        variables={},
    )


@pytest.fixture
def chain_context(chain_metadata, chain_state) -> ChainContext:
    """Chain context with basic configuration."""
    return ChainContext(
        artifacts={},
        metadata=chain_metadata,
        state=chain_state,
    )


@pytest.fixture
async def chain_step(prompt) -> ChainStep:
    """Basic chain step."""
    return ChainStep(
        task_type="test_task",
        prompt=prompt,
        retry_strategy=RetryStrategy(),
    )


@pytest.fixture
async def chain_node(chain_step, chain_metadata) -> ChainNode:
    """Basic chain node."""
    return ChainNode(
        id=str(uuid4()),
        step=chain_step,
        node_type=NodeType.SEQUENTIAL,
        metadata=chain_metadata,
    )


@pytest.fixture
def chain_edge(chain_node) -> ChainEdge:
    """Basic chain edge."""
    target_node = ChainNode(
        id=str(uuid4()),
        step=chain_node.step,
        node_type=NodeType.SEQUENTIAL,
        metadata=chain_node.metadata,
    )
    return ChainEdge(
        source_id=chain_node.id,
        target_id=target_node.id,
        edge_type="next",
    )


@pytest.fixture
async def agent_chain_node(chain_step, chain_metadata) -> AgentChainNode:
    """Agent chain node."""
    return AgentChainNode(
        id=str(uuid4()),
        step=chain_step,
        node_type=NodeType.AGENT,
        agent_type=AgentType.GENERAL,
        metadata=chain_metadata,
        required_capabilities={AgentCapability.TEXT},
    )


@pytest.fixture
def agent_pool(monkeypatch) -> AgentPool:
    """Agent pool with monkeypatched methods."""
    async def mock_process_task(*args, **kwargs) -> BaseResponse:
        return BaseResponse(success=True)

    class TestAgent:
        async def process_task(self, *args, **kwargs):
            return await mock_process_task(*args, **kwargs)

    class TestAgentPool(AgentPool):
        def get_agent(self, *args, **kwargs):
            return TestAgent()

    return TestAgentPool()


@pytest.fixture
def task_manager(monkeypatch) -> TaskManager:
    """Task manager with monkeypatched methods."""
    async def mock_decompose_task(task: Task, config: DecompositionConfig) -> list[SubTask]:
        return [
            SubTask(
                id=str(uuid4()),
                type="test_task",
                input_data={"test": "data"},
                parent_task_id=task.id,
            )
        ]

    manager = TaskManager()
    monkeypatch.setattr(manager, "decompose_task", mock_decompose_task)
    return manager


@pytest.fixture
def task() -> Task:
    """Test task."""
    return Task(
        id=str(uuid4()),
        type="test_task",
        input_data={"test": "data"},
        config={},
    )


@pytest.fixture
async def chain_graph(llm, prompt_loader, chain_context, chain_node, chain_edge) -> ChainGraph:
    """Basic chain graph."""
    return ChainGraph(
        id=str(uuid4()),
        nodes={chain_node.id: chain_node},
        edges=[chain_edge],
        context=chain_context,
        llm=llm,
        prompt_loader=prompt_loader,
    )


@pytest.fixture
async def agent_chain_graph(chain_graph, agent_pool) -> AgentAwareChainGraph:
    """Agent-aware chain graph."""
    return AgentAwareChainGraph(
        id=chain_graph.id,
        nodes=chain_graph.nodes,
        edges=chain_graph.edges,
        context=chain_graph.context,
        llm=chain_graph.llm,
        prompt_loader=chain_graph.prompt_loader,
        agent_pool=agent_pool,
    )


@pytest.fixture
async def task_chain_graph(agent_chain_graph, task_manager, prompt_loader) -> TaskAwareChainGraph:
    """Task-aware chain graph."""
    return TaskAwareChainGraph(
        id=agent_chain_graph.id,
        nodes=agent_chain_graph.nodes,
        edges=agent_chain_graph.edges,
        context=agent_chain_graph.context,
        llm=agent_chain_graph.llm,
        prompt_loader=prompt_loader,
        agent_pool=agent_chain_graph._agent_pool,
        task_manager=task_manager,
    )


@pytest.fixture
def input_transform() -> Callable[[ChainContext, Dict[str, Any]], Dict[str, Any]]:
    """Sample input transform function."""
    def transform(context: ChainContext, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"transformed": data}
    return transform


@pytest.fixture
def output_transform() -> Callable[[LLMResponse], Dict[str, Any]]:
    """Sample output transform function."""
    def transform(response: LLMResponse) -> Dict[str, Any]:
        return {"processed": response.content}
    return transform
