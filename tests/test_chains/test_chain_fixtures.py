"""Tests for chain fixtures instantiation."""

import pytest
from uuid import uuid4
from typing import Dict, Any

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
    )
from llmaestro.agents.models.models import AgentCapability
from llmaestro.llm.models import ModelDescriptor, ModelCapabilities
from llmaestro.core.models import TokenUsage, Task, TaskStatus
from llmaestro.llm.interfaces import LLMResponse
from llmaestro.core.task_manager import TaskManager
from llmaestro.prompts.loader import PromptLoader

# Import fixtures from fixtures.py
pytest_plugins = ["tests.test_chains.fixtures"]


@pytest.fixture
def chain_metadata():
    """Fixture for ChainMetadata."""
    return ChainMetadata(description="Test", tags={"test"}, version="1.0")


@pytest.fixture
def chain_state():
    """Fixture for ChainState."""
    return ChainState(status="pending")


@pytest.fixture
def chain_step(prompt):
    """Fixture for ChainStep."""
    task = Task(
        id=str(uuid4()),
        type="test_task",
        input_data={},
        config={},
        status=TaskStatus.PENDING
    )
    return ChainStep(
        task=task,
        retry_strategy=RetryStrategy(),
    )


@pytest.fixture
def chain_node(chain_step, chain_metadata):
    """Fixture for ChainNode."""
    return ChainNode(
        id=str(uuid4()),
        step=chain_step,
        node_type=NodeType.SEQUENTIAL,
        metadata=chain_metadata,
    )


@pytest.fixture
def chain_edge(chain_node):
    """Fixture for ChainEdge."""
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
def chain_context(chain_metadata, chain_state):
    """Fixture for ChainContext."""
    return ChainContext(
        artifacts={},
        metadata=chain_metadata,
        state=chain_state,
    )


@pytest.fixture
def prompt_loader():
    """Fixture for PromptLoader."""
    return PromptLoader()


@pytest.fixture
def task_manager():
    """Fixture for TaskManager."""
    return TaskManager()


@pytest.fixture
def chain_graph(chain_node, chain_edge, chain_context, llm, prompt_loader, task_manager):
    """Fixture for ChainGraph."""
    return ChainGraph(
        id=str(uuid4()),
        nodes={chain_node.id: chain_node},
        edges=[chain_edge],
        context=chain_context,
        task_manager=task_manager,
    )


@pytest.fixture
def agent_chain_graph(chain_graph, agent_pool, task_manager):
    """Fixture for AgentAwareChainGraph."""
    return AgentAwareChainGraph(
        id=chain_graph.id,
        nodes=chain_graph.nodes,
        edges=chain_graph.edges,
        context=chain_graph.context,
        agent_pool=agent_pool,
        task_manager=task_manager,
    )


@pytest.fixture
def input_transform():
    """Fixture for input transform function."""
    def transform(context: ChainContext, data: Dict[str, Any]) -> Dict[str, Any]:
        return data
    return transform


@pytest.fixture
def output_transform():
    """Fixture for output transform function."""
    def transform(response: LLMResponse) -> Dict[str, Any]:
        return {"content": response.content}
    return transform


@pytest.mark.parametrize("model_cls,params", [
    (RetryStrategy, {"max_retries": 3, "delay": 0.1}),
    (ChainMetadata, {"description": "Test", "tags": {"test"}, "version": "1.0"}),
    (ChainState, {"status": "pending"}),
])
def test_basic_models_instantiation(model_cls, params):
    """Test that basic chain models can be instantiated."""
    instance = model_cls(**params)
    assert instance is not None


def test_chain_context_instantiation(chain_context):
    """Test ChainContext instantiation."""
    assert chain_context is not None


@pytest.mark.asyncio
async def test_chain_step_instantiation(chain_step):
    """Test ChainStep instantiation."""
    assert chain_step is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("node_type", [
    NodeType.SEQUENTIAL,
    NodeType.PARALLEL,
    NodeType.CONDITIONAL,
    NodeType.AGENT,
])
async def test_chain_node_instantiation(chain_step, chain_metadata, node_type):
    """Test ChainNode instantiation with different node types."""
    node = ChainNode(
        id=str(uuid4()),
        step=chain_step,
        node_type=node_type,
        metadata=chain_metadata,
    )
    assert node is not None


def test_chain_edge_instantiation(chain_edge):
    """Test ChainEdge instantiation."""
    assert chain_edge is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [
    AgentType.GENERAL,
    AgentType.FAST,
    AgentType.SPECIALIST,
])
async def test_agent_chain_node_instantiation(chain_step, chain_metadata, agent_type):
    """Test AgentChainNode instantiation with different agent types."""
    node = AgentChainNode(
        id=str(uuid4()),
        step=chain_step,
        node_type=NodeType.AGENT,
        agent_type=agent_type,
        metadata=chain_metadata,
        required_capabilities={AgentCapability.TEXT},
    )
    assert node is not None


@pytest.mark.asyncio
async def test_chain_graph_instantiation(chain_graph):
    """Test ChainGraph instantiation."""
    assert chain_graph is not None


@pytest.mark.asyncio
async def test_agent_chain_graph_instantiation(agent_chain_graph):
    """Test AgentAwareChainGraph instantiation."""
    assert agent_chain_graph is not None


def test_transform_functions_instantiation(input_transform, output_transform, test_response):
    """Test that transform functions can be called."""
    context = ChainContext(artifacts={}, metadata=ChainMetadata(), state=ChainState())
    data: Dict[str, Any] = {"test": "data"}

    # Test input transform
    input_result = input_transform(context, data)
    assert input_result is not None

    # Test output transform
    output_result = output_transform(test_response)
    assert output_result is not None
