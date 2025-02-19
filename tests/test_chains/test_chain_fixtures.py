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
    TaskAwareChainGraph,
)
from llmaestro.agents.models.models import AgentCapability


@pytest.mark.parametrize("model_cls,params", [
    (RetryStrategy, {"max_retries": 3, "delay": 0.1}),
    (ChainMetadata, {"description": "Test", "tags": {"test"}, "version": "1.0"}),
    (ChainState, {"status": "pending"}),
])
def test_basic_models_instantiation(model_cls, params):
    """Test that basic chain models can be instantiated."""
    instance = model_cls(**params)
    assert instance is not None


def test_chain_context_instantiation(chain_metadata, chain_state):
    """Test ChainContext instantiation."""
    context = ChainContext(
        artifacts={},
        metadata=chain_metadata,
        state=chain_state,
    )
    assert context is not None


@pytest.mark.asyncio
async def test_chain_step_instantiation(prompt):
    """Test ChainStep instantiation."""
    step = ChainStep(
        task_type="test_task",
        prompt=prompt,
        retry_strategy=RetryStrategy(),
    )
    assert step is not None


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


def test_chain_edge_instantiation(chain_node):
    """Test ChainEdge instantiation."""
    target_node = ChainNode(
        id=str(uuid4()),
        step=chain_node.step,
        node_type=NodeType.SEQUENTIAL,
        metadata=chain_node.metadata,
    )
    edge = ChainEdge(
        source_id=chain_node.id,
        target_id=target_node.id,
        edge_type="next",
    )
    assert edge is not None


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
async def test_chain_graph_instantiation(llm, prompt_loader, chain_context, chain_node, chain_edge):
    """Test ChainGraph instantiation."""
    graph = ChainGraph(
        id=str(uuid4()),
        nodes={chain_node.id: chain_node},
        edges=[chain_edge],
        context=chain_context,
        llm=llm,
        prompt_loader=prompt_loader,
    )
    assert graph is not None


@pytest.mark.asyncio
async def test_agent_chain_graph_instantiation(chain_graph, agent_pool):
    """Test AgentAwareChainGraph instantiation."""
    graph = AgentAwareChainGraph(
        id=chain_graph.id,
        nodes=chain_graph.nodes,
        edges=chain_graph.edges,
        context=chain_graph.context,
        llm=chain_graph.llm,
        prompt_loader=chain_graph.prompt_loader,
        agent_pool=agent_pool,
    )
    assert graph is not None


@pytest.mark.asyncio
async def test_task_chain_graph_instantiation(agent_chain_graph, task_manager, prompt_loader):
    """Test TaskAwareChainGraph instantiation."""
    graph = TaskAwareChainGraph(
        id=agent_chain_graph.id,
        nodes=agent_chain_graph.nodes,
        edges=agent_chain_graph.edges,
        context=agent_chain_graph.context,
        llm=agent_chain_graph.llm,
        prompt_loader=prompt_loader,
        agent_pool=agent_chain_graph._agent_pool,
        task_manager=task_manager,
    )
    assert graph is not None


def test_transform_functions_instantiation(input_transform, output_transform):
    """Test that transform functions can be called."""
    context = ChainContext(artifacts={}, metadata=ChainMetadata(), state=ChainState())
    data: Dict[str, Any] = {"test": "data"}

    # Test input transform
    input_result = input_transform(context, data)
    assert input_result is not None

    # Test output transform
    from llmaestro.llm.interfaces import LLMResponse
    response = LLMResponse(content="test", success=True, provider="test")
    output_result = output_transform(response)
    assert output_result is not None
