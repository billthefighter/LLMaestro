"""Tests for chain execution methods."""

import pytest
from uuid import uuid4
from typing import Dict, Any

from llmaestro.chains.chains import (
    NodeType,
    RetryStrategy,
    ChainMetadata,
    ChainState,
    ChainContext,
    ChainStep,
    ChainNode,
    ChainEdge,
    ChainGraph,
)
from llmaestro.agents.models.models import AgentCapability
from llmaestro.llm.models import ModelDescriptor, ModelCapabilities
from llmaestro.core.models import TokenUsage
from llmaestro.llm.interfaces import LLMResponse

# Import fixtures
pytest_plugins = ["tests.test_chains.fixtures"]


@pytest.mark.asyncio
async def test_chain_step_execution(chain_step, agent_pool, chain_context):
    """Test execution of a single chain step."""
    result = await chain_step.execute(agent_pool, chain_context)
    assert isinstance(result, LLMResponse)
    assert result.content == "Test response"


@pytest.mark.asyncio
async def test_chain_execution_with_dependencies(chain_graph, chain_step):
    """Test execution of a chain with dependencies."""
    # Create a new chain graph for this test
    graph = ChainGraph(
        id=str(uuid4()),
        nodes={},
        edges=[],
        context=chain_graph.context,
        agent_pool=chain_graph.agent_pool
    )

    # Add first node
    first_node = ChainNode(
        id=str(uuid4()),
        step=chain_step,
        node_type=NodeType.SEQUENTIAL,
        metadata=ChainMetadata(description="First node")
    )
    first_node_id = graph.add_node(first_node)

    # Add second node
    second_step = await ChainStep.create(
        prompt=chain_step.prompt,
        retry_strategy=RetryStrategy()
    )
    second_node = ChainNode(
        id=str(uuid4()),
        step=second_step,
        node_type=NodeType.SEQUENTIAL,
        metadata=ChainMetadata(description="Second node")
    )
    second_node_id = graph.add_node(second_node)

    # Add dependency edge
    graph.add_edge(ChainEdge(
        source_id=first_node_id,
        target_id=second_node_id,
        edge_type="depends_on"
    ))

    # Execute chain
    results = await graph.execute()
    assert len(results) == 2
    assert isinstance(results[first_node_id], LLMResponse)
    assert isinstance(results[second_node_id], LLMResponse)


@pytest.mark.asyncio
async def test_parallel_node_execution(chain_graph, chain_step):
    """Test execution of parallel nodes."""
    # Create a new chain graph for this test
    graph = ChainGraph(
        id=str(uuid4()),
        nodes={},
        edges=[],
        context=chain_graph.context,
        agent_pool=chain_graph.agent_pool
    )

    # Add nodes
    node_ids = []
    for i in range(3):  # Create 3 parallel nodes
        node = ChainNode(
            id=str(uuid4()),
            step=chain_step,
            node_type=NodeType.PARALLEL,
            metadata=ChainMetadata(description=f"Parallel node {i}")
        )
        node_id = graph.add_node(node)
        node_ids.append(node_id)

    # Execute chain
    results = await graph.execute()
    assert len(results) == 3
    for node_id in node_ids:
        assert isinstance(results[node_id], LLMResponse)


@pytest.mark.asyncio
async def test_chain_execution_order(chain_graph, chain_step):
    """Test that chain executes nodes in correct order."""
    # Create a new chain graph for this test
    graph = ChainGraph(
        id=str(uuid4()),
        nodes={},
        edges=[],
        context=chain_graph.context,
        agent_pool=chain_graph.agent_pool
    )

    # Create a linear chain: A -> B -> C
    prev_node_id = None
    node_ids = []
    for i in range(3):
        node = ChainNode(
            id=str(uuid4()),
            step=chain_step,
            node_type=NodeType.SEQUENTIAL,
            metadata=ChainMetadata(description=f"Node {i}")
        )
        node_id = graph.add_node(node)
        node_ids.append(node_id)

        if prev_node_id:
            graph.add_edge(ChainEdge(
                source_id=prev_node_id,
                target_id=node_id,
                edge_type="next"
            ))
        prev_node_id = node_id

    execution_order = graph.get_execution_order()
    assert len(execution_order) == 3  # Should have 3 levels
    assert node_ids[0] in execution_order[0]  # First node should be in first level
    assert node_ids[1] in execution_order[1]  # Second node should be in second level
    assert node_ids[2] in execution_order[2]  # Third node should be in third level


@pytest.mark.asyncio
async def test_chain_error_handling(chain_graph, chain_step, monkeypatch):
    """Test chain error handling during execution."""
    # Create a new chain graph for this test
    graph = ChainGraph(
        id=str(uuid4()),
        nodes={},
        edges=[],
        context=chain_graph.context,
        agent_pool=chain_graph.agent_pool
    )

    # Add a node
    node = ChainNode(
        id=str(uuid4()),
        step=chain_step,
        node_type=NodeType.SEQUENTIAL,
        metadata=ChainMetadata(description="Test node")
    )
    node_id = graph.add_node(node)

    # Mock the execute method at the agent_pool level instead of the step level
    async def mock_execute(*args, **kwargs):
        raise RuntimeError("Test error")

    monkeypatch.setattr(graph.agent_pool, "execute_prompt", mock_execute)

    with pytest.raises(RuntimeError, match="Test error"):
        await graph.execute()


def test_chain_node_dependencies(chain_graph):
    """Test getting node dependencies."""
    first_node_id = list(chain_graph.nodes.keys())[0]
    second_step = ChainStep(
        prompt=chain_graph.nodes[first_node_id].step.prompt,
        retry_strategy=RetryStrategy()
    )
    second_node = ChainNode(
        id=str(uuid4()),
        step=second_step,
        node_type=NodeType.SEQUENTIAL,
        metadata=ChainMetadata()
    )
    second_node_id = chain_graph.add_node(second_node)

    chain_graph.add_edge(ChainEdge(
        source_id=first_node_id,
        target_id=second_node_id,
        edge_type="depends_on"
    ))

    deps = chain_graph.get_node_dependencies(second_node_id)
    assert len(deps) == 1
    assert first_node_id in deps
