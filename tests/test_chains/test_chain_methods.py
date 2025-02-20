"""Tests for chain methods and functionality."""

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
from llmaestro.core.models import Task, SubTask

# Import fixtures from fixtures.py
pytest_plugins = ["tests.test_chains.fixtures"]


@pytest.mark.asyncio
async def test_chain_step_execution(chain_step, task_manager, chain_context, input_transform, output_transform, test_response):
    """Test ChainStep execution with transforms."""
    # Set transforms
    chain_step.input_transform = input_transform
    chain_step.output_transform = output_transform

    # Execute step
    test_data = {"test": "data"}
    result = await chain_step.execute(task_manager, chain_context, **test_data)

    # Verify input transform was applied
    assert chain_step.task.input_data == {"transformed": test_data}

    # Verify output transform was applied
    assert result == {"processed": test_response.content}


@pytest.mark.asyncio
async def test_chain_node_creation(task, input_transform, output_transform):
    """Test ChainNode creation with different configurations."""
    node = await ChainNode.create(
        task=task,
        node_type=NodeType.SEQUENTIAL,
        input_transform=input_transform,
        output_transform=output_transform,
    )
    assert node.node_type == NodeType.SEQUENTIAL
    assert node.step.input_transform == input_transform
    assert node.step.output_transform == output_transform


@pytest.mark.asyncio
async def test_agent_chain_node_creation(task):
    """Test AgentChainNode creation with different agent types."""
    required_capabilities = {AgentCapability.TEXT}
    node = await AgentChainNode.create(
        task=task,
        node_type=NodeType.AGENT,
        agent_type=AgentType.GENERAL,
        required_capabilities=required_capabilities,
    )
    assert node.node_type == NodeType.AGENT
    assert node.agent_type == AgentType.GENERAL
    assert node.required_capabilities == required_capabilities


def test_chain_graph_node_management(chain_graph, chain_node, chain_edge):
    """Test adding and managing nodes in ChainGraph."""
    # Test adding node
    node_id = chain_graph.add_node(chain_node)
    assert node_id in chain_graph.nodes

    # Test adding edge with existing nodes
    chain_graph.add_edge(chain_edge)
    assert chain_edge in chain_graph.edges

    # Test adding edge with non-existent nodes
    invalid_edge = ChainEdge(
        source_id=str(uuid4()),
        target_id=str(uuid4()),
        edge_type="next",
    )
    with pytest.raises(ValueError):
        chain_graph.add_edge(invalid_edge)


def test_chain_graph_dependency_management(chain_graph, chain_node):
    """Test dependency management in ChainGraph."""
    # Add nodes and create dependencies
    node1_id = chain_graph.add_node(chain_node)
    node2 = ChainNode(
        id=str(uuid4()),
        step=chain_node.step,
        node_type=NodeType.SEQUENTIAL,
        metadata=chain_node.metadata,
    )
    node2_id = chain_graph.add_node(node2)

    edge = ChainEdge(
        source_id=node1_id,
        target_id=node2_id,
        edge_type="next",
    )
    chain_graph.add_edge(edge)

    # Test dependency retrieval
    deps = chain_graph.get_node_dependencies(node2_id)
    assert node1_id in deps


def test_chain_graph_execution_order(chain_graph, chain_node):
    """Test execution order calculation in ChainGraph."""
    # Create a simple chain: A -> B -> C
    node_a = chain_node
    node_b = ChainNode(
        id=str(uuid4()),
        step=chain_node.step,
        node_type=NodeType.SEQUENTIAL,
        metadata=chain_node.metadata,
    )
    node_c = ChainNode(
        id=str(uuid4()),
        step=chain_node.step,
        node_type=NodeType.SEQUENTIAL,
        metadata=chain_node.metadata,
    )

    # Add nodes
    id_a = chain_graph.add_node(node_a)
    id_b = chain_graph.add_node(node_b)
    id_c = chain_graph.add_node(node_c)

    # Add edges
    chain_graph.add_edge(ChainEdge(source_id=id_a, target_id=id_b, edge_type="next"))
    chain_graph.add_edge(ChainEdge(source_id=id_b, target_id=id_c, edge_type="next"))

    # Get execution order
    order = chain_graph.get_execution_order()
    assert len(order) == 3  # Should have 3 levels
    assert id_a in order[0]  # A should be first
    assert id_b in order[1]  # B should be second
    assert id_c in order[2]  # C should be last


@pytest.mark.asyncio
async def test_chain_graph_execution(chain_graph, chain_node, task_manager):
    """Test execution of ChainGraph."""
    # Ensure task manager is set
    chain_graph.task_manager = task_manager

    # Add a simple node
    node_id = chain_graph.add_node(chain_node)

    # Execute graph
    results = await chain_graph.execute()
    assert results is not None
    assert node_id in results


@pytest.mark.asyncio
async def test_agent_chain_graph_execution(agent_chain_graph, agent_chain_node, task_manager):
    """Test execution of AgentAwareChainGraph."""
    # Ensure task manager is set
    agent_chain_graph.task_manager = task_manager

    # Add agent node
    node_id = agent_chain_graph.add_node(agent_chain_node)

    # Execute graph
    results = await agent_chain_graph.execute()
    assert results is not None
    assert node_id in results


@pytest.mark.parametrize("node_type,expected_error", [
    (NodeType.SEQUENTIAL, None),
    (NodeType.PARALLEL, None),
    (NodeType.CONDITIONAL, None),
    (NodeType.AGENT, None),
])
@pytest.mark.asyncio
async def test_chain_node_types(task, node_type, expected_error):
    """Test creation of chain nodes with different node types."""
    if expected_error:
        with pytest.raises(expected_error):
            await ChainNode.create(task=task, node_type=node_type)
    else:
        node = await ChainNode.create(task=task, node_type=node_type)
        assert node.node_type == node_type


@pytest.mark.parametrize("agent_type,capabilities", [
    (AgentType.GENERAL, {AgentCapability.TEXT}),
    (AgentType.FAST, {AgentCapability.TEXT}),
    (AgentType.SPECIALIST, {AgentCapability.TEXT, AgentCapability.CODE}),
])
@pytest.mark.asyncio
async def test_agent_chain_node_types(task, agent_type, capabilities):
    """Test creation of agent chain nodes with different agent types and capabilities."""
    node = await AgentChainNode.create(
        task=task,
        node_type=NodeType.AGENT,
        agent_type=agent_type,
        required_capabilities=capabilities,
    )
    assert node.agent_type == agent_type
    assert node.required_capabilities == capabilities


def test_cyclic_dependency_detection(chain_graph, chain_node):
    """Test detection of cyclic dependencies in chain graph."""
    # Create a cycle: A -> B -> A
    node_a = chain_node
    node_b = ChainNode(
        id=str(uuid4()),
        step=chain_node.step,
        node_type=NodeType.SEQUENTIAL,
        metadata=chain_node.metadata,
    )

    # Add nodes
    id_a = chain_graph.add_node(node_a)
    id_b = chain_graph.add_node(node_b)

    # Create cycle
    chain_graph.add_edge(ChainEdge(source_id=id_a, target_id=id_b, edge_type="next"))
    chain_graph.add_edge(ChainEdge(source_id=id_b, target_id=id_a, edge_type="next"))

    # Attempting to get execution order should raise error
    with pytest.raises(ValueError, match="Cycle detected in chain graph"):
        chain_graph.get_execution_order()
