"""Test for cycle detection in ChainGraph."""

import asyncio
import sys
import os
from typing import Any, Dict, List, Optional

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llmaestro.chains.chains import (
    ChainGraph,
    ChainNode,
    ChainEdge,
    ChainStep,
    NodeType,
    BasePrompt
)
from src.llmaestro.prompts.types import PromptMetadata


# Create a mock prompt for testing
class MockPrompt(BasePrompt):
    """Mock implementation of BasePrompt for testing."""

    async def load(self, path: str) -> None:
        """Mock implementation of load."""
        pass

    async def save(self, path: str) -> None:
        """Mock implementation of save."""
        pass


def create_mock_prompt(name: str) -> MockPrompt:
    """Create a mock prompt for testing."""
    return MockPrompt(
        name=name,
        description=f"Mock prompt for {name}",
        system_prompt="",
        user_prompt="",
        metadata=PromptMetadata(type="mock"),
        variables=[],
    )


def create_acyclic_chain_graph() -> ChainGraph:
    """Create a simple acyclic chain graph."""
    graph = ChainGraph()

    # Create some nodes
    node1 = ChainNode(
        id="node1",
        step=ChainStep(prompt=create_mock_prompt("node1")),
        node_type=NodeType.AGENT
    )
    node2 = ChainNode(
        id="node2",
        step=ChainStep(prompt=create_mock_prompt("node2")),
        node_type=NodeType.AGENT
    )
    node3 = ChainNode(
        id="node3",
        step=ChainStep(prompt=create_mock_prompt("node3")),
        node_type=NodeType.CONDITIONAL
    )
    node4 = ChainNode(
        id="node4",
        step=ChainStep(prompt=create_mock_prompt("node4")),
        node_type=NodeType.AGENT
    )
    node5 = ChainNode(
        id="node5",
        step=ChainStep(prompt=create_mock_prompt("node5")),
        node_type=NodeType.AGENT
    )

    # Add nodes to graph
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_node(node4)
    graph.add_node(node5)

    # Create edges (acyclic)
    edge1 = ChainEdge(source_id="node1", target_id="node2", edge_type="next")
    edge2 = ChainEdge(source_id="node2", target_id="node3", edge_type="next")
    edge3 = ChainEdge(source_id="node3", target_id="node4", edge_type="path1")
    edge4 = ChainEdge(source_id="node3", target_id="node5", edge_type="path2")

    # Add edges to graph
    graph.add_edge(edge1)
    graph.add_edge(edge2)
    graph.add_edge(edge3)
    graph.add_edge(edge4)

    return graph


def create_cyclic_chain_graph() -> ChainGraph:
    """Create a chain graph with a cycle."""
    # Create a graph with verify_acyclic=False to bypass validation
    graph = ChainGraph(verify_acyclic=False)

    # Create some nodes
    node1 = ChainNode(
        id="node1",
        step=ChainStep(prompt=create_mock_prompt("node1")),
        node_type=NodeType.AGENT
    )
    node2 = ChainNode(
        id="node2",
        step=ChainStep(prompt=create_mock_prompt("node2")),
        node_type=NodeType.AGENT
    )
    node3 = ChainNode(
        id="node3",
        step=ChainStep(prompt=create_mock_prompt("node3")),
        node_type=NodeType.CONDITIONAL
    )
    node4 = ChainNode(
        id="node4",
        step=ChainStep(prompt=create_mock_prompt("node4")),
        node_type=NodeType.AGENT
    )

    # Add nodes to graph
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_node(node4)

    # Create edges (with a cycle)
    edge1 = ChainEdge(source_id="node1", target_id="node2", edge_type="next")
    edge2 = ChainEdge(source_id="node2", target_id="node3", edge_type="next")
    edge3 = ChainEdge(source_id="node3", target_id="node4", edge_type="path1")
    edge4 = ChainEdge(source_id="node4", target_id="node1", edge_type="next")  # Creates a cycle

    # Add edges to graph
    graph.add_edge(edge1)
    graph.add_edge(edge2)
    graph.add_edge(edge3)
    graph.add_edge(edge4)

    return graph


async def main():
    """Run the test."""
    print("Testing acyclic chain graph...")
    acyclic_graph = create_acyclic_chain_graph()

    # Check if the graph is acyclic
    is_acyclic = acyclic_graph.is_acyclic()
    print(f"Is the graph acyclic? {is_acyclic}")

    print("\nTesting cyclic chain graph...")
    cyclic_graph = create_cyclic_chain_graph()

    # Check if the graph is acyclic
    is_acyclic = cyclic_graph.is_acyclic()
    print(f"Is the graph acyclic? {is_acyclic}")

    # Find the cycle
    cycle = cyclic_graph.find_cycle()
    if cycle:
        print(f"Cycle found: {' -> '.join(cycle)}")

    print("\nTesting cycle validation...")
    try:
        # This should raise a ValueError
        cyclic_graph.verify_acyclic_graph()
    except ValueError as e:
        print(f"Validation error: {e}")

    print("\nTesting automatic validation during initialization...")
    try:
        # Create a new graph with the same nodes and edges, but with validation enabled
        graph = ChainGraph()

        for node_id, node in cyclic_graph.nodes.items():
            graph.add_node(node)

        for edge in cyclic_graph.edges:
            graph.add_edge(edge)
    except ValueError as e:
        print(f"Initialization validation error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
