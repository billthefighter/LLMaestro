"""Example demonstrating cycle detection in ChainGraph."""

import asyncio
import sys
import os
from typing import Any, Dict, List, Optional

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llmaestro.core.graph import BaseGraph, BaseNode, BaseEdge
from pydantic import Field


# Create simple node and edge classes for testing
class SimpleNode(BaseNode):
    """Simple node implementation for testing."""
    name: str


class SimpleEdge(BaseEdge):
    """Simple edge implementation for testing."""
    pass


class SimpleGraph(BaseGraph[SimpleNode, SimpleEdge]):
    """Simple graph implementation with cycle detection."""

    # Add verify_acyclic as a field
    verify_acyclic: bool = Field(default=True, description="Whether to verify the graph is acyclic during initialization")

    def __init__(self, **data: Any):
        """Initialize the graph and verify it is acyclic by default."""
        super().__init__(**data)
        # Get the verify_acyclic value from self after initialization
        should_verify = getattr(self, "verify_acyclic", True)
        if should_verify and self.nodes and self.edges:
            self.verify_acyclic_graph()

    def verify_acyclic_graph(self) -> None:
        """Verify that the graph is acyclic (contains no cycles).

        Raises:
            ValueError: If a cycle is detected in the graph.
        """
        cycle = self.find_cycle()
        if cycle:
            cycle_str = " -> ".join(cycle)
            raise ValueError(f"Cycle detected in graph: {cycle_str}")

    def is_acyclic(self) -> bool:
        """Check if the graph is acyclic.

        Returns:
            bool: True if the graph is acyclic, False otherwise.
        """
        return self.find_cycle() is None

    def find_cycle(self) -> Optional[List[str]]:
        """Find a cycle in the graph if one exists.

        Returns:
            Optional[List[str]]: A list of node IDs forming a cycle, or None if no cycle exists.
        """
        # Use depth-first search to detect cycles
        visited = set()  # Nodes that have been fully processed
        path = []  # Current path being explored
        path_set = set()  # Set version of path for O(1) lookups
        cycle_found: List[Optional[List[str]]] = [None]  # Use a list to store the cycle

        def dfs(node_id: str) -> bool:
            """Depth-first search to detect cycles.

            Returns:
                bool: True if a cycle was found, False otherwise.
            """
            if node_id in path_set:
                # We've found a cycle
                cycle_start_idx = path.index(node_id)
                cycle_found[0] = path[cycle_start_idx:] + [node_id]
                return True

            if node_id in visited:
                return False

            visited.add(node_id)
            path.append(node_id)
            path_set.add(node_id)

            # Visit all neighbors
            for edge in self.edges:
                if edge.source_id == node_id:
                    if dfs(edge.target_id):
                        return True

            # Remove from current path
            path.pop()
            path_set.remove(node_id)
            return False

        # Start DFS from each node that hasn't been visited
        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return cycle_found[0]

        return None


def create_acyclic_graph() -> SimpleGraph:
    """Create a simple acyclic graph."""
    graph = SimpleGraph()

    # Create some nodes
    node1 = SimpleNode(id="node1", name="Node 1")
    node2 = SimpleNode(id="node2", name="Node 2")
    node3 = SimpleNode(id="node3", name="Node 3")

    # Add nodes to graph
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    # Create edges (acyclic)
    edge1 = SimpleEdge(source_id="node1", target_id="node2", edge_type="next")
    edge2 = SimpleEdge(source_id="node2", target_id="node3", edge_type="next")

    # Add edges to graph
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    return graph


def create_cyclic_graph() -> SimpleGraph:
    """Create a graph with a cycle."""
    # Create a graph with verify_acyclic=False to bypass validation
    graph = SimpleGraph(verify_acyclic=False)

    # Create some nodes
    node1 = SimpleNode(id="node1", name="Node 1")
    node2 = SimpleNode(id="node2", name="Node 2")
    node3 = SimpleNode(id="node3", name="Node 3")

    # Add nodes to graph
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    # Create edges (with a cycle)
    edge1 = SimpleEdge(source_id="node1", target_id="node2", edge_type="next")
    edge2 = SimpleEdge(source_id="node2", target_id="node3", edge_type="next")
    edge3 = SimpleEdge(source_id="node3", target_id="node1", edge_type="next")  # Creates a cycle

    # Add edges to graph
    graph.add_edge(edge1)
    graph.add_edge(edge2)
    graph.add_edge(edge3)

    return graph


async def main():
    """Run the example."""
    print("Testing acyclic graph...")
    acyclic_graph = create_acyclic_graph()

    # Check if the graph is acyclic
    is_acyclic = acyclic_graph.is_acyclic()
    print(f"Is the graph acyclic? {is_acyclic}")

    print("\nTesting cyclic graph...")
    cyclic_graph = create_cyclic_graph()

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
        graph = SimpleGraph()

        for node_id, node in cyclic_graph.nodes.items():
            graph.add_node(node)

        for edge in cyclic_graph.edges:
            graph.add_edge(edge)
    except ValueError as e:
        print(f"Initialization validation error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
