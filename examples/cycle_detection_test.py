"""Test for cycle detection algorithm."""

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


def test_acyclic_graph():
    """Test cycle detection on an acyclic graph."""
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

    # Check if the graph is acyclic
    is_acyclic = graph.is_acyclic()
    print(f"Is the graph acyclic? {is_acyclic}")

    # Try to find a cycle
    cycle = graph.find_cycle()
    if cycle:
        print(f"Cycle found: {' -> '.join(cycle)}")
    else:
        print("No cycle found")

    return is_acyclic


def test_cyclic_graph():
    """Test cycle detection on a graph with a cycle."""
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

    # Check if the graph is acyclic
    is_acyclic = graph.is_acyclic()
    print(f"Is the graph acyclic? {is_acyclic}")

    # Try to find a cycle
    cycle = graph.find_cycle()
    if cycle:
        print(f"Cycle found: {' -> '.join(cycle)}")
    else:
        print("No cycle found")

    # Test validation
    try:
        graph.verify_acyclic_graph()
        print("Validation passed (unexpected)")
    except ValueError as e:
        print(f"Validation error (expected): {e}")

    return not is_acyclic


def test_initialization_validation():
    """Test that validation happens during initialization."""
    # Create a cyclic graph first without validation
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

    # Now try to create a new graph with validation enabled
    try:
        new_graph = SimpleGraph()

        for node_id, node in graph.nodes.items():
            new_graph.add_node(node)

        for edge in graph.edges:
            new_graph.add_edge(edge)

        print("Initialization validation failed (unexpected)")
        return False
    except ValueError as e:
        print(f"Initialization validation error (expected): {e}")
        return True


def main():
    """Run all tests."""
    print("Testing acyclic graph...")
    acyclic_result = test_acyclic_graph()

    print("\nTesting cyclic graph...")
    cyclic_result = test_cyclic_graph()

    print("\nTesting initialization validation...")
    init_result = test_initialization_validation()

    # Print summary
    print("\nTest Results:")
    print(f"Acyclic graph test: {'PASSED' if acyclic_result else 'FAILED'}")
    print(f"Cyclic graph test: {'PASSED' if cyclic_result else 'FAILED'}")
    print(f"Initialization validation test: {'PASSED' if init_result else 'FAILED'}")

    # Return overall result
    return acyclic_result and cyclic_result and init_result


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
