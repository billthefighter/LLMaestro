"""Base graph implementation for LLM orchestration."""

from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class BaseNode(BaseModel):
    """Base class for all graph nodes."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)


class BaseEdge(BaseModel):
    """Base class for all graph edges."""

    source_id: str = Field(..., description="ID of the source node")
    target_id: str = Field(..., description="ID of the target node")
    edge_type: str = Field(..., description="Type of relationship")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)


NodeType = TypeVar("NodeType", bound=BaseNode)
EdgeType = TypeVar("EdgeType", bound=BaseEdge)


class BaseGraph(BaseModel, Generic[NodeType, EdgeType]):
    """Base graph implementation that can be used for both chains and conversations."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    nodes: Dict[str, NodeType] = Field(default_factory=dict)
    edges: List[EdgeType] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    def add_node(self, node: NodeType) -> str:
        """Add a node to the graph."""
        node_id = str(node.id)
        self.nodes[node_id] = node
        self.updated_at = datetime.now()
        return node_id

    def add_edge(self, edge: EdgeType) -> None:
        """Add an edge to the graph."""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError("Both source and target nodes must exist in the graph")

        self.edges.append(edge)
        self.updated_at = datetime.now()

    def get_node_dependencies(self, node_id: str) -> List[str]:
        """Get IDs of nodes that must complete before this node."""
        return [edge.source_id for edge in self.edges if edge.target_id == node_id]

    def get_node_dependents(self, node_id: str) -> List[str]:
        """Get IDs of nodes that depend on this node."""
        return [edge.target_id for edge in self.edges if edge.source_id == node_id]

    def get_node_history(self, node_id: str, max_depth: Optional[int] = None) -> List[NodeType]:
        """Get the history of nodes leading to the specified node."""
        history = []
        visited = set()

        def traverse(current_id: str, depth: int = 0) -> None:
            if current_id in visited or (max_depth is not None and depth >= max_depth):
                return
            visited.add(current_id)

            # Get incoming edges to current node
            incoming = [edge for edge in self.edges if edge.target_id == current_id]
            for edge in incoming:
                source_node = self.nodes[edge.source_id]
                traverse(edge.source_id, depth + 1)
                history.append(source_node)

        traverse(node_id)
        return history

    def get_execution_order(self) -> List[List[str]]:
        """Get nodes grouped by execution level (for parallel execution)."""
        # Initialize in-degree count for each node
        in_degree = {node_id: 0 for node_id in self.nodes}
        for edge in self.edges:
            in_degree[edge.target_id] += 1

        # Group nodes by level
        levels: List[List[str]] = []
        while in_degree:
            # Get all nodes with no dependencies
            current_level = [node_id for node_id, degree in in_degree.items() if degree == 0]
            if not current_level:
                raise ValueError("Cycle detected in graph")

            levels.append(current_level)

            # Remove processed nodes and update dependencies
            for node_id in current_level:
                del in_degree[node_id]
                for edge in self.edges:
                    if edge.source_id == node_id and edge.target_id in in_degree:
                        in_degree[edge.target_id] -= 1

        return levels

    def prune_nodes(
        self,
        older_than: Optional[datetime] = None,
        max_nodes: Optional[int] = None,
        exclude_nodes: Optional[Set[str]] = None,
    ) -> None:
        """Prune nodes from the graph based on age or count while maintaining graph integrity."""
        if not older_than and not max_nodes:
            return

        nodes_to_remove = set()
        exclude_nodes = exclude_nodes or set()

        # Identify nodes to remove based on age
        if older_than:
            nodes_to_remove.update(
                node_id
                for node_id, node in self.nodes.items()
                if node.created_at < older_than and node_id not in exclude_nodes
            )

        # Identify nodes to remove based on count while preserving recent ones
        if max_nodes and len(self.nodes) > max_nodes:
            sorted_nodes = sorted(
                [(nid, n) for nid, n in self.nodes.items() if nid not in exclude_nodes],
                key=lambda x: x[1].created_at,
                reverse=True,
            )
            nodes_to_keep = sorted_nodes[:max_nodes]
            nodes_to_remove.update(node_id for node_id, _ in sorted_nodes[max_nodes:])

        # Remove edges connected to nodes being removed
        self.edges = [
            edge
            for edge in self.edges
            if edge.source_id not in nodes_to_remove and edge.target_id not in nodes_to_remove
        ]

        # Remove the nodes
        for node_id in nodes_to_remove:
            self.nodes.pop(node_id, None)

        self.updated_at = datetime.now()

    def get_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the graph including metrics and statistics."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
