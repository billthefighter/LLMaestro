"""Base graph implementation for LLM orchestration."""

from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar, Type
from uuid import uuid4

from pydantic import ConfigDict, Field
from llmaestro.core.persistence import PersistentModel
from llmaestro.prompts.base import BasePrompt


class BaseNode(PersistentModel):
    """Base class for all graph nodes."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)


NodeType = TypeVar("NodeType", bound=BaseNode)
EdgeType = TypeVar("EdgeType", bound="BaseEdge")
PromptType = TypeVar("PromptType", bound=BasePrompt)


class BaseEdge(PersistentModel, Generic[NodeType]):
    """Base class for all graph edges."""

    source: NodeType = Field(..., description="Source node")
    target: NodeType = Field(..., description="Target node")
    edge_type: str = Field(..., description="Type of relationship")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    @property
    def source_id(self) -> str:
        """Get the ID of the source node."""
        return self.source.id

    @property
    def target_id(self) -> str:
        """Get the ID of the target node."""
        return self.target.id


class BaseGraph(PersistentModel, Generic[NodeType, EdgeType, PromptType]):
    """Base graph implementation that can be used for both chains and conversations.

    Generic Parameters:
        NodeType: The type of nodes in the graph
        EdgeType: The type of edges in the graph
        PromptType: The type of prompts to use in this graph
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    nodes: Dict[str, NodeType] = Field(default_factory=dict)
    edges: List[Any] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # The type of prompt to use in this graph
    prompt_type: Optional[Type[PromptType]] = None

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    def __init__(self, **data: Any):
        """Initialize the graph with optional prompt_type validation."""
        super().__init__(**data)

        # Validate prompt_type if provided
        if self.prompt_type is not None:
            if not isinstance(self.prompt_type, type) or not issubclass(self.prompt_type, BasePrompt):
                raise TypeError(f"prompt_type must be a subclass of BasePrompt, got {self.prompt_type}")
            # Verify it's not the abstract base class itself
            if self.prompt_type is BasePrompt:
                raise TypeError("prompt_type cannot be BasePrompt itself as it's an abstract base class")

    def create_prompt(self, **kwargs: Any) -> PromptType:
        """Create a prompt of the type specified in prompt_type.

        Args:
            **kwargs: Keyword arguments to pass to the prompt constructor

        Returns:
            A prompt of the specified type

        Raises:
            TypeError: If prompt_type is not set or is not a valid subclass of BasePrompt
        """
        if self.prompt_type is None:
            raise TypeError("No prompt_type specified for this graph")

        return self.prompt_type(**kwargs)

    def add_node(self, node: NodeType) -> str:
        """Add a node to the graph."""
        node_id = str(node.id)
        self.nodes[node_id] = node
        self.updated_at = datetime.now()
        return node_id

    def add_edge(self, edge: EdgeType) -> None:
        """Add an edge to the graph."""
        source_id = edge.source_id
        target_id = edge.target_id

        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Both source and target nodes must exist in the graph")

        self.edges.append(edge)
        self.updated_at = datetime.now()

    def create_edge(
        self, source: NodeType, target: NodeType, edge_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> EdgeType:
        """Create an edge between two nodes.

        This method should be implemented by subclasses to create the specific edge type.
        """
        raise NotImplementedError("Subclasses must implement create_edge")

    def connect(
        self, source: NodeType, target: NodeType, edge_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Connect two nodes with an edge."""
        edge = self.create_edge(source, target, edge_type, metadata or {})
        self.add_edge(edge)

    def get_node_dependencies(self, node_id: str) -> List[NodeType]:
        """Get nodes that must complete before this node."""
        return [edge.source for edge in self.edges if edge.target_id == node_id]

    def get_node_dependents(self, node_id: str) -> List[NodeType]:
        """Get nodes that depend on this node."""
        return [edge.target for edge in self.edges if edge.source_id == node_id]

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
                source_node = edge.source
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
