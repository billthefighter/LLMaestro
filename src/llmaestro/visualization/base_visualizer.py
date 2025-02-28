"""Base visualization components for graph structures."""

from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar

from llmaestro.core.graph import BaseEdge, BaseGraph, BaseNode


@dataclass
class CytoscapeNode:
    """Represents a node in the Cytoscape graph."""

    id: str
    label: str
    type: str
    data: Dict[str, Any]


@dataclass
class CytoscapeEdge:
    """Represents an edge in the Cytoscape graph."""

    id: str
    source: str
    target: str
    label: str
    data: Dict[str, Any]


@dataclass
class CytoscapeGraph:
    """Complete graph representation for Cytoscape."""

    nodes: List[CytoscapeNode]
    edges: List[CytoscapeEdge]

    def to_cytoscape_format(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert to Cytoscape.js compatible format."""
        return {
            "nodes": [
                {"data": {"id": node.id, "label": node.label, "type": node.type, **node.data}} for node in self.nodes
            ],
            "edges": [
                {
                    "data": {
                        "id": edge.id,
                        "source": edge.source,
                        "target": edge.target,
                        "label": edge.label,
                        **edge.data,
                    }
                }
                for edge in self.edges
            ],
        }


NodeType = TypeVar("NodeType", bound=BaseNode)
EdgeType = TypeVar("EdgeType", bound=BaseEdge)


class BaseVisualizer(Generic[NodeType, EdgeType]):
    """Base class for graph visualization."""

    def __init__(self):
        self.nodes: List[CytoscapeNode] = []
        self.edges: List[CytoscapeEdge] = []
        self.processed_nodes: set[str] = set()
        self.edge_counter = 0

    def _add_node(self, node_id: str, label: str, type_: str, data: Dict[str, Any]) -> None:
        """Add a node if it doesn't already exist."""
        if not any(n.id == node_id for n in self.nodes):
            self.nodes.append(CytoscapeNode(id=node_id, label=label, type=type_, data=data))

    def _add_edge(self, source: str, target: str, label: str, data: Dict[str, Any]) -> None:
        """Add an edge between nodes."""
        self.edge_counter += 1
        self.edges.append(
            CytoscapeEdge(id=f"e{self.edge_counter}", source=source, target=target, label=label, data=data)
        )

    def _process_node(self, node: NodeType, parent_id: Optional[str] = None) -> None:
        """Process a single node. Override this in subclasses."""
        raise NotImplementedError

    def process_graph(self, graph: BaseGraph[NodeType, EdgeType], parent_id: Optional[str] = None) -> None:
        """Process a graph and extract its structure."""
        # Process all nodes
        for node_id, node in graph.nodes.items():
            if node_id not in self.processed_nodes:
                self._process_node(node, parent_id)
                self.processed_nodes.add(node_id)

        # Process all edges
        for edge in graph.edges:
            self._add_edge(source=edge.source_id, target=edge.target_id, label=edge.edge_type, data=edge.metadata)

    def get_visualization_data(self) -> CytoscapeGraph:
        """Get the complete graph data in Cytoscape format."""
        return CytoscapeGraph(nodes=self.nodes, edges=self.edges)
