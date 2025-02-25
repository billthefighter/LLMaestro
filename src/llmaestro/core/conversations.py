"""Models for representing and managing conversation structures."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field

from llmaestro.core.models import LLMResponse
from llmaestro.prompts.base import BasePrompt




class ConversationNode(BaseModel):
    """Represents a single node in the conversation graph."""

    id: str = Field(..., description="Unique identifier for this node")
    content: Union[BasePrompt, LLMResponse] = Field(..., description="The prompt or response content")
    node_type: str = Field(..., description="Type of node (prompt/response)")
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for this node")

    model_config = ConfigDict(validate_assignment=True)


class ConversationEdge(BaseModel):
    """Represents a directed edge between conversation nodes."""

    source_id: str = Field(..., description="ID of the source node")
    target_id: str = Field(..., description="ID of the target node")
    edge_type: str = Field(
        ..., description="Type of relationship (e.g., 'response_to', 'references', 'continues_from')"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for this edge")

    model_config = ConfigDict(validate_assignment=True)


class ConversationGraph(BaseModel):
    """A graph-based representation of an LLM conversation."""

    id: str = Field(..., description="Unique identifier for this conversation")
    nodes: Dict[str, ConversationNode] = Field(default_factory=dict, description="Map of node IDs to nodes")
    edges: List[ConversationEdge] = Field(default_factory=list, description="List of edges connecting nodes")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Conversation-level metadata including model info, context, etc."
    )

    model_config = ConfigDict(validate_assignment=True)

    def add_node(
        self, content: Union[BasePrompt, LLMResponse], node_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new node to the conversation graph."""
        node_id = str(uuid4())
        self.nodes[node_id] = ConversationNode(
            id=node_id, content=content, node_type=node_type, metadata=metadata or {}
        )
        self.updated_at = datetime.now()
        return node_id

    def add_edge(
        self, source_id: str, target_id: str, edge_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new edge between nodes in the conversation graph."""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Both source and target nodes must exist in the graph")

        edge = ConversationEdge(source_id=source_id, target_id=target_id, edge_type=edge_type, metadata=metadata or {})
        self.edges.append(edge)
        self.updated_at = datetime.now()

    def get_node_history(self, node_id: str, max_depth: Optional[int] = None) -> List[ConversationNode]:
        """Get the history of nodes leading to the specified node."""
        history = []
        visited = set()

        def traverse(current_id: str, depth: int = 0):
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

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation including metrics and statistics."""
        prompt_nodes = [node for node in self.nodes.values() if node.node_type == "prompt"]
        response_nodes = [node for node in self.nodes.values() if node.node_type == "response"]

        # Calculate token usage if available
        total_tokens = 0
        for node in response_nodes:
            if isinstance(node.content, LLMResponse) and node.content.token_usage:
                total_tokens += node.content.token_usage.total_tokens

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "prompt_count": len(prompt_nodes),
            "response_count": len(response_nodes),
            "total_tokens": total_tokens,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    def prune_nodes(self, older_than: Optional[datetime] = None, max_nodes: Optional[int] = None) -> None:
        """Prune nodes from the graph based on age or count while maintaining graph integrity."""
        if not older_than and not max_nodes:
            return

        nodes_to_remove = set()

        # Identify nodes to remove based on age
        if older_than:
            nodes_to_remove.update(node_id for node_id, node in self.nodes.items() if node.created_at < older_than)

        # Identify nodes to remove based on count while preserving recent ones
        if max_nodes and len(self.nodes) > max_nodes:
            sorted_nodes = sorted(self.nodes.items(), key=lambda x: x[1].created_at, reverse=True)
            _nodes_to_keep = sorted_nodes[:max_nodes]
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

class ConversationContext(BaseModel):
    """Represents the current conversation context with graph-based history tracking."""
    
    graph: ConversationGraph = Field(
        default_factory=lambda: ConversationGraph(id=str(uuid4())),
        description="The underlying conversation graph"
    )
    current_message: Optional[str] = Field(
        default=None, 
        description="ID of the current node being processed"
    )
    max_nodes: Optional[int] = Field(
        default=None,
        description="Maximum number of nodes before auto-pruning. None means no auto-pruning."
    )
    
    model_config = ConfigDict(validate_assignment=True)
    
    @computed_field
    @property
    def messages(self) -> List[ConversationNode]:
        """Get the history of nodes leading to the current node."""
        if not self.current_message:
            return []
        return self.graph.get_node_history(self.current_message)
    
    @computed_field
    @property
    def message_count(self) -> int:
        """Get the total number of messages in the history."""
        return len(self.messages)
    
    @computed_field
    @property
    def initial_task(self) -> Optional[BasePrompt]:
        """Get the root prompt that started this conversation."""
        if not self.messages:
            return None
            
        # Find the first prompt node
        for node in reversed(self.messages):
            if (
                node.node_type == "prompt" 
                and isinstance(node.content, BasePrompt)
            ):
                return node.content
        return None
    
    def add_node(
        self, 
        content: Union[BasePrompt, LLMResponse], 
        node_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new node to the conversation graph and optionally set it as current."""
        # Add the node
        node_id = self.graph.add_node(content, node_type, metadata)
        
        # If this is the first node, set it as current
        if not self.current_message:
            self.current_message = node_id
            
        # Add edge from current message if it exists
        if self.current_message and self.current_message != node_id:
            self.graph.add_edge(
                source_id=self.current_message,
                target_id=node_id,
                edge_type="next"
            )
            
        # Auto-prune if max_nodes is set
        if self.max_nodes and len(self.graph.nodes) > self.max_nodes:
            self.graph.prune_nodes(max_nodes=self.max_nodes)
            
        return node_id
    
    def set_node(self, node_id: str) -> None:
        """Set the current message to a specific node ID."""
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found in conversation graph")
        self.current_message = node_id