"""Models for representing and managing conversation structures."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from llmaestro.core.graph import BaseEdge, BaseGraph, BaseNode
from llmaestro.core.models import LLMResponse, TokenUsage
from llmaestro.prompts.base import BasePrompt


class ConversationNode(BaseNode):
    """Represents a single node in the conversation graph."""

    content: Union[BasePrompt, LLMResponse] = Field(..., description="The prompt or response content")
    node_type: str = Field(..., description="Type of node (prompt/response)")

    @property
    def token_usage(self) -> Optional[TokenUsage]:
        """Get token usage for this node if available."""
        if isinstance(self.content, LLMResponse) and self.content.token_usage:
            return self.content.token_usage
        return None


class ConversationEdge(BaseEdge):
    """Represents a directed edge between conversation nodes."""

    model_config = ConfigDict(validate_assignment=True)


class ConversationGraph(BaseGraph[ConversationNode, ConversationEdge]):
    """A graph-based representation of an LLM conversation."""

    @property
    def total_tokens(self) -> TokenUsage:
        """Get total token usage across all nodes."""
        total_completion = 0
        total_prompt = 0
        total = 0

        for node in self.nodes.values():
            if token_usage := node.token_usage:
                total_completion += token_usage.completion_tokens
                total_prompt += token_usage.prompt_tokens
                total += token_usage.total_tokens

        return TokenUsage(completion_tokens=total_completion, prompt_tokens=total_prompt, total_tokens=total)

    def get_token_usage_by_type(self, node_type: str) -> TokenUsage:
        """Get token usage for all nodes of a specific type."""
        completion = 0
        prompt = 0
        total = 0

        for node in self.nodes.values():
            if node.node_type == node_type and (token_usage := node.token_usage):
                completion += token_usage.completion_tokens
                prompt += token_usage.prompt_tokens
                total += token_usage.total_tokens

        return TokenUsage(completion_tokens=completion, prompt_tokens=prompt, total_tokens=total)

    def add_conversation_node(
        self, content: Union[BasePrompt, LLMResponse], node_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new node to the conversation graph."""
        node = ConversationNode(content=content, node_type=node_type, metadata=metadata or {})
        return self.add_node(node)

    def add_conversation_edge(
        self, source_id: str, target_id: str, edge_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new edge between nodes in the conversation graph."""
        edge = ConversationEdge(source_id=source_id, target_id=target_id, edge_type=edge_type, metadata=metadata or {})
        self.add_edge(edge)

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation including metrics and statistics."""
        prompt_nodes = [node for node in self.nodes.values() if node.node_type == "prompt"]
        response_nodes = [node for node in self.nodes.values() if node.node_type == "response"]

        token_usage = self.total_tokens

        return {
            **self.get_graph_summary(),
            "prompt_count": len(prompt_nodes),
            "response_count": len(response_nodes),
            "token_usage": {
                "total": token_usage.total_tokens,
                "prompt": token_usage.prompt_tokens,
                "completion": token_usage.completion_tokens,
            },
        }


class ConversationContext(BaseModel):
    """Represents the current conversation context with graph-based history tracking."""

    graph: ConversationGraph = Field(
        default_factory=lambda: ConversationGraph(id=str(uuid4())), description="The underlying conversation graph"
    )
    current_message: Optional[str] = Field(default=None, description="ID of the current node being processed")
    max_nodes: Optional[int] = Field(
        default=None, description="Maximum number of nodes before auto-pruning. None means no auto-pruning."
    )

    model_config = ConfigDict(validate_assignment=True)

    @property
    def messages(self) -> List[ConversationNode]:
        """Get the history of nodes leading to the current node."""
        if not self.current_message:
            return []
        return self.graph.get_node_history(self.current_message)

    @property
    def message_count(self) -> int:
        """Get the total number of messages in the history."""
        return len(self.messages)

    @property
    def initial_task(self) -> Optional[BasePrompt]:
        """Get the root prompt that started this conversation."""
        if not self.messages:
            return None

        # Find the first prompt node
        for node in reversed(self.messages):
            if node.node_type == "prompt" and isinstance(node.content, BasePrompt):
                return node.content
        return None

    @property
    def total_tokens(self) -> TokenUsage:
        """Get total token usage for the conversation."""
        return self.graph.total_tokens

    @property
    def prompt_tokens(self) -> TokenUsage:
        """Get token usage for prompt messages."""
        return self.graph.get_token_usage_by_type("prompt")

    @property
    def response_tokens(self) -> TokenUsage:
        """Get token usage for response messages."""
        return self.graph.get_token_usage_by_type("response")

    def get_token_usage_since(self, timestamp: datetime) -> TokenUsage:
        """Get token usage since a specific timestamp."""
        completion = 0
        prompt = 0
        total = 0

        for node in self.graph.nodes.values():
            if node.created_at >= timestamp and (token_usage := node.token_usage):
                completion += token_usage.completion_tokens
                prompt += token_usage.prompt_tokens
                total += token_usage.total_tokens

        return TokenUsage(completion_tokens=completion, prompt_tokens=prompt, total_tokens=total)

    def add_node(
        self, content: Union[BasePrompt, LLMResponse], node_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new node to the conversation graph and optionally set it as current."""
        # Add the node
        node_id = self.graph.add_conversation_node(content, node_type, metadata)

        # If this is the first node, set it as current
        if not self.current_message:
            self.current_message = node_id

        # Add edge from current message if it exists
        if self.current_message and self.current_message != node_id:
            self.graph.add_conversation_edge(source_id=self.current_message, target_id=node_id, edge_type="next")

        # Auto-prune if max_nodes is set
        if self.max_nodes and len(self.graph.nodes) > self.max_nodes:
            self.graph.prune_nodes(max_nodes=self.max_nodes)

        return node_id

    def set_node(self, node_id: str) -> None:
        """Set the current message to a specific node ID."""
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found in conversation graph")
        self.current_message = node_id


def get_detailed_conversation_dump(conversation: ConversationGraph) -> Dict[str, Any]:
    """Get a detailed structured dump of a conversation."""
    return {
        # Basic summary
        **conversation.get_conversation_summary(),
        # Detailed node information
        "nodes": {
            node_id: {
                "type": node.node_type,
                "created_at": node.created_at.isoformat(),
                "content": node.content.model_dump() if hasattr(node.content, "model_dump") else str(node.content),
                "metadata": node.metadata,
                "token_usage": node.token_usage.model_dump() if node.token_usage else None,
            }
            for node_id, node in conversation.nodes.items()
        },
        # Edge relationships
        "edges": [
            {"source": edge.source_id, "target": edge.target_id, "type": edge.edge_type, "metadata": edge.metadata}
            for edge in conversation.edges
        ],
        # Token usage by type
        "token_usage_by_type": {
            "prompt": conversation.get_token_usage_by_type("prompt").model_dump(),
            "response": conversation.get_token_usage_by_type("response").model_dump(),
        },
    }
