"""Conversation-aware chain system for LLM orchestration."""

import asyncio
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

from llmaestro.chains.chains import ChainEdge, ChainGraph, ChainNode, NodeType, RetryStrategy
from llmaestro.core.conversations import ConversationGraph
from llmaestro.core.orchestrator import ExecutionMetadata, Orchestrator
from llmaestro.prompts.base import BasePrompt


class ConversationChainNode(ChainNode):
    """A chain node that integrates with the conversation system."""

    conversation_id: Optional[str] = None
    prompt_node_id: Optional[str] = None
    response_node_id: Optional[str] = None

    @classmethod
    async def create(
        cls,
        prompt: BasePrompt,
        node_type: NodeType,
        retry_strategy: Optional[RetryStrategy] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ConversationChainNode":
        """Create a new conversation chain node."""
        return cls(
            id=str(uuid4()),
            prompt=prompt,
            node_type=node_type,
            retry_strategy=retry_strategy or RetryStrategy(),
            metadata=metadata or {},
        )


class ConversationChain(ChainGraph):
    """A chain that integrates with the conversation system."""

    def __init__(self, orchestrator: Orchestrator):
        super().__init__()
        self.orchestrator = orchestrator
        self.conversation: Optional[ConversationGraph] = None

    async def initialize(
        self, name: str, initial_prompt: BasePrompt, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the chain with a conversation."""
        self.conversation = await self.orchestrator.create_conversation(
            name=name, initial_prompt=initial_prompt, metadata=metadata
        )

    async def add_prompt_node(
        self,
        prompt: BasePrompt,
        dependencies: Optional[List[str]] = None,
        retry_strategy: Optional[RetryStrategy] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a prompt node to both the chain and conversation."""
        if not self.conversation:
            raise ValueError("Chain not initialized with conversation")

        # Create chain node
        node = await ConversationChainNode.create(
            prompt=prompt, node_type=NodeType.SEQUENTIAL, retry_strategy=retry_strategy, metadata=metadata
        )
        node_id = self.add_node(node)

        # Add dependencies
        if dependencies:
            for dep_id in dependencies:
                self.add_edge(ChainEdge(source_id=dep_id, target_id=node_id, edge_type="depends_on"))

        return node_id

    async def execute_node(self, node_id: str) -> str:
        """Execute a single node in the chain."""
        if not self.conversation:
            raise ValueError("Chain not initialized with conversation")

        node = self.nodes.get(node_id)
        if not isinstance(node, ConversationChainNode):
            raise ValueError(f"Node {node_id} is not a ConversationChainNode")

        # Get dependencies
        dependencies = [
            cast(ConversationChainNode, self.nodes[edge.source_id]).response_node_id
            for edge in self.edges
            if edge.target_id == node_id and edge.edge_type == "depends_on"
        ]

        # Execute in conversation
        response_id = await self.orchestrator.execute_prompt(
            conversation_id=self.conversation.id, prompt=node.prompt, dependencies=dependencies
        )

        # Update node with conversation references
        node.conversation_id = self.conversation.id
        node.response_node_id = response_id

        return response_id

    async def execute(self) -> Dict[str, str]:
        """Execute the entire chain.

        Returns:
            Dict mapping chain node IDs to conversation response node IDs
        """
        if not self.conversation:
            raise ValueError("Chain not initialized with conversation")

        results: Dict[str, str] = {}
        execution_order = self.get_execution_order()

        # Execute nodes level by level
        for level in execution_order:
            # Execute nodes in current level in parallel
            level_tasks = [self.execute_node(node_id) for node_id in level]
            level_results = await asyncio.gather(*level_tasks)

            # Store results
            for node_id, response_id in zip(level, level_results, strict=False):
                results[node_id] = response_id

        return results

    def get_node_status(self, node_id: str) -> ExecutionMetadata:
        """Get the execution status of a node."""
        if not self.conversation:
            raise ValueError("Chain not initialized with conversation")

        node = self.nodes.get(node_id)
        if not isinstance(node, ConversationChainNode):
            raise ValueError(f"Node {node_id} is not a ConversationChainNode")

        if not node.response_node_id:
            raise ValueError(f"Node {node_id} has not been executed")

        return self.orchestrator.get_execution_status(
            conversation_id=self.conversation.id, node_id=node.response_node_id
        )
