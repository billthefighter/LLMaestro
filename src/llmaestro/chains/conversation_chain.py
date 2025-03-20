"""Conversation-aware chain system for LLM orchestration."""

import asyncio
from typing import Any, Dict, List, Optional, cast, Type, TypeVar, Generic
from uuid import uuid4
from pydantic import Field

from llmaestro.chains.chains import ChainEdge, ChainGraph, ChainNode, NodeType, RetryStrategy, ChainStep, ChainMetadata
from llmaestro.core.conversations import ConversationGraph, ConversationNode
from llmaestro.core.orchestrator import ExecutionMetadata, Orchestrator
from llmaestro.prompts.implementations import SimplePrompt

P = TypeVar("P", bound=SimplePrompt)


class ConversationChainNode(ChainNode):
    """A chain node that integrates with the conversation system."""

    conversation: Optional[ConversationGraph] = None
    prompt_node: Optional[ConversationNode] = None
    response_node: Optional[ConversationNode] = None
    prompt: Optional[SimplePrompt] = Field(default=None)  # Use Field with default=None

    @classmethod
    async def create(
        cls,
        prompt: SimplePrompt,
        node_type: NodeType,
        retry_strategy: Optional[RetryStrategy] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ConversationChainNode":
        """Create a new conversation chain node."""
        # Create a ChainStep for the prompt
        step = ChainStep(
            prompt=prompt,
            retry_strategy=retry_strategy or RetryStrategy(),
        )

        # Create the node
        return cls(
            id=str(uuid4()),
            step=step,
            node_type=node_type,
            metadata=ChainMetadata(**(metadata or {})),  # Convert dict to ChainMetadata
            prompt=prompt,  # Store the prompt for later use
        )


class ConversationChain(ChainGraph, Generic[P]):
    """A chain that integrates with the conversation system.

    Generic Parameters:
        P: The type of prompt to use in this chain (must inherit from SimplePrompt)
    """

    def __init__(self, orchestrator: Orchestrator, prompt_type: Optional[Type[P]] = None):
        """Initialize the conversation chain.

        Args:
            orchestrator: The orchestrator to use for execution
            prompt_type: The type of prompt to use for this chain (must inherit from SimplePrompt)
        """
        super().__init__(prompt_type=prompt_type)
        self.orchestrator = orchestrator
        self.conversation: Optional[ConversationGraph] = None

    async def initialize(
        self, name: str, initial_prompt: SimplePrompt, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the chain with a conversation."""
        self.conversation = await self.orchestrator.create_conversation(
            name=name, initial_prompt=initial_prompt, metadata=metadata
        )

    async def add_prompt_node(
        self,
        prompt: SimplePrompt,
        dependencies: Optional[List[ConversationChainNode]] = None,
        retry_strategy: Optional[RetryStrategy] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationChainNode:
        """Add a prompt node to both the chain and conversation.

        Args:
            prompt: The prompt to add (can be any prompt type that inherits from SimplePrompt)
            dependencies: Optional list of node objects that this node depends on
            retry_strategy: Optional configuration for retry behavior
            metadata: Optional metadata for the node

        Returns:
            The conversation chain node

        Raises:
            ValueError: If the chain is not initialized with a conversation
        """
        if not self.conversation:
            raise ValueError("Chain not initialized with conversation")

        # Create chain node
        node_result = await ConversationChainNode.create(
            prompt=prompt, node_type=NodeType.SEQUENTIAL, retry_strategy=retry_strategy, metadata=metadata
        )
        node = cast(ConversationChainNode, self.add_node(node_result))

        # Add dependencies
        if dependencies:
            for dep_node in dependencies:
                # Create and add the edge
                edge = ChainEdge(source=dep_node, target=node, edge_type="depends_on")
                self.add_edge(edge)

        return node

    async def execute_node(self, node: ConversationChainNode) -> ConversationNode:
        """Execute a single node in the chain."""
        if not self.conversation:
            raise ValueError("Chain not initialized with conversation")

        # Get dependencies - ensure all are strings and not None
        dependencies: List[str] = []
        for edge in self.edges:
            if edge.target_id == node.id and edge.edge_type == "depends_on":
                source_node = cast(ConversationChainNode, self.nodes[edge.source_id])
                if source_node.response_node is not None:
                    dependencies.append(source_node.response_node.id)

        # Ensure prompt is not None
        if node.prompt is None:
            raise ValueError(f"Node {node.id} has no prompt")

        # Execute in conversation
        response_id = await self.orchestrator.execute_prompt(
            conversation=self.conversation, prompt=node.prompt, dependencies=dependencies if dependencies else None
        )

        # Get the response node
        response_node = self.conversation.nodes.get(response_id)
        if response_node is None:
            raise ValueError(f"Response node with ID {response_id} not found in conversation")

        # Update node with conversation references
        node.conversation = self.conversation
        node.response_node = response_node

        return response_node

    async def execute(self) -> Dict[str, ConversationNode]:
        """Execute the entire chain.

        Returns:
            Dict mapping chain node IDs to conversation response nodes
        """
        if not self.conversation:
            raise ValueError("Chain not initialized with conversation")

        results: Dict[str, ConversationNode] = {}
        execution_order = self.get_execution_order()

        # Execute nodes level by level
        for level in execution_order:
            # Execute nodes in current level in parallel
            level_tasks = []
            for node_id in level:
                node = self.nodes.get(node_id)
                if isinstance(node, ConversationChainNode):
                    level_tasks.append(self.execute_node(node))

            level_results = await asyncio.gather(*level_tasks)

            # Store results
            for node_id, response_node in zip([n.id for n in level_tasks], level_results, strict=False):
                results[node_id] = response_node

        return results

    def get_node_status(self, node: ConversationChainNode) -> ExecutionMetadata:
        """Get the execution status of a node."""
        if not self.conversation:
            raise ValueError("Chain not initialized with conversation")

        if not node.response_node:
            raise ValueError(f"Node {node.id} has not been executed")

        return self.orchestrator.get_execution_status(
            conversation=self.conversation,  # Pass the conversation object
            node_id=node.response_node.id,
        )
