"""Orchestration layer for managing LLM conversations and execution."""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Callable, Awaitable
from uuid import uuid4

from pydantic import BaseModel

from llmaestro.core.conversations import ConversationGraph, ConversationNode
from llmaestro.core.models import LLMResponse
from llmaestro.prompts.base import BasePrompt

if TYPE_CHECKING:
    from llmaestro.agents.agent_pool import AgentPool


class ExecutionMetadata(BaseModel):
    """Metadata for tracking execution status of nodes."""

    status: str = "pending"  # pending, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    parallel_group: Optional[str] = None
    dependencies: List[str] = []


class Orchestrator:
    """Manages LLM conversation execution and resource coordination."""

    def __init__(self, agent_pool: "AgentPool"):
        self.agent_pool = agent_pool
        self.active_conversations: Dict[str, ConversationGraph] = {}
        self.active_conversation_id: Optional[str] = None

        # Callback functions for visualization
        self.conversation_created_callback: Optional[Callable[[ConversationGraph], Awaitable[None]]] = None
        self.conversation_updated_callback: Optional[Callable[[ConversationGraph], Awaitable[None]]] = None
        self.node_added_callback: Optional[Callable[[str, str], Awaitable[None]]] = None
        self.node_updated_callback: Optional[Callable[[str, str], Awaitable[None]]] = None

    def on_conversation_created(self, handler: Callable[[ConversationGraph], Awaitable[None]]) -> None:
        """Register a handler for conversation creation events."""
        self.conversation_created_callback = handler

    def on_conversation_updated(self, handler: Callable[[ConversationGraph], Awaitable[None]]) -> None:
        """Register a handler for conversation update events."""
        self.conversation_updated_callback = handler

    def on_node_added(self, handler: Callable[[str, str], Awaitable[None]]) -> None:
        """Register a handler for node addition events."""
        self.node_added_callback = handler

    def on_node_updated(self, handler: Callable[[str, str], Awaitable[None]]) -> None:
        """Register a handler for node update events."""
        self.node_updated_callback = handler

    async def _notify_conversation_created(self, conversation: ConversationGraph) -> None:
        """Notify callback of conversation creation."""
        if self.conversation_created_callback:
            await self.conversation_created_callback(conversation)

    async def _notify_conversation_updated(self, conversation: ConversationGraph) -> None:
        """Notify callback of conversation update."""
        if self.conversation_updated_callback:
            await self.conversation_updated_callback(conversation)

    async def _notify_node_added(self, conversation_id: str, node_id: str) -> None:
        """Notify callback of node addition."""
        if self.node_added_callback:
            await self.node_added_callback(conversation_id, node_id)

    async def _notify_node_updated(self, conversation_id: str, node_id: str) -> None:
        """Notify callback of node update."""
        if self.node_updated_callback:
            await self.node_updated_callback(conversation_id, node_id)

    def _resolve_conversation_id(
        self, conversation: Optional[Union[str, ConversationGraph, ConversationNode]] = None
    ) -> Optional[str]:
        """Resolve conversation ID from various input types."""
        if conversation is None:
            return self.active_conversation_id
        elif isinstance(conversation, str):
            return conversation
        elif isinstance(conversation, ConversationGraph):
            return conversation.id
        elif isinstance(conversation, ConversationNode):
            # Find the conversation containing this node
            for conv_id, conv in self.active_conversations.items():
                if conversation.id in conv.nodes:
                    return conv_id
            raise ValueError(f"Node {conversation.id} not found in any active conversation")
        else:
            raise TypeError(f"Unsupported conversation identifier type: {type(conversation)}")

    def _get_conversation(
        self, conversation: Optional[Union[str, ConversationGraph, ConversationNode]] = None
    ) -> ConversationGraph:
        """Get conversation from various input types with validation.

        Args:
            conversation: Can be one of:
                - None: Uses active conversation
                - str: Conversation ID
                - ConversationGraph: Direct conversation object
                - ConversationNode: Node from which to find parent conversation

        Returns:
            ConversationGraph: The resolved conversation

        Raises:
            ValueError: If no conversation could be found or if input is invalid
            TypeError: If input type is not supported
        """
        # If passed a ConversationGraph directly, validate and return it
        if isinstance(conversation, ConversationGraph):
            if conversation.id not in self.active_conversations:
                raise ValueError(f"Conversation {conversation.id} not found in active conversations")
            return conversation

        # Otherwise resolve the ID
        conv_id = self._resolve_conversation_id(conversation)
        if not conv_id:
            raise ValueError("No active conversation")

        conversation = self.active_conversations.get(conv_id)
        if not conversation:
            raise ValueError(f"Conversation {conv_id} not found")

        return conversation

    def set_active_conversation(self, conversation: Union[str, ConversationGraph, ConversationNode]) -> None:
        """Set the active conversation using various input types."""
        conv_id = self._resolve_conversation_id(conversation)
        if not conv_id or conv_id not in self.active_conversations:
            raise ValueError(f"Conversation {conv_id} not found")
        self.active_conversation_id = conv_id

    def get_active_conversation(self) -> Optional[ConversationGraph]:
        """Get the currently active conversation."""
        if not self.active_conversation_id:
            return None
        return self.active_conversations.get(self.active_conversation_id)

    async def create_conversation(
        self, name: str, initial_prompt: BasePrompt, metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationGraph:
        """Create a new conversation with an initial prompt."""
        conversation = ConversationGraph(id=str(uuid4()), metadata={"name": name, **(metadata or {})})

        # Add initial prompt node
        node_id = conversation.add_conversation_node(
            content=initial_prompt, node_type="prompt", metadata={"execution": ExecutionMetadata().model_dump()}
        )

        self.active_conversations[conversation.id] = conversation
        self.active_conversation_id = conversation.id  # Set as active by default

        # Notify handlers
        await self._notify_conversation_created(conversation)
        await self._notify_node_added(conversation.id, node_id)

        return conversation

    async def execute_prompt_in_active(self, prompt: BasePrompt, dependencies: Optional[List[str]] = None) -> str:
        """Execute prompt in active conversation with built-in validation."""
        conversation = self._get_conversation()
        return await self.execute_prompt(conversation.id, prompt, dependencies)

    async def execute_prompt(
        self,
        conversation: Union[str, ConversationGraph, ConversationNode],
        prompt: BasePrompt,
        dependencies: Optional[List[str]] = None,
        parallel_group: Optional[str] = None,
    ) -> str:
        """Execute a prompt and add it to the conversation."""
        conversation = self._get_conversation(conversation)

        # Create execution metadata
        exec_metadata = ExecutionMetadata(
            status="pending", started_at=datetime.now(), parallel_group=parallel_group, dependencies=dependencies or []
        )

        # Add prompt node
        prompt_node_id = conversation.add_conversation_node(
            content=prompt, node_type="prompt", metadata={"execution": exec_metadata.model_dump()}
        )
        await self._notify_node_added(conversation.id, prompt_node_id)

        # Add dependency edges
        if dependencies:
            for dep_id in dependencies:
                conversation.add_conversation_edge(source_id=dep_id, target_id=prompt_node_id, edge_type="depends_on")

        try:
            # Check if dependencies are complete
            await self._wait_for_dependencies(conversation, prompt_node_id)

            # Update status
            node = conversation.nodes[prompt_node_id]
            node.metadata["execution"]["status"] = "running"
            await self._notify_node_updated(conversation.id, prompt_node_id)

            # Execute prompt
            response = await self.agent_pool.execute_prompt(prompt)

            # Add response node
            response_node_id = conversation.add_conversation_node(
                content=response,
                node_type="response",
                metadata={
                    "execution": ExecutionMetadata(
                        status="completed", started_at=datetime.now(), completed_at=datetime.now()
                    ).model_dump()
                },
            )
            await self._notify_node_added(conversation.id, response_node_id)

            # Link response to prompt
            conversation.add_conversation_edge(
                source_id=prompt_node_id, target_id=response_node_id, edge_type="response_to"
            )

            # Update prompt node status
            node.metadata["execution"]["status"] = "completed"
            node.metadata["execution"]["completed_at"] = datetime.now()
            await self._notify_node_updated(conversation.id, prompt_node_id)
            await self._notify_conversation_updated(conversation)

            return response_node_id

        except Exception as e:
            # Update status on error
            node = conversation.nodes[prompt_node_id]
            node.metadata["execution"]["status"] = "failed"
            node.metadata["execution"]["error"] = str(e)
            await self._notify_node_updated(conversation.id, prompt_node_id)
            await self._notify_conversation_updated(conversation)
            raise

    async def execute_parallel_in_active(
        self, prompts: List[BasePrompt], max_parallel: Optional[int] = None
    ) -> List[str]:
        """Execute multiple prompts in parallel in active conversation."""
        conversation = self._get_conversation()
        return await self.execute_parallel(conversation.id, prompts, max_parallel)

    async def execute_parallel(
        self,
        conversation: Union[str, ConversationGraph, ConversationNode],
        prompts: List[BasePrompt],
        max_parallel: Optional[int] = None,
    ) -> List[str]:
        """Execute multiple prompts in parallel."""
        conversation = self._get_conversation(conversation)
        group_id = str(uuid4())

        # Create semaphore for parallel execution control
        max_parallel = max_parallel or len(prompts)

        # Execute prompts in parallel with controlled concurrency
        async with asyncio.Semaphore(max_parallel):
            tasks = [
                self.execute_prompt(conversation=conversation, prompt=prompt, parallel_group=group_id)
                for prompt in prompts
            ]
            response_ids = await asyncio.gather(*tasks)

        return response_ids

    async def _wait_for_dependencies(self, conversation: ConversationGraph, node_id: str) -> None:
        """Wait for all dependencies of a node to complete."""
        node = conversation.nodes[node_id]
        dependencies = node.metadata["execution"]["dependencies"]

        if not dependencies:
            return

        # Wait for all dependencies to complete
        for dep_id in dependencies:
            dep_node = conversation.nodes[dep_id]
            while dep_node.metadata["execution"]["status"] not in ["completed", "failed"]:
                if dep_node.metadata["execution"]["status"] == "failed":
                    raise RuntimeError(f"Dependency {dep_id} failed: {dep_node.metadata['execution']['error']}")
                await asyncio.sleep(0.1)

    def get_execution_status(
        self, node_id: str, conversation: Optional[Union[str, ConversationGraph, ConversationNode]] = None
    ) -> ExecutionMetadata:
        """Get the execution status of a node."""
        conversation = self._get_conversation(conversation)
        node = conversation.nodes.get(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")
        return ExecutionMetadata.model_validate(node.metadata["execution"])

    def get_conversation_history(
        self,
        node_id: Optional[str] = None,
        max_depth: Optional[int] = None,
        conversation: Optional[Union[str, ConversationGraph, ConversationNode]] = None,
    ) -> List[ConversationNode]:
        """Get conversation history leading to a specific node."""
        conversation = self._get_conversation(conversation)
        if node_id:
            return conversation.get_node_history(node_id, max_depth)
        return list(conversation.nodes.values())

    def get_parallel_group_status(
        self, group_id: str, conversation: Optional[Union[str, ConversationGraph, ConversationNode]] = None
    ) -> Dict[str, ExecutionMetadata]:
        """Get the status of all nodes in a parallel group."""
        conversation = self._get_conversation(conversation)
        return {
            node_id: ExecutionMetadata.model_validate(node.metadata["execution"])
            for node_id, node in conversation.nodes.items()
            if node.metadata.get("execution", {}).get("parallel_group") == group_id
        }

    def add_node_to_conversation(
        self,
        content: Union[BasePrompt, LLMResponse],
        node_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        conversation: Optional[Union[str, ConversationGraph, ConversationNode]] = None,
    ) -> str:
        """Add a node to a conversation."""
        conversation = self._get_conversation(conversation)
        node_id = conversation.add_conversation_node(content=content, node_type=node_type, metadata=metadata or {})

        if parent_id:
            conversation.add_conversation_edge(source_id=parent_id, target_id=node_id, edge_type="response_to")

        return node_id
