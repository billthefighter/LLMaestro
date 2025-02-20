"""Orchestration layer for managing LLM conversations and execution."""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from llmaestro.core.conversations import ConversationGraph
from llmaestro.prompts.base import BasePrompt

if TYPE_CHECKING:
    from llmaestro.agents.agent_pool import AgentPool


class ExecutionMetadata(BaseModel):
    """Metadata for tracking execution status and resources."""

    status: str = Field("pending", description="Current execution status")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    resources: Dict[str, Any] = Field(default_factory=dict)
    parallel_group: Optional[str] = None  # For grouping parallel executions
    dependencies: List[str] = Field(default_factory=list)  # Node IDs this execution depends on

    model_config = ConfigDict(validate_assignment=True)


class Orchestrator:
    """Manages LLM conversation execution and resource coordination."""

    def __init__(self, agent_pool: "AgentPool"):
        self.agent_pool = agent_pool
        self.active_conversations: Dict[str, ConversationGraph] = {}

    async def create_conversation(
        self, name: str, initial_prompt: BasePrompt, metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationGraph:
        """Create a new conversation with an initial prompt."""
        conversation = ConversationGraph(id=str(uuid4()), metadata={"name": name, **(metadata or {})})

        # Add initial prompt node
        conversation.add_node(
            content=initial_prompt, node_type="prompt", metadata={"execution": ExecutionMetadata().model_dump()}
        )

        self.active_conversations[conversation.id] = conversation
        return conversation

    async def execute_prompt(
        self,
        conversation_id: str,
        prompt: BasePrompt,
        dependencies: Optional[List[str]] = None,
        parallel_group: Optional[str] = None,
    ) -> str:
        """Execute a prompt and add it to the conversation."""
        conversation = self.active_conversations.get(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Create execution metadata
        exec_metadata = ExecutionMetadata(
            status="pending", started_at=datetime.now(), parallel_group=parallel_group, dependencies=dependencies or []
        )

        # Add prompt node
        prompt_node_id = conversation.add_node(
            content=prompt, node_type="prompt", metadata={"execution": exec_metadata.model_dump()}
        )

        # Add dependency edges
        if dependencies:
            for dep_id in dependencies:
                conversation.add_edge(source_id=dep_id, target_id=prompt_node_id, edge_type="depends_on")

        try:
            # Check if dependencies are complete
            await self._wait_for_dependencies(conversation, prompt_node_id)

            # Update status
            node = conversation.nodes[prompt_node_id]
            node.metadata["execution"]["status"] = "running"

            # Execute prompt
            response = await self.agent_pool.execute_prompt(prompt)

            # Add response node
            response_node_id = conversation.add_node(
                content=response,
                node_type="response",
                metadata={
                    "execution": ExecutionMetadata(
                        status="completed", started_at=datetime.now(), completed_at=datetime.now()
                    ).model_dump()
                },
            )

            # Link response to prompt
            conversation.add_edge(source_id=prompt_node_id, target_id=response_node_id, edge_type="response_to")

            # Update prompt node status
            node.metadata["execution"]["status"] = "completed"
            node.metadata["execution"]["completed_at"] = datetime.now()

            return response_node_id

        except Exception as e:
            # Update status on error
            node = conversation.nodes[prompt_node_id]
            node.metadata["execution"]["status"] = "failed"
            node.metadata["execution"]["error"] = str(e)
            raise

    async def execute_parallel(
        self, conversation_id: str, prompts: List[BasePrompt], max_parallel: Optional[int] = None
    ) -> List[str]:
        """Execute multiple prompts in parallel."""
        conversation = self.active_conversations.get(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        group_id = str(uuid4())
        response_ids = []

        # Create semaphore for parallel execution control
        max_parallel = max_parallel or len(prompts)

        # Execute prompts in parallel with controlled concurrency
        async with asyncio.Semaphore(max_parallel):
            tasks = [
                self.execute_prompt(conversation_id=conversation_id, prompt=prompt, parallel_group=group_id)
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

    def get_execution_status(self, conversation_id: str, node_id: str) -> ExecutionMetadata:
        """Get the execution status of a node."""
        conversation = self.active_conversations.get(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        node = conversation.nodes.get(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")

        return ExecutionMetadata.model_validate(node.metadata["execution"])

    def get_parallel_group_status(self, conversation_id: str, group_id: str) -> Dict[str, ExecutionMetadata]:
        """Get the status of all nodes in a parallel group."""
        conversation = self.active_conversations.get(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        group_nodes = {
            node_id: ExecutionMetadata.model_validate(node.metadata["execution"])
            for node_id, node in conversation.nodes.items()
            if node.metadata.get("execution", {}).get("parallel_group") == group_id
        }

        return group_nodes
