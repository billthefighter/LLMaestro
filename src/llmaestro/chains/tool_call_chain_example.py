"""Tool call chain example for LLM orchestration.

This module implements a chain that handles tool calls with retry logic and validation,
following this general pattern:

1. Takes an initial prompt with a tool call
2. Loop until N repeats is reached OR a successful tool execution occurs:
   - If the LLM opts to use a tool, run the tool code and return the result to the LLM
   - If there is an error in the tool call, pass the error result back to the LLM and let it try again
3. Loop until N repeats is reached OR a successful tool execution occurs:
   - If the LLM requests another tool call, run the tool code and return the result to the LLM
   - If there is an error in the tool call, pass the error result back to the LLM and let it try again
4. If the LLM is ready to proceed with answering the initial question:
   - If the LLM returns a result that matches our initial result expectations, pass it to the user
   - If the result does not match, respond with a retry and an explanation of the validation error
   - Retry N times and then return an error if continuing to fail validation
"""

from typing import Any, Dict, Optional, Callable
from uuid import uuid4

from llmaestro.agents.agent_pool import AgentPool
from llmaestro.chains.chains import (
    ChainEdge,
    ChainMetadata,
    ConditionalNode,
    RetryStrategy,
    ValidationNode,
)
from llmaestro.chains.conversation_chain import ConversationChain
from llmaestro.core.orchestrator import Orchestrator
from llmaestro.core.models import LLMResponse
from llmaestro.llm.responses import ResponseFormat
from llmaestro.prompts.base import BasePrompt


class ToolCallChain(ConversationChain):
    """A chain that handles tool calls with retry logic and validation."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        max_tool_call_attempts: int = 3,
        max_validation_attempts: int = 3,
    ):
        """Initialize the tool call chain.

        Args:
            orchestrator: The orchestrator to use for execution
            max_tool_call_attempts: Maximum number of attempts for tool calls
            max_validation_attempts: Maximum number of attempts for validation
        """
        super().__init__(orchestrator)
        self.max_tool_call_attempts = max_tool_call_attempts
        self.max_validation_attempts = max_validation_attempts
        self.tool_call_count = 0
        self.validation_attempt_count = 0

    async def create_tool_call_chain(
        self,
        name: str,
        initial_prompt: BasePrompt,
        expected_response_format: ResponseFormat,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolCallChain":
        """Create a new tool call chain.

        Args:
            name: Name of the conversation
            initial_prompt: The initial prompt that may trigger tool calls
            expected_response_format: The expected format for the final response
            metadata: Optional metadata for the conversation

        Returns:
            The configured tool call chain
        """
        # Initialize the conversation
        await self.initialize(name, initial_prompt, metadata)

        # Create the initial prompt node
        initial_node_id = await self.add_prompt_node(
            prompt=initial_prompt, metadata={"description": "Initial prompt that may trigger tool calls"}
        )

        # Create a conditional node to check if the initial response contains tool calls
        has_tool_calls_condition = self._create_has_tool_calls_condition()
        tool_call_check_node = ConditionalNode(
            id=str(uuid4()),
            conditions={
                "has_tool_calls": has_tool_calls_condition,
                "no_tool_calls": lambda result: not has_tool_calls_condition(result),
            },
            metadata=ChainMetadata(description="Check if response contains tool calls"),
        )
        tool_check_id = self.add_node(tool_call_check_node)

        # Connect initial node to tool check
        self.add_edge(
            ChainEdge(
                source_id=initial_node_id,
                target_id=tool_check_id,
                edge_type="depends_on",
            )
        )

        # Create a validation node for the final response
        validation_node = ValidationNode(
            response_format=expected_response_format,
            retry_strategy=RetryStrategy(max_retries=self.max_validation_attempts),
        )
        validation_id = self.add_node(validation_node)

        # Create a node for handling tool execution results
        tool_execution_node_id = await self.add_prompt_node(
            prompt=BasePrompt(
                name="tool_execution_result",
                description="Handles tool execution results",
                system_prompt="You are processing the results of a tool execution.",
                user_prompt=(
                    "The tool has been executed with the following result:\n\n"
                    "{tool_result}\n\n"
                    "Please process this result and either request another tool call "
                    "or provide a final response that addresses the original question."
                ),
            ),
            metadata={"description": "Processes tool execution results"},
        )

        # Connect the conditional paths
        # Path 1: Has tool calls -> Tool execution -> Tool check (loop back)
        self.add_edge(
            ChainEdge(
                id="has_tool_calls",
                source_id=tool_check_id,
                target_id=tool_execution_node_id,
                edge_type="conditional",
            )
        )

        self.add_edge(
            ChainEdge(
                source_id=tool_execution_node_id,
                target_id=tool_check_id,
                edge_type="depends_on",
            )
        )

        # Path 2: No tool calls -> Validation
        self.add_edge(
            ChainEdge(
                id="no_tool_calls",
                source_id=tool_check_id,
                target_id=validation_id,
                edge_type="conditional",
            )
        )

        return self

    def _create_has_tool_calls_condition(self) -> Callable[[Any], bool]:
        """Create a condition function that checks if a response has tool calls."""

        def has_tool_calls(result: Any) -> bool:
            if isinstance(result, LLMResponse):
                tool_calls = result.metadata.get("tool_calls", [])
                return len(tool_calls) > 0
            return False

        return has_tool_calls

    async def execute_with_tool_handling(self) -> Dict[str, Any]:
        """Execute the chain with tool call handling.

        Returns:
            The results of the chain execution
        """
        if not self.conversation:
            raise ValueError("Chain not initialized with conversation")

        # Execute the chain
        results = await self.execute()

        # Get the final validation result
        validation_nodes = [node_id for node_id, node in self.nodes.items() if isinstance(node, ValidationNode)]

        if not validation_nodes:
            raise ValueError("No validation node found in the chain")

        validation_id = validation_nodes[0]
        validation_status = self.get_node_status(validation_id)

        return {
            "results": results,
            "validation_status": validation_status,
            "conversation_id": self.conversation.id,
        }


async def create_tool_call_example(
    orchestrator: Orchestrator,
    agent_pool: AgentPool,
    initial_prompt: BasePrompt,
    expected_response_format: ResponseFormat,
) -> Dict[str, Any]:
    """Create and execute a tool call chain example.

    Args:
        orchestrator: The orchestrator to use for execution
        agent_pool: The agent pool to use for execution
        initial_prompt: The initial prompt that may trigger tool calls
        expected_response_format: The expected format for the final response

    Returns:
        The results of the chain execution
    """
    # Create the chain
    chain = ToolCallChain(orchestrator)
    chain.agent_pool = agent_pool

    # Configure the chain
    await chain.create_tool_call_chain(
        name="Tool Call Example",
        initial_prompt=initial_prompt,
        expected_response_format=expected_response_format,
    )

    # Execute the chain
    results = await chain.execute_with_tool_handling()

    return results


# Example usage
# Usage example:
# from llmaestro.core.orchestrator import Orchestrator
# from llmaestro.agents.agent_pool import AgentPool
# from llmaestro.prompts.base import BasePrompt
# from llmaestro.llm.responses import ResponseFormat, ResponseFormatType
#
# # Create orchestrator and agent pool
