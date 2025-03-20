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

from typing import Any, Dict, Optional, Callable, Type, TypeVar, Generic
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

P = TypeVar("P", bound=BasePrompt)


class ToolCallChain(ConversationChain[P], Generic[P]):
    """A chain that handles tool calls with retry logic and validation.

    Generic Parameters:
        P: The type of prompt to use in this chain (must inherit from BasePrompt)
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        max_tool_call_attempts: int = 3,
        max_validation_attempts: int = 3,
        prompt_type: Optional[Type[P]] = None,
    ):
        """Initialize the tool call chain.

        Args:
            orchestrator: The orchestrator to use for execution
            max_tool_call_attempts: Maximum number of attempts for tool calls
            max_validation_attempts: Maximum number of attempts for validation
            prompt_type: The type of prompt to use (must inherit from BasePrompt)
        """
        # Initialize the parent class with the prompt_type
        super().__init__(orchestrator, prompt_type=prompt_type)
        self.max_tool_call_attempts = max_tool_call_attempts
        self.max_validation_attempts = max_validation_attempts
        self.tool_call_count = 0
        self.validation_attempt_count = 0

    async def create_tool_call_chain(
        self,
        name: str,
        initial_prompt: P,
        expected_response_format: ResponseFormat,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolCallChain[P]":
        """Create a new tool call chain.

        Args:
            name: Name of the conversation
            initial_prompt: The initial prompt that may trigger tool calls (must be of the same type as this chain's
            prompt_type)
            expected_response_format: The expected format for the final response
            metadata: Optional metadata for the conversation

        Returns:
            The configured tool call chain with the same prompt type as this instance
        """
        # Initialize the conversation
        await self.initialize(name, initial_prompt, metadata)

        # Create the initial prompt node
        initial_node_id = await self.add_prompt_node(
            prompt=initial_prompt, metadata={"description": "Initial prompt that may trigger tool calls"}
        )
        initial_node = self.nodes[initial_node_id]

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
        tool_check_node = self.nodes[tool_check_id]

        # Connect initial node to tool check
        self.add_edge(
            ChainEdge(
                source=initial_node,
                target=tool_check_node,
                edge_type="depends_on",
            )
        )

        # Create a validation node for the final response using the same prompt type
        validation_node = ValidationNode(
            response_format=expected_response_format,
            retry_strategy=RetryStrategy(max_retries=self.max_validation_attempts),
            prompt_type=self.prompt_type,
        )
        validation_id = self.add_node(validation_node)
        validation_node_obj = self.nodes[validation_id]

        # Create a node for handling tool execution results using the graph's create_prompt method
        tool_execution_prompt = self.create_prompt(
            name="tool_execution_result",
            description="Handles tool execution results",
            system_prompt="You are processing the results of a tool execution.",
            user_prompt=(
                "The tool has been executed with the following result:\n\n"
                "{tool_result}\n\n"
                "Please process this result and either request another tool call "
                "or provide a final response that addresses the original question."
            ),
        )

        tool_execution_node_id = await self.add_prompt_node(
            prompt=tool_execution_prompt,
            metadata={"description": "Processes tool execution results"},
        )
        tool_execution_node = self.nodes[tool_execution_node_id]

        # Connect the conditional paths
        # Path 1: Has tool calls -> Tool execution -> Tool check (loop back)
        self.add_edge(
            ChainEdge(
                id="has_tool_calls",
                source=tool_check_node,
                target=tool_execution_node,
                edge_type="conditional",
                condition="has_tool_calls",
            )
        )

        self.add_edge(
            ChainEdge(
                source=tool_execution_node,
                target=tool_check_node,
                edge_type="depends_on",
            )
        )

        # Path 2: No tool calls -> Validation
        self.add_edge(
            ChainEdge(
                id="no_tool_calls",
                source=tool_check_node,
                target=validation_node_obj,
                edge_type="conditional",
                condition="no_tool_calls",
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
    initial_prompt: P,
    expected_response_format: ResponseFormat,
    prompt_type: Optional[Type[P]] = None,
) -> Dict[str, Any]:
    """Create and execute a tool call chain example.

    Args:
        orchestrator: The orchestrator to use for execution
        agent_pool: The agent pool to use for execution
        initial_prompt: The initial prompt that may trigger tool calls (must match the prompt_type)
        expected_response_format: The expected format for the final response
        prompt_type: The type of prompt to use (must inherit from BasePrompt)

    Returns:
        The results of the chain execution
    """
    # Create the chain with the specified prompt type
    chain = ToolCallChain[P](orchestrator, prompt_type=prompt_type)
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
# from llmaestro.prompts.memory import SimplePrompt
# from llmaestro.llm.responses import ResponseFormat, ResponseFormatType
#
# # Create orchestrator and agent pool
#
# # Example of using a custom prompt type:
# class MyCustomPrompt(BasePrompt):
#     # Additional properties or methods specific to your prompt type
#     custom_property: str = "default"
#
#     async def save(self) -> bool:
#         return True
#
#     @classmethod
#     async def load(cls, identifier: str) -> Optional["MyCustomPrompt"]:
#         return None
#
# # Create a chain with the custom prompt type explicitly using generics
# # This provides full type safety throughout the chain
# chain = ToolCallChain[MyCustomPrompt](orchestrator, prompt_type=MyCustomPrompt)
#
# # Another way to create the chain with the same prompt type
# chain2 = ToolCallChain(orchestrator, prompt_type=MyCustomPrompt)  # Type will be inferred
#
# # Using SimplePrompt as the default type
# chain3 = ToolCallChain[SimplePrompt](orchestrator)  # Will use SimplePrompt for create_prompt calls
