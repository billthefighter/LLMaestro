"""Common chain recipes and patterns."""

from typing import Any, Dict, Optional, cast

from llmaestro.agents.agent_pool import AgentPool
from llmaestro.chains.chains import (
    ChainContext,
    ChainEdge,
    ChainGraph,
    ChainMetadata,
    ChainNode,
    ChainState,
    ChainStep,
    NodeType,
    RetryStrategy,
    ValidationNode,
)
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.types import PromptMetadata, ResponseFormat, ResponseFormatType, ValidationResult


async def create_validation_chain(
    prompt: BasePrompt,
    response_format: ResponseFormat,
    agent_pool: AgentPool,
    retry_strategy: Optional[RetryStrategy] = None,
    preserve_context: bool = True,
) -> ChainGraph:
    """Create a chain that validates LLM responses and handles retries with context preservation.

    Args:
        prompt: The initial prompt to execute
        response_format: Expected format and schema for validation
        agent_pool: Pool of agents for execution
        retry_strategy: Optional custom retry strategy
        preserve_context: Whether to preserve conversation context during retries

    Returns:
        A configured chain graph ready for execution
    """
    chain = ChainGraph(agent_pool=agent_pool, context=ChainContext(state=ChainState(status="pending", variables={})))

    # Create the initial prompt node
    prompt_node = ChainNode(
        step=ChainStep(prompt=prompt),
        node_type=NodeType.AGENT,
        metadata=ChainMetadata(description="Initial prompt execution", tags={"initial_prompt"}),
    )
    prompt_id = chain.add_node(prompt_node)

    # Create validation node with context preservation
    def context_preserving_error_handler(validation_result: ValidationResult) -> Dict[str, Any]:
        """Handle validation errors while preserving context."""
        # Get current chain state
        state = cast(ChainState, chain.context.state)
        return {
            "status": "error",
            "validation_result": validation_result,
            "context": {
                "status": state.status,
                "current_step": state.current_step,
                "completed_steps": list(state.completed_steps),
                "failed_steps": list(state.failed_steps),
                "step_results": state.step_results,
                "variables": state.variables,
            },
            "retry_count": validation_result.retry_count,
        }

    validation_node = ValidationNode(
        response_format=response_format,
        retry_strategy=retry_strategy or RetryStrategy(max_retries=3),
        error_handler=context_preserving_error_handler if preserve_context else None,
    )
    validation_id = chain.add_node(validation_node)

    # Connect nodes
    chain.add_edge(ChainEdge(source_id=prompt_id, target_id=validation_id, edge_type="validates"))

    return chain


# Example usage:
async def example_json_validation_chain():
    """Example of using validation chain for JSON responses."""
    # Create agent pool
    agent_pool = AgentPool()  # Configure as needed

    # Create prompt expecting JSON response
    prompt = BasePrompt(
        name="user_info",
        description="Get user information in JSON format",
        system_prompt="You are a helpful assistant that provides user information in JSON format.",
        user_prompt="Please provide information about user {name} including their age and location.",
        metadata=PromptMetadata(
            type="user_info",
            expected_response=ResponseFormat(
                format=ResponseFormatType.JSON,
                schema="""
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "location": {"type": "string"}
                    },
                    "required": ["name", "age", "location"]
                }
                """,
            ),
        ),
        variables=[{"name": "name", "description": "User's name"}],
    )

    # Create validation chain
    chain = await create_validation_chain(
        prompt=prompt, response_format=prompt.metadata.expected_response, agent_pool=agent_pool, preserve_context=True
    )

    # Execute chain
    results = await chain.execute(name="Alice")

    # Results will contain either:
    # 1. Valid JSON response
    # 2. Error with context if validation failed after retries
    return results


async def example_structured_conversation_chain():
    """Example of validation chain for multi-turn conversation."""
    agent_pool = AgentPool()  # Configure as needed

    # Create a conversation with structured responses
    conversation_prompt = BasePrompt(
        name="structured_conversation",
        description="Multi-turn conversation with structured responses",
        system_prompt="""You are an assistant that provides structured responses.
Each response should include: thought process, action taken, and next steps.""",
        user_prompt="{user_input}",
        metadata=PromptMetadata(
            type="conversation",
            expected_response=ResponseFormat(
                format=ResponseFormatType.JSON,
                schema="""
                {
                    "type": "object",
                    "properties": {
                        "thought_process": {"type": "string"},
                        "action": {"type": "string"},
                        "next_steps": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["thought_process", "action", "next_steps"]
                }
                """,
            ),
        ),
        variables=[{"name": "user_input", "description": "User's input"}],
    )

    # Custom retry strategy with context preservation
    retry_strategy = RetryStrategy(max_retries=3, delay=1.0, exponential_backoff=True)

    # Create chain
    chain = await create_validation_chain(
        prompt=conversation_prompt,
        response_format=conversation_prompt.metadata.expected_response,
        agent_pool=agent_pool,
        retry_strategy=retry_strategy,
        preserve_context=True,
    )

    # Execute multiple turns
    conversation_history = []

    async def execute_turn(user_input: str) -> Dict[str, Any]:
        results = await chain.execute(user_input=user_input, conversation_history=conversation_history)
        if isinstance(results, dict) and "status" not in results:  # Valid response
            conversation_history.append({"user": user_input, "assistant": results})
        return results

    # Example conversation
    turns = ["Tell me about machine learning", "What are neural networks?", "How do I get started with ML?"]

    results = []
    for turn in turns:
        result = await execute_turn(turn)
        results.append(result)

    return results
