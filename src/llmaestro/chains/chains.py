"""Graph-based chain system for LLM orchestration."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, Protocol, Set, Tuple, TypeVar, cast
from uuid import uuid4

from llmaestro.agents.agent_pool import AgentPool
from llmaestro.core.graph import BaseEdge, BaseGraph, BaseNode
from llmaestro.core.models import LLMResponse
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.types import PromptMetadata
from llmaestro.llm.responses import ResponseFormat
from pydantic import BaseModel, ConfigDict, Field
from llmaestro.llm.responses import ValidationResult

T = TypeVar("T")
ChainResult = TypeVar("ChainResult")


class NodeType(str, Enum):
    """Types of nodes in the chain graph."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    AGENT = "agent"
    VALIDATION = "validation"  # New type for validation nodes


class AgentType(str, Enum):
    """Types of agents that can execute nodes."""

    GENERAL = "general"
    FAST = "fast"
    SPECIALIST = "specialist"


class RetryStrategy(BaseModel):
    """Configuration for retry behavior."""

    max_retries: int = Field(default=3, ge=0)
    delay: float = Field(default=1.0, ge=0)
    exponential_backoff: bool = Field(default=False)
    max_delay: Optional[float] = Field(default=None, ge=0)

    model_config = ConfigDict(validate_assignment=True)


class ChainMetadata(BaseModel):
    """Structured metadata for chain components."""

    description: Optional[str] = None
    tags: Set[str] = Field(default_factory=set)
    version: Optional[str] = None
    created_at: Optional[str] = None
    custom_data: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)


class ChainState(BaseModel):
    """State management for chain execution."""

    status: str = Field(default="pending")
    current_step: Optional[str] = None
    completed_steps: Set[str] = Field(default_factory=set)
    failed_steps: Set[str] = Field(default_factory=set)
    step_results: Dict[str, Any] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)


@dataclass
class ChainContext:
    """Context object passed between chain steps."""

    metadata: ChainMetadata = field(default_factory=ChainMetadata)
    state: ChainState = field(default_factory=ChainState)
    variables: Dict[str, Any] = field(default_factory=dict)


class InputTransform(Protocol):
    def __call__(self, context: ChainContext, **kwargs: Any) -> Dict[str, Any]:
        ...


class OutputTransform(Protocol, Generic[T]):
    def __call__(self, response: LLMResponse) -> T:
        ...


class ChainStep(BaseModel, Generic[T]):
    """Represents a single step in a chain."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    prompt: BasePrompt
    input_transform: Optional[Callable[[ChainContext, Any], Dict[str, Any]]] = None
    output_transform: Optional[Callable[[LLMResponse], T]] = None
    retry_strategy: RetryStrategy = Field(default_factory=RetryStrategy)

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    @classmethod
    async def create(
        cls,
        prompt: BasePrompt,
        input_transform: Optional[Callable[[ChainContext, Any], Dict[str, Any]]] = None,
        output_transform: Optional[Callable[[LLMResponse], T]] = None,
        retry_strategy: Optional[RetryStrategy] = None,
    ) -> "ChainStep[T]":
        """Create a new chain step."""
        return cls(
            prompt=prompt,
            input_transform=input_transform,
            output_transform=output_transform,
            retry_strategy=retry_strategy or RetryStrategy(),
        )

    async def execute(
        self,
        agent_pool: AgentPool,
        context: ChainContext,
        **kwargs: Any,
    ) -> T:
        """Execute this chain step using the agent pool."""
        # Transform input using context if needed
        if self.input_transform:
            transformed_data = self.input_transform(context, kwargs)
            # Update prompt with transformed data
            self.prompt = BasePrompt(
                name=self.prompt.name,
                description=self.prompt.description,
                system_prompt=self.prompt.system_prompt.format(**transformed_data),
                user_prompt=self.prompt.user_prompt.format(**transformed_data),
                metadata=self.prompt.metadata,
                variables=self.prompt.variables,
            )

        # Execute prompt
        result = await agent_pool.execute_prompt(self.prompt)

        # Transform output if needed
        if self.output_transform:
            return self.output_transform(result)
        return cast(T, result)


class ChainNode(BaseNode):
    """Represents a single node in the chain graph."""

    step: ChainStep = Field(...)
    node_type: NodeType = Field(...)
    metadata: ChainMetadata = Field(default_factory=ChainMetadata)

    model_config = ConfigDict(validate_assignment=True)


class ChainEdge(BaseEdge):
    """Represents a directed edge between chain nodes."""

    condition: Optional[str] = Field(default=None, description="Optional condition for edge traversal")

    model_config = ConfigDict(validate_assignment=True)


class ChainExecutor:
    """Handles execution of chain nodes with retry logic."""

    @staticmethod
    async def execute_with_retry(
        node: ChainNode, agent_pool: AgentPool, context: ChainContext, retry_strategy: RetryStrategy, **kwargs: Any
    ) -> Any:
        """Execute a node with retry logic."""
        max_retries = retry_strategy.max_retries
        delay = retry_strategy.delay

        for attempt in range(max_retries):
            try:
                return await node.step.execute(agent_pool, context, **kwargs)
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(delay)
        raise RuntimeError("Node execution failed after all retries")


class ValidationNode(ChainNode):
    """A specialized node for response validation and retry logic."""

    def __init__(
        self,
        response_format: ResponseFormat,
        retry_strategy: Optional[RetryStrategy] = None,
        error_handler: Optional[Callable[[ValidationResult], Dict[str, Any]]] = None,
    ):
        super().__init__(
            node_type=NodeType.VALIDATION,
            step=ChainStep(
                prompt=self._create_retry_prompt(),
                retry_strategy=retry_strategy or RetryStrategy(),
            ),
        )
        self.response_format = response_format
        self.error_handler = error_handler

    async def validate_and_retry(
        self,
        response: LLMResponse,
        agent_pool: AgentPool,
        context: ChainContext,
    ) -> Tuple[bool, Any]:
        """Validate response and handle retries if needed.

        Args:
            response: The LLM response to validate
            agent_pool: Pool of agents for retry execution
            context: Current chain context

        Returns:
            Tuple of (is_valid, final_result)
        """
        validation_result = self.response_format.validate_response(response.content)

        if validation_result.is_valid:
            return True, validation_result.formatted_response

        # Handle retry if needed
        retry_prompt = self.response_format.generate_retry_prompt(validation_result)
        if not retry_prompt:
            if self.error_handler:
                return False, self.error_handler(validation_result)
            return False, validation_result

        # Update prompt with retry context
        self.step.prompt.user_prompt = retry_prompt

        # Execute retry
        retry_response = await self.step.execute(agent_pool, context)
        validation_result.retry_count += 1

        # Validate retry response
        return await self.validate_and_retry(retry_response, agent_pool, context)

    def _create_retry_prompt(self) -> BasePrompt:
        """Create a prompt for retry attempts."""
        return BasePrompt(
            name="validation_retry",
            description="Retry prompt for invalid responses",
            system_prompt="You are helping to fix an invalid response. Please address the validation errors and provide a corrected response.",
            user_prompt="{retry_message}",
            metadata=PromptMetadata(type="validation", expected_response=self.response_format),
            variables=[],  # No version control for retry prompts
        )


class ChainGraph(BaseGraph[ChainNode, ChainEdge]):
    """A graph-based representation of an LLM chain."""

    context: ChainContext = Field(default_factory=ChainContext)
    agent_pool: Optional[AgentPool] = None

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the chain graph."""
        if not self.agent_pool:
            raise ValueError("AgentPool must be set before execution")

        results: Dict[str, Any] = {}
        execution_order = self.get_execution_order()

        # Execute nodes level by level
        for level in execution_order:
            # Create tasks for all nodes in current level
            tasks = []
            for node_id in level:
                node = self.nodes[node_id]
                # Get results from dependencies
                dep_results = {dep_id: results[dep_id] for dep_id in self.get_node_dependencies(node_id)}

                # Special handling for validation nodes
                if isinstance(node, ValidationNode):
                    # Find the response to validate from dependencies
                    response_to_validate = next(
                        (result for result in dep_results.values() if isinstance(result, LLMResponse)), None
                    )
                    if response_to_validate:
                        is_valid, validated_result = await node.validate_and_retry(
                            response_to_validate, self.agent_pool, self.context
                        )
                        results[node_id] = validated_result
                        continue

                # Regular node execution
                task = ChainExecutor.execute_with_retry(
                    node=node,
                    agent_pool=self.agent_pool,
                    context=self.context,
                    retry_strategy=node.step.retry_strategy,
                    dependency_results=dep_results,
                    **kwargs,
                )
                tasks.append((node_id, task))

            # Execute current level
            level_results = await asyncio.gather(*(task for _, task in tasks))

            # Store results
            for (node_id, _), result in zip(tasks, level_results, strict=False):
                results[node_id] = result

        return results
