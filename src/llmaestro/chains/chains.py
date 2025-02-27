"""Graph-based chain system for LLM orchestration."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Set, TypeVar, cast
from uuid import uuid4

from llmaestro.agents.agent_pool import AgentPool
from llmaestro.core.models import LLMResponse
from llmaestro.prompts.base import BasePrompt
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")
ChainResult = TypeVar("ChainResult")


class NodeType(str, Enum):
    """Types of nodes in the chain graph."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    AGENT = "agent"


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
                current_version=self.prompt.current_version,
            )

        # Execute prompt
        result = await agent_pool.execute_prompt(self.prompt)

        # Transform output if needed
        if self.output_transform:
            return self.output_transform(result)
        return cast(T, result)


class ChainNode(BaseModel):
    """Represents a single node in the chain graph."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    step: ChainStep = Field(...)
    node_type: NodeType = Field(...)
    metadata: ChainMetadata = Field(default_factory=ChainMetadata)

    model_config = ConfigDict(validate_assignment=True)


class ChainEdge(BaseModel):
    """Represents a directed edge between chain nodes."""

    source_id: str = Field(..., description="ID of the source node")
    target_id: str = Field(..., description="ID of the target node")
    edge_type: str = Field(..., description="Type of relationship")
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


class ChainGraph(BaseModel):
    """A graph-based representation of an LLM chain."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    nodes: Dict[str, ChainNode] = Field(default_factory=dict)
    edges: List[ChainEdge] = Field(default_factory=list)
    context: ChainContext = Field(default_factory=ChainContext)
    agent_pool: Optional[AgentPool] = None

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    def add_node(self, node: ChainNode) -> str:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        return node.id

    def add_edge(self, edge: ChainEdge) -> None:
        """Add an edge to the graph."""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError("Both source and target nodes must exist in the graph")
        self.edges.append(edge)

    def get_node_dependencies(self, node_id: str) -> List[str]:
        """Get IDs of nodes that must complete before this node."""
        return [edge.source_id for edge in self.edges if edge.target_id == node_id]

    def get_execution_order(self) -> List[List[str]]:
        """Get nodes grouped by execution level (for parallel execution)."""
        # Initialize
        in_degree = {node_id: 0 for node_id in self.nodes}
        for edge in self.edges:
            in_degree[edge.target_id] += 1

        # Group nodes by level
        levels: List[List[str]] = []
        while in_degree:
            # Get all nodes with no dependencies
            current_level = [node_id for node_id, degree in in_degree.items() if degree == 0]
            if not current_level:
                raise ValueError("Cycle detected in chain graph")

            levels.append(current_level)

            # Remove processed nodes and update dependencies
            for node_id in current_level:
                del in_degree[node_id]
                for edge in self.edges:
                    if edge.source_id == node_id and edge.target_id in in_degree:
                        in_degree[edge.target_id] -= 1

        return levels

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
                # Prepare node execution
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
