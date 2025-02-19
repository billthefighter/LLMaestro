"""Graph-based chain system for LLM orchestration."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, TypeVar, cast
from uuid import uuid4

from llmaestro.agents.agent_pool import AgentPool
from llmaestro.agents.models.models import AgentCapability
from llmaestro.core.models import Artifact, ArtifactStorage, SubTask, Task
from llmaestro.core.task_manager import DecompositionConfig, TaskManager
from llmaestro.llm.interfaces import BaseLLMInterface, LLMResponse
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.loader import PromptLoader
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


class InputTransform(Protocol):
    def __call__(self, context: "ChainContext", **kwargs: Any) -> Dict[str, Any]:
        ...


class OutputTransform(Protocol):
    def __call__(self, response: LLMResponse) -> T:
        ...


@dataclass
class ChainContext:
    """Context object passed between chain steps."""

    artifacts: Dict[str, Artifact] = field(default_factory=dict)
    metadata: ChainMetadata = field(default_factory=ChainMetadata)
    state: ChainState = field(default_factory=ChainState)


class ChainStep(BaseModel):
    """Represents a single step in a prompt chain."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    task_type: str
    prompt: BasePrompt
    input_transform: Optional[InputTransform] = None
    output_transform: Optional[OutputTransform] = None
    retry_strategy: RetryStrategy = Field(default_factory=RetryStrategy)

    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    async def create(
        cls,
        task_type: str,
        prompt_loader: PromptLoader,
        input_transform: Optional[InputTransform] = None,
        output_transform: Optional[OutputTransform] = None,
        retry_strategy: Optional[RetryStrategy] = None,
    ) -> "ChainStep":
        """Create a new chain step with loaded prompt."""
        prompt = await prompt_loader.load_prompt("file", f"prompts/{task_type}.yaml")
        if not prompt:
            raise ValueError(f"Could not load prompt for task type: {task_type}")

        return cls(
            task_type=task_type,
            prompt=prompt,
            input_transform=input_transform,
            output_transform=output_transform,
            retry_strategy=retry_strategy or RetryStrategy(),
        )

    async def execute(
        self,
        llm: BaseLLMInterface,
        context: ChainContext,
        **kwargs: Any,
    ) -> T:
        """Execute this chain step."""
        # Transform input using context if needed
        input_data = kwargs
        if self.input_transform:
            input_data = self.input_transform(context, **kwargs)

        # Process with LLM
        response = await llm.process(self.prompt, variables=input_data)

        # Transform output if needed
        if self.output_transform:
            return self.output_transform(response)
        return cast(T, response)


class ChainNode(BaseModel):
    """Represents a single node in the chain graph."""

    id: str = Field(..., description="Unique identifier for this node")
    step: ChainStep = Field(..., description="The chain step to execute")
    node_type: NodeType = Field(..., description="Type of node (sequential/parallel/etc)")
    metadata: ChainMetadata = Field(default_factory=ChainMetadata)

    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    async def create(
        cls,
        task_type: str,
        prompt_loader: PromptLoader,
        node_type: NodeType,
        input_transform: Optional[InputTransform] = None,
        output_transform: Optional[OutputTransform] = None,
        retry_strategy: Optional[RetryStrategy] = None,
        metadata: Optional[ChainMetadata] = None,
    ) -> "ChainNode":
        """Create a new chain node with loaded prompt."""
        step = await ChainStep.create(
            task_type=task_type,
            prompt_loader=prompt_loader,
            input_transform=input_transform,
            output_transform=output_transform,
            retry_strategy=retry_strategy,
        )

        return cls(
            id=str(uuid4()),
            step=step,
            node_type=node_type,
            metadata=metadata or ChainMetadata(),
        )


class ChainEdge(BaseModel):
    """Represents a directed edge between chain nodes."""

    source_id: str = Field(..., description="ID of the source node")
    target_id: str = Field(..., description="ID of the target node")
    edge_type: str = Field(..., description="Type of relationship")
    condition: Optional[str] = Field(default=None, description="Optional condition for edge traversal")

    model_config = ConfigDict(validate_assignment=True)


class AgentChainNode(ChainNode):
    """A chain node that is executed by a specific agent."""

    agent_type: Optional[AgentType] = Field(None, description="Type of agent to use")
    required_capabilities: Set[AgentCapability] = Field(default_factory=set, description="Required agent capabilities")

    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    async def create(
        cls,
        task_type: str,
        prompt_loader: PromptLoader,
        node_type: NodeType,
        agent_type: Optional[AgentType] = None,
        input_transform: Optional[InputTransform] = None,
        output_transform: Optional[OutputTransform] = None,
        retry_strategy: Optional[RetryStrategy] = None,
        metadata: Optional[ChainMetadata] = None,
        required_capabilities: Optional[Set[AgentCapability]] = None,
    ) -> "AgentChainNode":
        """Create a new agent chain node with loaded prompt."""
        step = await ChainStep.create(
            task_type=task_type,
            prompt_loader=prompt_loader,
            input_transform=input_transform,
            output_transform=output_transform,
            retry_strategy=retry_strategy,
        )

        return cls(
            id=str(uuid4()),
            step=step,
            node_type=node_type,
            agent_type=agent_type,
            metadata=metadata or ChainMetadata(),
            required_capabilities=required_capabilities or set(),
        )


class ChainExecutor:
    """Handles execution of chain nodes with retry logic."""

    @staticmethod
    async def execute_with_retry(
        node: ChainNode, llm: BaseLLMInterface, context: ChainContext, retry_strategy: RetryStrategy, **kwargs: Any
    ) -> Any:
        """Execute a node with retry logic."""
        max_retries = retry_strategy.max_retries
        delay = retry_strategy.delay

        for attempt in range(max_retries):
            try:
                return await node.step.execute(llm, context, **kwargs)
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(delay)
        raise RuntimeError("Node execution failed after all retries")


class ChainGraph(BaseModel):
    """A graph-based representation of an LLM chain."""

    id: str = Field(..., description="Unique identifier for this chain")
    nodes: Dict[str, ChainNode] = Field(default_factory=dict)
    edges: List[ChainEdge] = Field(default_factory=list)
    context: ChainContext = Field(default_factory=ChainContext)
    storage: Optional[ArtifactStorage] = None
    llm: Optional[BaseLLMInterface] = None
    prompt_loader: Optional[PromptLoader] = None

    model_config = ConfigDict(validate_assignment=True)

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
        if not self.llm or not self.prompt_loader:
            raise ValueError("LLM and PromptLoader must be set before execution")

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
                    llm=self.llm,
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
                if self.storage:
                    await self._store_result(node_id, result)

        return results

    async def _store_result(self, node_id: str, result: Any) -> None:
        """Store a node's result."""
        if not self.storage:
            return

        # Create an artifact from the result
        artifact = Artifact(
            name=f"node_{node_id}_result",
            content_type="application/json",
            data=result,
            metadata={"node_id": node_id, "chain_id": self.id},
        )

        # Store using ArtifactStorage (non-async)
        self.storage.save_artifact(artifact)


class AgentAwareChainGraph(ChainGraph):
    """A chain graph that works with the agent pool."""

    def __init__(self, agent_pool: AgentPool, **kwargs):
        super().__init__(**kwargs)
        self._agent_pool = agent_pool

    async def execute_node(self, node: AgentChainNode, **kwargs) -> Any:
        """Execute a node using an appropriate agent."""
        agent = self._agent_pool.get_agent(node.agent_type)
        # Convert ChainStep to SubTask
        subtask = SubTask(
            id=node.id,
            type=node.step.task_type,
            input_data=kwargs.get("input_data", {}),
            parent_task_id=None,
        )
        return await agent.process_task(subtask)


class TaskAwareChainGraph(AgentAwareChainGraph):
    """A chain graph that integrates with the task manager."""

    def __init__(self, task_manager: TaskManager, prompt_loader: PromptLoader, **kwargs):
        super().__init__(**kwargs)
        self._task_manager = task_manager
        self._prompt_loader = prompt_loader

    async def execute_task(self, task: Task, config: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the chain for a specific task."""
        # Convert config to DecompositionConfig
        decomp_config: DecompositionConfig = (
            {
                "strategy": config.get("strategy", "chunk"),
                "chunk_size": config.get("chunk_size", 1000),
                "max_parallel": config.get("max_parallel", 5),
                "aggregation": config.get("aggregation", "concatenate"),
            }
            if config
            else {"strategy": "chunk", "chunk_size": 1000, "max_parallel": 5, "aggregation": "concatenate"}
        )

        # Decompose task if needed
        subtasks = self._task_manager.decompose_task(task, decomp_config)

        # Create nodes for subtasks
        for subtask in subtasks:
            node = await AgentChainNode.create(
                task_type=subtask.type,
                prompt_loader=self._prompt_loader,
                node_type=NodeType.AGENT,
                agent_type=AgentType.GENERAL,  # Default to general agent
            )
            self.add_node(node)

        # Execute the chain
        execution_kwargs = {"input_data": task.input_data}
        if config:
            execution_kwargs.update(config)
        return await super().execute(**execution_kwargs)
