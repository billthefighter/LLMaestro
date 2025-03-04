"""Agent pool for managing multiple LLM agents for prompt processing."""
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Protocol, Set, TypeVar

from llmaestro.agents.models import Agent, AgentMetrics, AgentState
from llmaestro.llm.capabilities import LLMCapabilities
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.models import LLMInstance
from llmaestro.prompts.base import BasePrompt
from llmaestro.core.models import LLMResponse


T = TypeVar("T")


class JsonOutputTransform(Protocol):
    """Protocol for JSON output transformers."""

    def transform(self, response: LLMResponse) -> Dict[str, Any]:
        ...


class RuntimeAgent:
    """Runtime agent that processes prompts using LLM.

    This agent handles the actual execution of prompts through the LLM interface,
    managing state and metrics for each execution.
    """

    def __init__(self, model_name: str, llm_instance: LLMInstance, description: Optional[str] = None):
        """Initialize a runtime agent.

        Args:
            model_name: Name of the model this agent uses
            llm_instance: LLM instance to use for processing
            description: Optional description of the agent's purpose
        """
        self.agent = Agent(
            id=str(uuid.uuid4()),
            type=description or "unknown",
            model=model_name,
            capabilities=llm_instance.state.profile.capabilities,
            metadata={"provider": llm_instance.state.provider.family},
        )
        self.model_name = model_name
        self.llm_instance = llm_instance
        self.active_prompts: Dict[str, asyncio.Task[Any]] = {}

    async def process_prompt(self, prompt: BasePrompt) -> LLMResponse:
        """Process a prompt using this agent's LLM.

        Args:
            prompt: The prompt to process

        Returns:
            The LLM response
        """
        self.agent.update_state(AgentState.BUSY)

        try:
            start_time = asyncio.get_event_loop().time()
            result = await self.llm_instance.interface.process(prompt)
            execution_time = asyncio.get_event_loop().time() - start_time

            # Update agent metrics
            if result.token_usage and result.context_metrics:
                self.agent.update_metrics(
                    AgentMetrics(
                        token_usage=result.token_usage,
                        context_metrics=result.context_metrics,
                        execution_time=execution_time,
                    )
                )

            return result

        except Exception as e:
            self.agent.update_state(AgentState.ERROR)
            raise e

        finally:
            if self.agent.state != AgentState.ERROR:
                self.agent.update_state(AgentState.IDLE)


class AgentPool:
    """Pool of agents that can be reused for prompt processing.

    Manages a collection of runtime agents, handling agent allocation,
    prompt execution, and resource management.
    """

    def __init__(
        self,
        llm_registry: LLMRegistry,
        max_agents: int = 10,
        default_model_name: Optional[str] = None,
    ):
        """Initialize the agent pool.

        Args:
            llm_registry: LLM registry instance for managing models and credentials
            max_agents: Maximum number of concurrent agents
            default_model_name: Optional default model name to use when no specific capabilities are required
        """
        self._llm_registry = llm_registry
        self._max_agents = max_agents
        self._active_agents: Dict[str, RuntimeAgent] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_agents)
        self.prompts: Dict[str, asyncio.Task[Any]] = {}
        self.loop = asyncio.get_event_loop()
        self.default_model_name = default_model_name

    async def get_agent(
        self, required_capabilities: Optional[Set[str]] = None, description: Optional[str] = None
    ) -> RuntimeAgent:
        """Get an agent suitable for prompt processing.

        Args:
            required_capabilities: Optional set of capability flags from LLMCapabilities.VALID_CAPABILITY_FLAGS
                                that the agent must support.
            description: Optional description of the agent's purpose.

        Returns:
            A RuntimeAgent instance capable of processing prompts

        Raises:
            ValueError: If no suitable agent is available or pool is full
        """
        # Get model state from registry
        model_states = self._llm_registry.model_states
        if not model_states:
            raise ValueError("No models registered in LLM registry")

        # Validate capability requirements if provided
        if required_capabilities:
            LLMCapabilities.validate_capability_flags(required_capabilities)

        # Find a suitable model based on capabilities
        model_name = None
        if required_capabilities:
            for name, state in model_states.items():
                caps = state.profile.capabilities
                if all(getattr(caps, cap, False) for cap in required_capabilities):
                    model_name = name
                    break

            if not model_name:
                raise ValueError(f"No models found supporting required capabilities: {required_capabilities}")
        else:
            # If default_model_name is set and available in registry, use it
            if self.default_model_name and self.default_model_name in model_states:
                model_name = self.default_model_name
            else:
                # Otherwise use first available model
                model_name = next(iter(model_states.keys()))

        # Create a new agent if we haven't reached the limit
        if len(self._active_agents) < self._max_agents:
            agent = await self._create_agent(model_name, description)
            self._active_agents[agent.agent.id] = agent
            return agent

        # Otherwise, find the least busy agent with required capabilities
        compatible_agents = [agent for agent in self._active_agents.values() if agent.model_name == model_name]

        if not compatible_agents:
            raise ValueError(f"No agents available supporting capabilities: {required_capabilities}")

        return min(compatible_agents, key=lambda a: len(a.active_prompts))

    async def _create_agent(self, model_name: str, description: Optional[str] = None) -> RuntimeAgent:
        """Create a new agent for prompt processing.

        Args:
            model_name: Name of the model to use
            description: Optional description of the agent's purpose

        Returns:
            A new RuntimeAgent instance
        """
        # Create LLM instance using registry
        llm_instance = await self._llm_registry.create_instance(model_name)
        return RuntimeAgent(model_name=model_name, llm_instance=llm_instance, description=description)

    async def execute_prompt(
        self,
        prompt: BasePrompt,
        agent_type: Optional[str] = None,
        required_capabilities: Optional[Set[str]] = None,
    ) -> LLMResponse:
        """Execute a prompt using an appropriate agent.

        This method handles the complete lifecycle of prompt execution:
        1. Agent selection/creation
        2. Prompt submission
        3. Response retrieval
        4. Resource cleanup

        Args:
            prompt: The prompt to execute
            agent_type: Optional type of agent to use
            required_capabilities: Optional set of required capability flags from LLMCapabilities.
                                 Must be valid flags from LLMCapabilities.VALID_CAPABILITY_FLAGS.

        Returns:
            The LLM response from processing the prompt

        Raises:
            ValueError: If no suitable agent is available or if invalid capability flags are provided
            RuntimeError: If prompt execution fails
        """
        # Validate capability requirements if provided
        if required_capabilities:
            LLMCapabilities.validate_capability_flags(required_capabilities)

        # Get or create an agent
        agent = await self.get_agent(required_capabilities)

        # Verify agent has required capabilities
        if required_capabilities:
            missing_capabilities = {cap for cap in required_capabilities if not getattr(agent.agent.capabilities, cap)}
            if missing_capabilities:
                raise ValueError(f"Agent does not support required capabilities: {missing_capabilities}")

        # Create and store the async task
        prompt_id = str(uuid.uuid4())
        task = self.loop.create_task(agent.process_prompt(prompt))
        self.prompts[prompt_id] = task
        agent.active_prompts[prompt_id] = task

        try:
            return await task
        finally:
            # Cleanup
            if prompt_id in self.prompts:
                del self.prompts[prompt_id]
            if prompt_id in agent.active_prompts:
                del agent.active_prompts[prompt_id]

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent pool.

        Returns a dictionary containing current pool statistics including:
        - Total number of agents
        - Maximum allowed agents
        - Number of active prompts
        - Per-agent statistics
        """
        return {
            "total_agents": len(self._active_agents),
            "max_agents": self._max_agents,
            "active_prompts": sum(len(agent.active_prompts) for agent in self._active_agents.values()),
            "agents": [
                {
                    "id": agent.agent.id,
                    "type": agent.agent.type,
                    "state": agent.agent.state,
                    "active_prompts": len(agent.active_prompts),
                    "metrics": agent.agent.metrics.model_dump() if agent.agent.metrics else None,
                }
                for agent in self._active_agents.values()
            ],
        }
