"""Utility for filling an agent pool with specialized agents."""
from typing import Dict, List, Optional, Set, Any

from llmaestro.agents.agent_pool import AgentPool
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.capabilities import LLMCapabilities


class PoolFiller:
    """Utility class for filling an agent pool with specialized agents."""

    # Define standard capability sets for different agent types
    AGENT_CAPABILITY_SETS = {
        "code": {"supports_function_calling", "supports_json_mode"},
        "vision": {"supports_vision"},
        "planning": {"supports_tools"},  # Note: max_context_window checked separately
        "tool": {"supports_function_calling", "supports_tools"},
        "conversation": {"supports_message_role"},
    }

    def __init__(self, llm_registry: LLMRegistry):
        """Initialize the pool filler.

        Args:
            llm_registry: Registry containing available LLM models
        """
        self.llm_registry = llm_registry
        self._model_capabilities: Dict[str, Set[str]] = {}
        self._analyze_models()

    def _analyze_models(self) -> None:
        """Analyze available models and their capabilities."""
        for name, state in self.llm_registry.model_states.items():
            caps = state.profile.capabilities
            self._model_capabilities[name] = {
                cap for cap in LLMCapabilities.VALID_CAPABILITY_FLAGS if getattr(caps, cap, False)
            }

    def get_models_by_capabilities(self, required_capabilities: Set[str]) -> List[str]:
        """Get models that support the required capabilities.

        Args:
            required_capabilities: Set of capability flags that models must support

        Returns:
            List of model names that support all required capabilities
        """
        LLMCapabilities.validate_capability_flags(required_capabilities)

        matching_models = []
        for name, capabilities in self._model_capabilities.items():
            # For planning agents, also check context window
            if "supports_tools" in required_capabilities:
                state = self.llm_registry.model_states[name]
                if state.profile.capabilities.max_context_window < 16000:
                    continue

            if required_capabilities.issubset(capabilities):
                matching_models.append(name)

        return matching_models

    async def fill_pool(self, pool: AgentPool, agent_counts: Optional[Dict[str, int]] = None) -> None:
        """Fill the agent pool with specialized agents.

        Args:
            pool: The agent pool to fill
            agent_counts: Optional dictionary specifying how many of each agent type to create.
                        If not provided, will create one of each available type.

        Example:
            ```python
            filler = PoolFiller(llm_registry)
            await filler.fill_pool(pool, {
                "code": 2,      # Create 2 code agents
                "vision": 1,    # Create 1 vision agent
                "planning": 1,  # Create 1 planning agent
            })
            ```
        """
        # Default to one of each available type if not specified
        if agent_counts is None:
            agent_counts = {}
            for agent_type, required_caps in self.AGENT_CAPABILITY_SETS.items():
                if self.get_models_by_capabilities(required_caps):
                    agent_counts[agent_type] = 1

        # Create agents of each type
        for agent_type, count in agent_counts.items():
            if agent_type not in self.AGENT_CAPABILITY_SETS:
                continue

            required_caps = self.AGENT_CAPABILITY_SETS[agent_type]
            matching_models = self.get_models_by_capabilities(required_caps)

            if not matching_models:
                continue

            # Create agents using round-robin model selection
            for i in range(count):
                model = matching_models[i % len(matching_models)]
                await pool.get_agent(required_capabilities=required_caps, description=f"{agent_type}_specialist")

    def get_capability_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of available capabilities and models.

        Returns:
            Dictionary containing:
            - Required capabilities for each agent type
            - Models supporting each capability set
            - Full capability listing for each model
        """
        summary = {}
        for agent_type, required_caps in self.AGENT_CAPABILITY_SETS.items():
            matching_models = self.get_models_by_capabilities(required_caps)
            summary[agent_type] = {
                "required_capabilities": required_caps,
                "supported_models": matching_models,
                "model_capabilities": {model: self._model_capabilities[model] for model in matching_models},
            }
        return summary
