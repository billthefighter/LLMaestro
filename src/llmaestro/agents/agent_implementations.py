"""Common agent implementations for specific use cases."""
from typing import Optional, Set

from llmaestro.agents.agent_pool import RuntimeAgent
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.models import LLMInstance


class CodeAgent(RuntimeAgent):
    """Agent specialized for code-related tasks.

    Capabilities:
    - Code generation and analysis
    - Function calling for tool integration
    - JSON mode for structured output
    """

    def __init__(self, model_name: str, llm_instance: LLMInstance):
        super().__init__(model_name=model_name, llm_instance=llm_instance, description="code_specialist")
        # Verify required capabilities
        caps = llm_instance.state.profile.capabilities
        if not (caps.supports_function_calling and caps.supports_json_mode):
            raise ValueError(f"Model {model_name} does not support required code capabilities")


class VisionAgent(RuntimeAgent):
    """Agent specialized for vision-related tasks.

    Capabilities:
    - Image analysis and understanding
    - Vision-language tasks
    """

    def __init__(self, model_name: str, llm_instance: LLMInstance):
        super().__init__(model_name=model_name, llm_instance=llm_instance, description="vision_specialist")
        # Verify required capabilities
        caps = llm_instance.state.profile.capabilities
        if not caps.supports_vision:
            raise ValueError(f"Model {model_name} does not support vision capabilities")


class PlanningAgent(RuntimeAgent):
    """Agent specialized for planning and reasoning tasks.

    Capabilities:
    - Complex reasoning (requires large context window)
    - Task planning and decomposition
    - Tool use for plan execution
    """

    def __init__(self, model_name: str, llm_instance: LLMInstance):
        super().__init__(model_name=model_name, llm_instance=llm_instance, description="planning_specialist")
        # Verify required capabilities
        caps = llm_instance.state.profile.capabilities
        if not (caps.supports_tools and caps.max_context_window >= 16000):
            raise ValueError(
                f"Model {model_name} does not support planning capabilities "
                "(requires tools and large context window)"
            )


class ConversationAgent(RuntimeAgent):
    """Agent specialized for natural conversation and text generation.

    Capabilities:
    - Natural language understanding
    - Context maintenance
    - Message role support
    """

    def __init__(self, model_name: str, llm_instance: LLMInstance):
        super().__init__(model_name=model_name, llm_instance=llm_instance, description="conversation_specialist")
        # All models support basic conversation, but verify message role support
        caps = llm_instance.state.profile.capabilities
        if not caps.supports_message_role:
            raise ValueError(f"Model {model_name} does not support message roles")


class ToolAgent(RuntimeAgent):
    """Agent specialized for tool use and function calling.

    Capabilities:
    - Function calling
    - Tool use
    - JSON mode for structured output
    """

    def __init__(self, model_name: str, llm_instance: LLMInstance):
        super().__init__(model_name=model_name, llm_instance=llm_instance, description="tool_specialist")
        # Verify required capabilities
        caps = llm_instance.state.profile.capabilities
        if not (caps.supports_function_calling and caps.supports_tools):
            raise ValueError(f"Model {model_name} does not support required tool capabilities")


async def create_specialized_agent(
    agent_type: str,
    llm_registry: LLMRegistry,
    required_capabilities: Optional[Set[str]] = None,
) -> RuntimeAgent:
    """Factory function to create specialized agents.

    Args:
        agent_type: Type of agent to create
        llm_registry: Registry to get model instances from
        required_capabilities: Optional set of required capability flags from LLMCapabilities

    Returns:
        A specialized RuntimeAgent instance

    Raises:
        ValueError: If no suitable model is found for the agent type
    """
    # Map agent types to specialized classes
    agent_classes = {
        "code": CodeAgent,
        "vision": VisionAgent,
        "planning": PlanningAgent,
        "conversation": ConversationAgent,
        "tool": ToolAgent,
    }

    if agent_type not in agent_classes:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Find a suitable model from the registry
    model_name = None
    for name, state in llm_registry.model_states.items():
        caps = state.profile.capabilities
        if (
            (agent_type == "code" and caps.supports_function_calling and caps.supports_json_mode)
            or (agent_type == "vision" and caps.supports_vision)
            or (agent_type == "planning" and caps.supports_tools and caps.max_context_window >= 16000)
            or (agent_type == "tool" and caps.supports_function_calling and caps.supports_tools)
            or (agent_type == "conversation" and caps.supports_message_role)
        ):
            # Check additional required capabilities if specified
            if required_capabilities:
                if all(getattr(caps, cap, False) for cap in required_capabilities):
                    model_name = name
                    break
            else:
                model_name = name
                break

    if not model_name:
        raise ValueError(f"No suitable model found for agent type: {agent_type}")

    # Create the specialized agent
    llm_instance = await llm_registry.create_instance(model_name)
    return agent_classes[agent_type](model_name, llm_instance)
