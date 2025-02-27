"""Session management for LLM interactions."""
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llmaestro.config import ConfigurationManager
from llmaestro.config.agent import AgentRuntimeConfig, AgentTypeConfig
from llmaestro.core.conversations import ConversationNode
from llmaestro.core.models import LLMResponse
from llmaestro.core.orchestrator import ExecutionMetadata, Orchestrator
from llmaestro.core.storage import Artifact, FileSystemArtifactStorage
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.llm.interfaces.factory import create_llm_interface
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.models import LLMCapabilities, LLMProfile
from llmaestro.llm.provider_registry import Provider
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.loader import PromptLoader
from pydantic import BaseModel, ConfigDict, Field, field_validator


def get_default_config() -> ConfigurationManager:
    """Get default configuration manager."""
    return ConfigurationManager.from_env()


class Session(BaseModel):
    """
    Centralized session management for LLM interactions.

    Provides a unified interface for:
    - Configuration management
    - Prompt loading and management
    - LLM orchestration and execution
    - Response tracking and storage
    - Conversation management
    """

    # Core configuration
    config: ConfigurationManager = Field(default_factory=get_default_config)
    llm_registry: Optional[LLMRegistry] = None
    prompt_loader: PromptLoader = Field(default_factory=PromptLoader)

    # Session metadata
    session_id: str = Field(default_factory=lambda: str(datetime.now().timestamp()).replace(".", "_"))
    created_at: datetime = Field(default_factory=datetime.now)

    # Storage and tracking
    storage_path: Path = Field(default_factory=lambda: Path("./session_storage"))
    storage: Optional[FileSystemArtifactStorage] = None

    # Orchestration
    orchestrator: Optional[Orchestrator] = None
    active_conversation_id: Optional[str] = None

    # LLM interface configuration
    api_key: Optional[str] = None
    _llm_interface: Optional[BaseLLMInterface] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.storage = FileSystemArtifactStorage.create(self.storage_path)
        self.llm_registry = self.config.llm_registry

        # Update provider configurations with API keys
        if self.api_key:
            # Update the provider API configuration with the API key
            provider = self.config.user_config.default_model.provider
            provider_config = self.config.provider_registry.get_provider(provider)
            if provider_config:
                # Create a new provider config with the API key
                updated_config = provider_config.model_dump()
                updated_config["api_key"] = self.api_key
                self.config.provider_registry.register_provider(provider, Provider(**updated_config))

        self._initialize_orchestrator()

    def _initialize_orchestrator(self) -> None:
        """Initialize the orchestrator with LLM interface."""
        llm_interface = self.get_llm_interface()
        if llm_interface and self.config.user_config.agents:
            from llmaestro.agents.agent_pool import AgentPool

            agent_pool = AgentPool(self.config.user_config.agents)
            self.orchestrator = Orchestrator(agent_pool)

    @field_validator("storage_path", mode="before")
    @classmethod
    def create_storage_path(cls, v):
        """Ensure storage path exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def start_conversation(self, name: str, initial_prompt: BasePrompt) -> str:
        """Start a new conversation with an initial prompt.

        Args:
            name: Name of the conversation
            initial_prompt: Initial prompt to start the conversation

        Returns:
            str: ID of the created conversation
        """
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not initialized")

        conversation = await self.orchestrator.create_conversation(
            name=name, initial_prompt=initial_prompt, metadata={"session_id": self.session_id}
        )
        self.active_conversation_id = conversation.id
        return conversation.id

    async def execute_prompt(
        self, prompt: BasePrompt, dependencies: Optional[List[str]] = None, conversation_id: Optional[str] = None
    ) -> str:
        """Execute a prompt in the current or specified conversation.

        Args:
            prompt: The prompt to execute
            dependencies: Optional list of node IDs this execution depends on
            conversation_id: Optional conversation ID. Uses active conversation if not specified

        Returns:
            str: ID of the response node
        """
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not initialized")

        conv_id = conversation_id or self.active_conversation_id
        if not conv_id:
            raise ValueError("No active conversation")

        return await self.orchestrator.execute_prompt(conversation_id=conv_id, prompt=prompt, dependencies=dependencies)

    async def execute_parallel(
        self, prompts: List[BasePrompt], max_parallel: Optional[int] = None, conversation_id: Optional[str] = None
    ) -> List[str]:
        """Execute multiple prompts in parallel.

        Args:
            prompts: List of prompts to execute
            max_parallel: Maximum number of parallel executions
            conversation_id: Optional conversation ID. Uses active conversation if not specified

        Returns:
            List[str]: List of response node IDs
        """
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not initialized")

        conv_id = conversation_id or self.active_conversation_id
        if not conv_id:
            raise ValueError("No active conversation")

        return await self.orchestrator.execute_parallel(
            conversation_id=conv_id, prompts=prompts, max_parallel=max_parallel
        )

    def get_execution_status(self, node_id: str, conversation_id: Optional[str] = None) -> ExecutionMetadata:
        """Get the execution status of a node.

        Args:
            node_id: ID of the node to check
            conversation_id: Optional conversation ID. Uses active conversation if not specified

        Returns:
            ExecutionMetadata: Current execution status
        """
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not initialized")

        conv_id = conversation_id or self.active_conversation_id
        if not conv_id:
            raise ValueError("No active conversation")

        return self.orchestrator.get_execution_status(conv_id, node_id)

    def get_conversation_history(
        self, node_id: Optional[str] = None, max_depth: Optional[int] = None
    ) -> List[ConversationNode]:
        """Get conversation history leading to a specific node.

        Args:
            node_id: Optional ID of the node to get history for
            max_depth: Optional maximum depth of history to retrieve

        Returns:
            List[ConversationNode]: List of conversation nodes
        """
        if not self.orchestrator or not self.active_conversation_id:
            raise RuntimeError("No active conversation")

        conversation = self.orchestrator.active_conversations[self.active_conversation_id]
        if node_id:
            return conversation.get_node_history(node_id, max_depth)
        return list(conversation.nodes.values())

    def get_model_capabilities(self, model_name: Optional[str] = None) -> Optional[LLMProfile]:
        """Get model capabilities from registry."""
        try:
            if not self.llm_registry:
                return None

            agent_config = self.config.get_agent_config()
            model_name = model_name or agent_config.model
            if not model_name:
                return None

            return self.llm_registry.get_model(model_name)
        except Exception as e:
            print(f"Error getting model capabilities: {e}")
            return None

    def summary(self) -> Dict[str, Any]:
        """Generate session summary."""
        if not self.orchestrator or not self.active_conversation_id:
            return {"session_id": self.session_id, "created_at": self.created_at.isoformat(), "status": "inactive"}

        conversation = self.orchestrator.active_conversations[self.active_conversation_id]
        conversation_summary = conversation.get_conversation_summary()
        model_descriptor = self.get_model_capabilities()

        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "conversation": conversation_summary,
            "model_capabilities": model_descriptor.model_dump() if model_descriptor else None,
        }

    def get_llm_interface(self) -> Optional[BaseLLMInterface]:
        """Get or create the LLM interface."""
        if self._llm_interface:
            return self._llm_interface

        try:
            # Get default model configuration from user config
            default_model = self.config.user_config.default_model
            if not default_model:
                return None

            provider = default_model.provider
            model = default_model.name
            api_key = self.api_key or self.config.user_config.api_keys.get(provider, "")

            if not all([provider, model, api_key]):
                return None

            # Create agent configuration
            agent_config = AgentTypeConfig(
                provider=provider,
                model=model,
                max_tokens=default_model.settings.get("max_tokens", 8192),
                temperature=default_model.settings.get("temperature", 0.7),
                runtime=AgentRuntimeConfig(),  # Use default runtime settings
            )

            # Create LLM interface with agent config
            self._llm_interface = create_llm_interface(agent_config)
            return self._llm_interface
        except Exception as e:
            print(f"Error creating LLM interface: {e}")
            return None

    def store_artifact(
        self, name: str, data: Any, content_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Artifact:
        """
        Store an artifact with enhanced tracking.

        Args:
            name: Unique name for the artifact
            data: Data to store
            content_type: Type of content being stored
            metadata: Optional additional metadata

        Returns:
            Artifact representing the stored data
        """
        artifact = Artifact(name=name, data=data, content_type=content_type, metadata=metadata or {})

        if not self.storage:
            self.storage = FileSystemArtifactStorage.create(self.storage_path)

        if self.storage.save_artifact(artifact):
            return artifact
        raise RuntimeError(f"Failed to save artifact: {name}")

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """
        Retrieve a stored artifact.

        Args:
            artifact_id: ID of the artifact to retrieve

        Returns:
            Optional[Artifact]: The retrieved artifact or None if not found
        """
        if not self.storage:
            self.storage = FileSystemArtifactStorage.create(self.storage_path)
        return self.storage.load_artifact(artifact_id)

    def list_artifacts(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Artifact]:
        """
        List artifacts matching the filter criteria.

        Args:
            filter_criteria: Optional criteria to filter artifacts

        Returns:
            List[Artifact]: List of matching artifacts
        """
        if not self.storage:
            self.storage = FileSystemArtifactStorage.create(self.storage_path)
        return self.storage.list_artifacts(filter_criteria)

    def validate_model_for_task(self, task_requirements: Dict[str, Any]) -> bool:
        """
        Validate if the current model meets task requirements using LLMRegistry.

        Args:
            task_requirements: Dictionary of required model features

        Returns:
            Boolean indicating if model meets requirements
        """
        model_config = self.get_model_capabilities()
        if not model_config:
            return False

        # Access capabilities through the LLMCapabilities interface
        model_caps = LLMCapabilities.model_validate(model_config.model_dump())

        for feature, required_value in task_requirements.items():
            if not hasattr(model_caps, feature):
                return False

            current_value = getattr(model_caps, feature, None)
            if current_value is None or current_value < required_value:
                return False

        return True

    async def store_artifact_async(
        self, name: str, data: Any, content_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Artifact:
        """Asynchronous artifact storage."""
        return await asyncio.to_thread(self.store_artifact, name, data, content_type, metadata)

    async def get_artifact_async(self, artifact_id: str) -> Optional[Artifact]:
        """Asynchronous artifact retrieval."""
        return await asyncio.to_thread(self.get_artifact, artifact_id)

    async def get_llm_interface_async(self) -> Optional[BaseLLMInterface]:
        """Asynchronous LLM interface retrieval."""
        return await asyncio.to_thread(self.get_llm_interface)

    async def summary_async(self) -> Dict[str, Any]:
        """Asynchronous session summary generation."""
        return await asyncio.to_thread(self.summary)

    def add_node_to_conversation(
        self,
        content: Union[BasePrompt, LLMResponse],
        node_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Add a node to the current conversation.

        Args:
            content: The content to add (prompt or response)
            node_type: Type of the node
            metadata: Optional metadata for the node
            parent_id: Optional ID of the parent node

        Returns:
            str: ID of the created node

        Raises:
            RuntimeError: If no active conversation exists
        """
        if not self.orchestrator or not self.active_conversation_id:
            raise RuntimeError("No active conversation")

        conversation = self.orchestrator.active_conversations[self.active_conversation_id]
        node_id = conversation.add_node(content=content, node_type=node_type, metadata=metadata or {})

        if parent_id:
            conversation.add_edge(source_id=parent_id, target_id=node_id, edge_type="response_to")

        return node_id

    async def add_node_to_conversation_async(
        self,
        content: Union[BasePrompt, LLMResponse],
        node_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Asynchronous conversation node addition."""
        return await asyncio.to_thread(self.add_node_to_conversation, content, node_type, metadata, parent_id)
