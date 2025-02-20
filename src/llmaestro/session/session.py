import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from llmaestro.core.config import ConfigurationManager, get_config
from llmaestro.core.conversations import ConversationGraph, ConversationNode
from llmaestro.core.models import AgentConfig, Artifact, LLMResponse
from llmaestro.core.storage import FileSystemArtifactStorage
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.llm.interfaces.factory import create_llm_interface
from llmaestro.llm.llm_registry import ModelRegistry
from llmaestro.llm.models import ModelDescriptor
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.loader import PromptLoader
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Session(BaseModel):
    """
    Centralized session management for LLM interactions.

    Provides a unified interface for:
    - Configuration management
    - Prompt loading and management
    - LLM interface creation
    - Response tracking and storage
    - Conversation management
    """

    # Core configuration
    config: ConfigurationManager = Field(default_factory=get_config)
    model_registry: ModelRegistry = Field(default_factory=lambda: get_config().model_registry)
    prompt_loader: PromptLoader = Field(default_factory=PromptLoader)

    # Session metadata
    session_id: str = Field(default_factory=lambda: str(datetime.now().timestamp()).replace(".", "_"))
    created_at: datetime = Field(default_factory=datetime.now)

    # Storage and tracking
    storage_path: Path = Field(default_factory=lambda: Path("./session_storage"))
    storage: Optional[FileSystemArtifactStorage] = None
    responses: Dict[str, LLMResponse] = Field(default_factory=dict)

    # Conversation management
    conversation: ConversationGraph = Field(default_factory=lambda: ConversationGraph(id=str(uuid4())))

    # LLM interface configuration
    api_key: Optional[str] = None
    _llm_interface: Optional[BaseLLMInterface] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.storage = FileSystemArtifactStorage.create(self.storage_path)

    @field_validator("storage_path", mode="before")
    @classmethod
    def create_storage_path(cls, v):
        """Ensure storage path exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_llm_interface(self) -> Optional[BaseLLMInterface]:
        """Get or create the LLM interface."""
        if self._llm_interface:
            return self._llm_interface

        try:
            agent_config_dict = self.config.get_agent_config()
            if not isinstance(agent_config_dict, dict):
                return None

            provider = agent_config_dict.get("provider", "")
            model = agent_config_dict.get("model", "")
            api_key = self.api_key or agent_config_dict.get("api_key", "")

            if not all([provider, model, api_key]):
                return None

            agent_config = AgentConfig(
                provider=provider,
                model_name=model,
                api_key=api_key,
                max_tokens=agent_config_dict.get("max_tokens", 8192),
                temperature=agent_config_dict.get("temperature", 0.7),
            )

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

    def get_model_capabilities(self, model_name: Optional[str] = None) -> Optional[ModelDescriptor]:
        """
        Retrieve model capabilities from ModelRegistry.

        Args:
            model_name: Optional model name. Uses default if not provided.

        Returns:
            Optional[ModelDescriptor] with model capabilities
        """
        try:
            agent_config_dict = self.config.get_agent_config()
            if not isinstance(agent_config_dict, dict):
                return None

            model_name = model_name or agent_config_dict.get("model", "")
            if not model_name:
                return None

            model_descriptor = self.model_registry.get_model(model_name)
            if not isinstance(model_descriptor, ModelDescriptor):
                return None

            return model_descriptor
        except Exception as e:
            print(f"Error getting model capabilities: {e}")
            return None

    def add_to_conversation(
        self,
        content: Union[BasePrompt, LLMResponse],
        node_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """
        Add a new node to the conversation graph.

        Args:
            content: The prompt or response content
            node_type: Type of node (prompt/response)
            metadata: Optional metadata for the node
            parent_id: Optional ID of the parent node to link to

        Returns:
            str: ID of the newly created node
        """
        # Ensure content is of the correct type
        if not isinstance(content, (BasePrompt, LLMResponse)):
            raise TypeError("Content must be either BasePrompt or LLMResponse")

        node_id = self.conversation.add_node(content, node_type, metadata)

        if parent_id:
            edge_type = "response_to" if node_type == "response" else "continues_from"
            self.conversation.add_edge(parent_id, node_id, edge_type)

        return node_id

    def get_conversation_history(
        self, node_id: Optional[str] = None, max_depth: Optional[int] = None
    ) -> List[ConversationNode]:
        """
        Get the conversation history leading to a specific node.

        Args:
            node_id: Optional ID of the node to get history for. If None, returns full history.
            max_depth: Optional maximum depth of history to retrieve

        Returns:
            List[ConversationNode]: List of conversation nodes in chronological order
        """
        if node_id:
            return self.conversation.get_node_history(node_id, max_depth)
        return list(self.conversation.nodes.values())

    def summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive session summary using ModelRegistry.

        Returns:
            Dictionary with session and model metadata
        """
        llm_interface = self.get_llm_interface()
        model_descriptor = self.get_model_capabilities()
        artifacts = self.list_artifacts()
        conversation_summary = self.conversation.get_conversation_summary()

        model_capabilities = {}
        if model_descriptor is not None:
            capabilities = model_descriptor.capabilities.model_dump()
            model_capabilities = {
                "family": model_descriptor.family,
                "capabilities": capabilities,
                "description": model_descriptor.description,
            }

        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "artifact_count": len(artifacts),
            "response_count": len(self.responses),
            "conversation": conversation_summary,
            "config": {
                "provider": llm_interface.config.provider if llm_interface else None,
                "model": llm_interface.config.model_name if llm_interface else None,
                "model_capabilities": model_capabilities,
            },
        }

    def validate_model_for_task(self, task_requirements: Dict[str, Any]) -> bool:
        """
        Validate if the current model meets task requirements using ModelRegistry.

        Args:
            task_requirements: Dictionary of required model features

        Returns:
            Boolean indicating if model meets requirements
        """
        model_config = self.get_model_capabilities()

        for feature, required_value in task_requirements.items():
            if feature not in model_config.features:
                return False

            # Add more complex validation logic as needed
            if model_config.features[feature] < required_value:
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

    async def get_llm_interface_async(
        self, model_name: Optional[str] = None, provider: Optional[str] = None
    ) -> Optional[BaseLLMInterface]:
        """Asynchronous LLM interface retrieval."""
        return await asyncio.to_thread(self.get_llm_interface, model_name, provider)

    async def summary_async(self) -> Dict[str, Any]:
        """Asynchronous session summary generation."""
        return await asyncio.to_thread(self.summary)

    async def add_to_conversation_async(
        self,
        content: Union[BasePrompt, LLMResponse],
        node_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Asynchronous conversation node addition."""
        return await asyncio.to_thread(self.add_to_conversation, content, node_type, metadata, parent_id)
