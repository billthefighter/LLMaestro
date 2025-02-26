"""Session management for LLM interactions."""
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set

from llmaestro.core.conversations import ConversationNode
from llmaestro.core.models import LLMResponse
from llmaestro.core.orchestrator import ExecutionMetadata, Orchestrator
from llmaestro.core.storage import Artifact, FileSystemArtifactStorage
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.models import LLMCapabilities, LLMState
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.loader import PromptLoader
from llmaestro.agents.agent_pool import AgentPool
from llmaestro.agents.pool_filler import PoolFiller
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Configure module logger
logger = logging.getLogger(__name__)


class Session(BaseModel):
    """
    Centralized session management for LLM interactions.

    Provides a unified interface for:
    - LLM Registry management
    - Prompt loading and management
    - Agent pool orchestration
    - Response tracking and storage
    - High-level conversation access
    """

    # Core components
    llm_registry: Optional[LLMRegistry] = None
    prompt_loader: PromptLoader = Field(default_factory=PromptLoader)
    agent_pool: Optional[AgentPool] = None
    pool_filler: Optional[PoolFiller] = None

    # Session metadata
    session_id: str = Field(default_factory=lambda: str(datetime.now().timestamp()).replace(".", "_"))
    created_at: datetime = Field(default_factory=datetime.now)

    # Storage and tracking
    storage_path: Path = Field(default_factory=lambda: Path("./session_storage"))
    storage: Optional[FileSystemArtifactStorage] = None

    # Orchestration
    orchestrator: Optional[Orchestrator] = None

    # LLM configuration
    api_key: Optional[str] = None
    default_model: Optional[str] = None
    default_capabilities: Optional[Set[str]] = None
    _llm_interface: Optional[BaseLLMInterface] = None
    initialized: bool = Field(default=False, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def _ensure_initialized(self) -> None:
        """Ensure session is initialized."""
        if not self.initialized:
            raise RuntimeError("Session not initialized")
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not initialized")

    @property
    def active_conversation_id(self) -> Optional[str]:
        """Get the active conversation ID."""
        if not self.orchestrator:
            return None
        return self.orchestrator.active_conversation_id

    async def start_conversation(self, name: str, initial_prompt: BasePrompt) -> str:
        """Start a new conversation with an initial prompt."""
        logger.info(f"Starting new conversation: {name}")
        self._ensure_initialized()
        assert self.orchestrator is not None  # for type checking

        logger.debug("Creating conversation with initial prompt")
        conversation = await self.orchestrator.create_conversation(
            name=name, initial_prompt=initial_prompt, metadata={"session_id": self.session_id}
        )
        logger.info(f"Conversation started with ID: {conversation.id}")
        return conversation.id

    async def execute_prompt(self, prompt: BasePrompt, dependencies: Optional[List[str]] = None) -> str:
        """Execute a prompt in the active conversation."""
        logger.debug(f"Executing prompt with dependencies: {dependencies}")
        self._ensure_initialized()
        assert self.orchestrator is not None  # for type checking

        node_id = await self.orchestrator.execute_prompt_in_active(prompt, dependencies)
        logger.info(f"Prompt executed successfully, node ID: {node_id}")
        return node_id

    async def execute_parallel(self, prompts: List[BasePrompt], max_parallel: Optional[int] = None) -> List[str]:
        """Execute multiple prompts in parallel in the active conversation."""
        self._ensure_initialized()
        assert self.orchestrator is not None  # for type checking
        return await self.orchestrator.execute_parallel_in_active(prompts, max_parallel)

    def get_execution_status(self, node_id: str) -> ExecutionMetadata:
        """Get the execution status of a node in the active conversation."""
        self._ensure_initialized()
        assert self.orchestrator is not None  # for type checking
        return self.orchestrator.get_execution_status(node_id)

    def get_conversation_history(
        self, node_id: Optional[str] = None, max_depth: Optional[int] = None
    ) -> List[ConversationNode]:
        """Get conversation history from the active conversation."""
        self._ensure_initialized()
        assert self.orchestrator is not None  # for type checking
        return self.orchestrator.get_conversation_history(node_id, max_depth)

    def add_node_to_conversation(
        self,
        content: Union[BasePrompt, LLMResponse],
        node_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Add a node to the active conversation."""
        self._ensure_initialized()
        assert self.orchestrator is not None  # for type checking
        return self.orchestrator.add_node_to_conversation(
            content=content, node_type=node_type, metadata=metadata, parent_id=parent_id
        )

    @classmethod
    async def create_default(
        cls,
        api_key: Optional[str] = None,
        storage_path: Optional[Union[str, Path]] = None,
        llm_registry: Optional[LLMRegistry] = None,
        default_model: Optional[str] = None,
        default_capabilities: Optional[Set[str]] = None,
        session_id: Optional[str] = None,
    ) -> "Session":
        """Create and initialize a new session with default settings.

        This is the recommended way to create a new session as it ensures proper
        initialization of all components, including async initialization.

        Args:
            api_key: Optional API key for the default provider
            storage_path: Optional custom storage path
            llm_registry: Optional custom LLM registry
            default_model: Optional default model for LLM interface
            default_capabilities: Optional default capabilities for agent pool
            session_id: Optional custom session ID

        Returns:
            A fully initialized Session instance

        Example:
            ```python
            session = await Session.create_default(
                api_key="your-api-key",
                storage_path="./custom_storage"
            )
            # Session is ready to use
            ```
        """
        logger.info("Creating new default session")

        # Build initialization kwargs
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if storage_path:
            kwargs["storage_path"] = Path(storage_path)
        if llm_registry:
            kwargs["llm_registry"] = llm_registry
        if default_model:
            kwargs["default_model"] = default_model
        if default_capabilities:
            kwargs["default_capabilities"] = default_capabilities
        if session_id:
            kwargs["session_id"] = session_id

        # Create session instance
        session = cls(**kwargs)

        # Perform async initialization
        await session.initialize()

        logger.info(f"Default session created with ID: {session.session_id}")
        return session

    def model_post_init(self, __context: Any) -> None:
        """Perform synchronous post-initialization setup.

        This method is called after the model is fully initialized, making it safe to
        access all attributes and perform additional setup or validation.

        Note: Async initialization is handled separately in initialize().
        """
        logger.info(f"Initializing new session with ID: {self.session_id}")

        logger.debug("Setting up storage")
        self.storage = FileSystemArtifactStorage.create(self.storage_path)

        # Initialize LLM registry if not provided
        if not self.llm_registry:
            logger.debug("Creating new LLM registry")
            self.llm_registry = LLMRegistry()

        # Create agent pool and pool filler
        if self.llm_registry:
            logger.debug("Creating agent pool and pool filler")
            self.agent_pool = AgentPool(llm_registry=self.llm_registry)
            self.pool_filler = PoolFiller(llm_registry=self.llm_registry)

    async def initialize(self) -> None:
        """Complete async initialization of the session.

        This method must be called after creating a new session to ensure
        all async components are properly initialized.
        """
        if self.initialized:
            return

        logger.debug("Performing async initialization")

        # Initialize orchestrator with agent pool
        if self.agent_pool:
            logger.debug("Initializing orchestrator with agent pool")
            self.orchestrator = Orchestrator(self.agent_pool)

            # Fill pool with default agents if capabilities specified
            if self.default_capabilities:
                logger.debug("Filling agent pool with default capabilities")
                await self.agent_pool.get_agent(
                    required_capabilities=self.default_capabilities, description="default_agent"
                )

            logger.info("Orchestrator initialized successfully")
        else:
            logger.warning("Could not initialize orchestrator - missing agent pool")

        self.initialized = True
        logger.info("Session async initialization complete")

    @field_validator("storage_path", mode="before")
    @classmethod
    def create_storage_path(cls, v):
        """Ensure storage path exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_model_capabilities(self, model_name: Optional[str] = None) -> Optional[LLMState]:
        """Get model capabilities from registry."""
        try:
            if not self.llm_registry:
                return None

            model_name = model_name or self.default_model
            if not model_name:
                return None

            return self.llm_registry.model_states.get(model_name)
        except Exception as e:
            logger.error(f"Error getting model capabilities: {e}")
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
        logger.debug(f"Storing artifact: {name} of type {content_type}")
        artifact = Artifact(name=name, data=data, content_type=content_type, metadata=metadata or {})

        if not self.storage:
            logger.debug("Creating new storage instance")
            self.storage = FileSystemArtifactStorage.create(self.storage_path)

        if self.storage.save_artifact(artifact):
            logger.info(f"Artifact stored successfully: {name}")
            return artifact

        error_msg = f"Failed to save artifact: {name}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

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

    async def summary_async(self) -> Dict[str, Any]:
        """Asynchronous session summary generation."""
        return await asyncio.to_thread(self.summary)

    async def add_node_to_conversation_async(
        self,
        content: Union[BasePrompt, LLMResponse],
        node_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Asynchronous conversation node addition."""
        return await asyncio.to_thread(self.add_node_to_conversation, content, node_type, metadata, parent_id)
