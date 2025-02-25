"""Session management for LLM interactions."""
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llmaestro.config import ConfigurationManager
from llmaestro.config.agent import AgentRuntimeConfig, AgentTypeConfig
from llmaestro.core import logging_config
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
from llmaestro.llm.factory import LLMFactory

# Configure module logger
logger = logging_config.configure_logging(module_name=__name__)


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

    Initialization:
        There are two ways to initialize a Session:

        1. Using the recommended factory method (preferred):
        ```python
        session = await Session.create_default(
            api_key="your-api-key",
            storage_path="./custom_storage"
        )
        # Session is ready to use immediately
        ```

        2. Manual initialization (advanced):
        ```python
        session = Session(api_key="your-api-key")
        await session.initialize()  # Required before use
        ```

        The factory method is recommended as it:
        - Ensures proper initialization of all components
        - Handles both sync and async setup automatically
        - Provides clear parameter validation
        - Returns a fully initialized session ready for use

    Attributes:
        config (ConfigurationManager): Core configuration manager
        llm_registry (Optional[LLMRegistry]): Registry for managing LLM models
        prompt_loader (PromptLoader): Loader for managing prompts
        session_id (str): Unique identifier for this session
        created_at (datetime): Session creation timestamp
        storage_path (Path): Path for artifact storage
        storage (Optional[FileSystemArtifactStorage]): Storage manager
        orchestrator (Optional[Orchestrator]): Conversation orchestrator
        active_conversation_id (Optional[str]): Current conversation ID
        api_key (Optional[str]): API key for LLM provider
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
    initialized: bool = Field(default=False, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @classmethod
    async def create_default(
        cls,
        api_key: Optional[str] = None,
        storage_path: Optional[Union[str, Path]] = None,
        config: Optional[ConfigurationManager] = None,
        llm_registry: Optional[LLMRegistry] = None,
        prompt_loader: Optional[PromptLoader] = None,
        session_id: Optional[str] = None,
    ) -> "Session":
        """Create and initialize a new session with default settings.

        This is the recommended way to create a new session as it ensures proper
        initialization of all components, including async initialization.

        Args:
            api_key: Optional API key for the default provider
            storage_path: Optional custom storage path
            config: Optional custom configuration manager
            llm_registry: Optional custom LLM registry
            prompt_loader: Optional custom prompt loader
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
        if config:
            kwargs["config"] = config
        if llm_registry:
            kwargs["llm_registry"] = llm_registry
        if prompt_loader:
            kwargs["prompt_loader"] = prompt_loader
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

        logger.debug("Initializing LLM registry")
        self.llm_registry = self.config.llm_registry

        # Update provider configurations with API keys
        if self.api_key and self.llm_registry:
            logger.debug("Updating provider configurations with API key")
            # Update the provider API configuration with the API key
            provider = self.config.user_config.default_model.provider
            provider_config = self.llm_registry.get_provider_config(provider)
            if provider_config:
                logger.debug(f"Configuring provider: {provider}")
                # Create a new provider config with the API key
                updated_config = provider_config.model_dump()
                updated_config["api_key"] = self.api_key
                self.llm_registry.provider_registry.register_provider(provider, Provider(**updated_config))

    async def initialize(self) -> None:
        """Complete async initialization of the session.

        This method must be called after creating a new session to ensure
        all async components are properly initialized.
        """
        if self.initialized:
            return

        logger.debug("Performing async initialization")
        if self.config.user_config.agents:
            logger.debug("Creating agent pool")
            from llmaestro.agents.agent_pool import AgentPool

            agent_pool = AgentPool(self.config.user_config.agents)
            self.orchestrator = Orchestrator(agent_pool)
            logger.info("Orchestrator initialized successfully")
        else:
            logger.warning("Could not initialize orchestrator - missing agents configuration")

        self.initialized = True
        logger.info("Session async initialization complete")

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
        logger.info(f"Starting new conversation: {name}")
        if not self.orchestrator:
            logger.error("Cannot start conversation - orchestrator not initialized")
            raise RuntimeError("Orchestrator not initialized")

        logger.debug("Creating conversation with initial prompt")
        conversation = await self.orchestrator.create_conversation(
            name=name, initial_prompt=initial_prompt, metadata={"session_id": self.session_id}
        )
        self.active_conversation_id = conversation.id
        logger.info(f"Conversation started with ID: {conversation.id}")
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
        logger.debug(f"Executing prompt with dependencies: {dependencies}")
        if not self.orchestrator:
            logger.error("Cannot execute prompt - orchestrator not initialized")
            raise RuntimeError("Orchestrator not initialized")

        conv_id = conversation_id or self.active_conversation_id
        if not conv_id:
            logger.error("No active conversation for prompt execution")
            raise ValueError("No active conversation")

        logger.debug(f"Executing prompt in conversation: {conv_id}")
        node_id = await self.orchestrator.execute_prompt(
            conversation_id=conv_id, prompt=prompt, dependencies=dependencies
        )
        logger.info(f"Prompt executed successfully, node ID: {node_id}")
        return node_id

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

    async def get_llm_interface(self) -> Optional[BaseLLMInterface]:
        """Get or create the LLM interface."""
        logger.debug("Getting LLM interface")
        if self._llm_interface:
            logger.debug("Using existing LLM interface")
            return self._llm_interface

        try:
            logger.debug("Creating new LLM interface")
            # Get default model configuration from user config
            default_model = self.config.user_config.default_model
            if not default_model:
                logger.warning("No default model configuration found")
                return None

            provider = default_model.provider
            model = default_model.name
            api_key = self.api_key or self.config.user_config.api_keys.get(provider, "")

            if not all([provider, model, api_key]):
                logger.warning("Missing required configuration for LLM interface")
                return None

            logger.debug(f"Configuring agent for provider: {provider}, model: {model}")
            # Create agent configuration
            agent_config = AgentTypeConfig(
                provider=provider,
                model=model,
                max_tokens=default_model.settings.get("max_tokens", 8192),
                temperature=default_model.settings.get("temperature", 0.7),
                runtime=AgentRuntimeConfig(),  # Use default runtime settings
            )

            # Create LLM interface with agent config
            interface = await create_llm_interface(agent_config)
            self._llm_interface = interface
            logger.info(f"LLM interface created successfully for {provider}/{model}")
            return interface
        except Exception as e:
            logger.error(f"Error creating LLM interface: {e}", exc_info=True)
            return None

    async def get_llm_interface_async(self) -> Optional[BaseLLMInterface]:
        """Asynchronous LLM interface retrieval."""
        return await self.get_llm_interface()

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

    async def _initialize_llm(self, agent_config: AgentTypeConfig) -> None:
        """Initialize the LLM interface."""
        factory = LLMFactory(
            registry=self.llm_registry,
            provider_manager=self.provider_manager,
            credential_manager=self.credential_manager
        )
        self.llm = factory.create_llm(model_name=agent_config.model, runtime_config=agent_config.runtime)
