"""Configuration manager for the application."""

import logging
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field

from llmaestro.config.agent import AgentPoolConfig, AgentTypeConfig
from llmaestro.config.system import SystemConfig
from llmaestro.config.user import UserConfig
from llmaestro.llm import LLMRegistry
from llmaestro.llm.capability_detector import BaseCapabilityDetector
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.llm.provider_registry import Provider


def resolve_capability_detector(
    detector_spec: Union[str, Type[BaseCapabilityDetector], None],
) -> Optional[Type[BaseCapabilityDetector]]:
    """Resolve a capability detector from either a string path or class reference.

    Args:
        detector_spec: Either a string path to a detector class (e.g. 'llmaestro.providers.openai.OpenAICapabilitiesDetector')
                      or a direct reference to a detector class

    Returns:
        Resolved detector class or None if no detector specified

    Raises:
        ValueError: If the detector cannot be resolved or is invalid
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    logger.debug(f"Attempting to resolve capability detector: {detector_spec}")
    logger.debug(f"Type of detector_spec: {type(detector_spec)}")

    if detector_spec is None:
        logger.debug("Detector spec is None, returning None")
        return None

    # If it's already a class and a subclass of BaseCapabilityDetector, return it
    if isinstance(detector_spec, type):
        logger.debug(f"Detector spec is a class: {detector_spec.__name__}")
        if issubclass(detector_spec, BaseCapabilityDetector):
            return detector_spec
        raise ValueError(f"Class {detector_spec.__name__} must be a subclass of BaseCapabilityDetector")

    # If it's a string, try to import and resolve it
    if isinstance(detector_spec, str):
        logger.debug(f"Detector spec is a string: {detector_spec}")
        module_path = ""
        class_name = ""
        try:
            module_path, class_name = detector_spec.rsplit(".", 1)
            logger.debug(f"Attempting to import module: {module_path}")
            module = import_module(module_path)
            detector = getattr(module, class_name)

            # Verify it's a class and proper subclass
            if not isinstance(detector, type):
                logger.error(f"{detector_spec} is not a class")
                raise ValueError(f"{detector_spec} must be a class")
            if not issubclass(detector, BaseCapabilityDetector):
                logger.error(f"{detector_spec} is not a subclass of BaseCapabilityDetector")
                raise ValueError(f"{detector_spec} must be a subclass of BaseCapabilityDetector")

            logger.debug(f"Successfully resolved capability detector: {detector}")
            return detector
        except (ImportError, AttributeError) as e:
            # Provide a more helpful error message
            if isinstance(e, ImportError):
                error_msg = (
                    f"Failed to import capability detector '{detector_spec}'. "
                    f"The module path '{module_path}' does not exist. "
                    "Built-in capability detectors are located in 'llmaestro.llm.capability_detector'. "
                    "Please check the path and ensure the module is installed correctly. "
                    f"Error: {str(e)}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg) from e
            else:
                error_msg = (
                    f"Failed to find capability detector class '{class_name}' in module '{module_path}'. "
                    "Please check that the class name is correct and exists in the specified module. "
                    f"Error: {str(e)}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg) from e

    error_msg = f"Invalid capability detector specification. Must be either a string path or BaseCapabilityDetector subclass, got {type(detector_spec)}"
    logger.error(error_msg)
    raise ValueError(error_msg)


class ConfigurationManager(BaseModel):
    """Configuration manager for the application."""

    user_config: UserConfig
    system_config: SystemConfig
    llm_registry: LLMRegistry = Field(
        default_factory=lambda: LLMRegistry.create_default(auto_update_capabilities=False),
        description="Registry for managing LLM models and their configurations",
    )
    agent_pool_config: Optional[AgentPoolConfig] = Field(default=None, description="Configuration for the agent pool")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Perform post-initialization setup.

        This method is called after the model is fully initialized, making it safe to
        access all attributes and perform additional setup or validation.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        logger.debug("Starting ConfigurationManager post-initialization")

        # Initialize registry with API keys if using the default
        if isinstance(self.llm_registry, LLMRegistry) and not self.llm_registry.provider_registry.list_providers():
            logger.debug("Creating default LLMRegistry with API keys")
            self.llm_registry = LLMRegistry.create_default(
                auto_update_capabilities=True, api_keys=self.user_config.api_keys
            )

        # Register any additional provider configurations from system config
        logger.debug("Processing provider configurations from system config")
        for name, config in self.system_config.providers.items():
            logger.debug(f"Processing provider: {name}")
            logger.debug(f"Provider config type: {type(config)}")
            logger.debug(f"Provider config: {config}")

            if isinstance(config, dict):
                logger.debug(f"Converting dict config to Provider: {config}")
                config = Provider(**config)
            elif not isinstance(config, Provider):
                # Get capability detector class if specified
                detector = None
                if hasattr(config, "capabilities_detector"):
                    logger.debug(f"Found capabilities_detector: {config.capabilities_detector}")
                    detector = resolve_capability_detector(config.capabilities_detector)

                # Convert ProviderSystemConfig to Provider
                logger.debug("Converting ProviderSystemConfig to Provider")
                config = Provider(
                    name=config.name,
                    api_base=config.api_base,
                    capabilities_detector=detector,
                    rate_limits=config.rate_limits,
                    features=set(),
                )
            self.llm_registry.provider_registry.register_provider(name, config)

        # Initialize agent pool config if not provided
        if self.agent_pool_config is None:
            self.agent_pool_config = self._create_agent_pool_config()

    @classmethod
    def from_yaml_files(
        cls,
        user_config_path: Optional[Union[str, Path]] = None,
        system_config_path: Optional[Union[str, Path]] = None,
    ) -> "ConfigurationManager":
        """Create configuration manager from YAML files."""
        root_dir = Path(__file__).parent.parent.parent

        # Handle system config
        if system_config_path:
            system_path = Path(system_config_path)
        else:
            system_path = root_dir / "config" / "system_config.yml"
            if not system_path.exists():
                raise FileNotFoundError(f"System configuration file not found at {system_path}")
        system_config = SystemConfig.from_yaml(system_path)

        # Handle user config
        if user_config_path:
            user_path = Path(user_config_path)
            user_config = UserConfig.from_yaml(user_path)
        else:
            # Try environment variables first
            try:
                user_config = UserConfig.from_env()
            except ValueError as err:
                # Try default config file
                user_path = root_dir / "config" / "user_config.yml"
                if not user_path.exists():
                    raise FileNotFoundError(
                        f"User configuration file not found at {user_path}. "
                        "Please copy user_config.yml.example to user_config.yml "
                        "or set required environment variables."
                    ) from err
                user_config = UserConfig.from_yaml(user_path)

        return cls(user_config=user_config, system_config=system_config)

    @classmethod
    def from_env(cls, system_config_path: Optional[Union[str, Path]] = None) -> "ConfigurationManager":
        """Create configuration manager from environment variables."""
        if system_config_path is None:
            system_config_path = Path(__file__).parent.parent.parent / "config" / "system_config.yml"

        system_config = SystemConfig.from_yaml(system_config_path)
        user_config = UserConfig.from_env()

        return cls(user_config=user_config, system_config=system_config)

    @classmethod
    def from_configs(cls, user_config: UserConfig, system_config: SystemConfig) -> "ConfigurationManager":
        """Create configuration manager from config objects."""
        return cls(user_config=user_config, system_config=system_config)

    def _create_agent_pool_config(self) -> AgentPoolConfig:
        """Create AgentPoolConfig from user configuration."""
        return self.user_config.agents

    @property
    def agents(self) -> AgentPoolConfig:
        """Get the agent pool configuration."""
        if self.agent_pool_config is None:
            self.agent_pool_config = self._create_agent_pool_config()
        return self.agent_pool_config

    async def get_model_interface(self, agent_type: Optional[str] = None) -> BaseLLMInterface:
        """Get an initialized LLM interface for an agent type.

        Args:
            agent_type: Optional agent type name. If None, uses default model.

        Returns:
            Initialized LLM interface
        """
        from llmaestro.llm.interfaces.factory import create_llm_interface

        if agent_type:
            config = self.get_agent_config(agent_type)
        else:
            # Convert LLMProfileReference to AgentTypeConfig
            default = self.user_config.default_model
            config = AgentTypeConfig(
                provider=default.provider,
                model=default.name,
                max_tokens=default.settings.get("max_tokens", 4096),
                temperature=default.settings.get("temperature", 0.7),
                runtime=default.settings.get("runtime", {}),
            )

        return await create_llm_interface(config=config, llm_registry=self.llm_registry)

    def get_agent_config(self, agent_type: Optional[str] = None) -> AgentTypeConfig:
        """Get the configuration for a specific agent type."""
        return self.user_config.agents.get_agent_config(agent_type)
