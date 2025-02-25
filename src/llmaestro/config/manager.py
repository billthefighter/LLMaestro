"""Configuration manager for the application."""

import logging
import sys
from pathlib import Path
from typing import Any, Optional, Union, Dict

import yaml
from pydantic import BaseModel, ConfigDict, Field

from llmaestro.config.agent import AgentPoolConfig, AgentTypeConfig
from llmaestro.config.system import SystemConfig
from llmaestro.config.user import UserConfig
from llmaestro.llm.credentials import APIKey
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.llm.models import (
    LLMCapabilities,
    LLMMetadata,
    LLMProfile,
    Provider,
)


class ConfigurationManager(BaseModel):
    """Configuration manager for the application."""

    user_config: UserConfig
    system_config: SystemConfig
    api_keys: Dict[str, APIKey]
    llm_registry: LLMRegistry
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

        # Initialize credential manager with user config
        self._init_credential_manager()

        # Initialize LLM registry with model library
        self._init_llm_registry()

        # Initialize agent pool config if not provided
        if self.agent_pool_config is None:
            self.agent_pool_config = self._create_agent_pool_config()

    def _init_llm_registry(self) -> None:
        """Initialize the LLM registry with provider configurations."""
        # First initialize credential manager
        self.credential_manager = CredentialManager()

        # Create LLM registry with credential manager
        self.llm_registry = LLMRegistry(credential_manager=self.credential_manager)

        # Load provider configurations from model library
        library_path = Path(__file__).parent.parent / "llm" / "model_library"
        if not library_path.exists():
            return

        for config_file in library_path.glob("*.yaml"):
            try:
                with open(config_file) as f:
                    data = yaml.safe_load(f)

                # Create provider config
                provider_data = data["provider"]
                provider = Provider.from_name(provider_data["provider"])
                if provider == Provider.CUSTOM:
                    continue

                try:
                    provider_config = ProviderConfig(
                        provider=provider, **{k: v for k, v in provider_data.items() if k != "provider"}
                    )
                except ValueError as e:
                    logging.getLogger(__name__).warning(f"Failed to create provider from {config_file}: {str(e)}")
                    continue

                # Validate provider
                if not self.security_manager.validate_provider(str(provider)):
                    continue

                # Get security policy and validate domain
                policy = self.security_manager.get_provider_security_policy(str(provider))
                if not self.security_manager.validate_api_domain(provider_config.api_base):
                    continue

                # Register each model with its provider
                for model_name, model_data in data["models"].items():
                    capabilities = LLMCapabilities(**model_data["capabilities"])
                    metadata = LLMMetadata(**model_data.get("metadata", {}))
                    profile = LLMProfile(capabilities=capabilities, metadata=metadata)
                    self.llm_registry.register_model(model_name, provider_config, profile)

                # Initialize if we have credentials
                api_key = self.user_config.api_keys.get(str(provider))
                if api_key:
                    self.llm_registry.initialize_provider(str(provider), api_key)

            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to load provider config from {config_file}: {str(e)}")

    def initialize_provider(self, provider: Provider, api_key: str) -> None:
        """Initialize a provider with its API key.

        This method should be called when a provider's API key becomes available,
        typically just-in-time when needed.

        Args:
            provider: Provider to initialize
            api_key: API key for the provider

        Raises:
            ValueError: If provider not found or security validation fails
        """
        # Initialize provider in registry
        self.llm_registry.initialize_provider(provider, api_key)

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

    async def get_llm_interface(self, config: AgentTypeConfig) -> BaseLLMInterface:
        """Get an LLM interface for the given configuration."""
        if not self.llm_registry:
            raise ValueError("LLMRegistry not initialized")

        if not self.provider_manager:
            raise ValueError("ProviderStateManager not initialized")

        if not self.credential_manager:
            raise ValueError("CredentialManager not initialized")

        factory = LLMFactory(
            registry=self.llm_registry,
            provider_manager=self.provider_manager,
            credential_manager=self.credential_manager,
        )
        return factory.create_llm(model_name=config.model, runtime_config=config.runtime)


class DefaultConfigurationManager(ConfigurationManager):
    # TODO Mirror default_llm_factory.py
    """Default configuration manager for the application."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_registry = LLMRegistry.create_default(auto_update_capabilities=False)
