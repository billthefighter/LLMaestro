"""Configuration manager for the application."""

from pathlib import Path
from typing import Optional, Type, Union

from pydantic import BaseModel, ConfigDict, PrivateAttr

from llmaestro.config.agent import AgentPoolConfig, AgentTypeConfig
from llmaestro.config.system import SystemConfig
from llmaestro.config.user import UserConfig
from llmaestro.llm import LLMRegistry
from llmaestro.llm.capability_detector import BaseCapabilityDetector
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.llm.provider_registry import Provider


class ConfigurationManager(BaseModel):
    """Configuration manager for the application."""

    user_config: UserConfig
    system_config: SystemConfig
    _llm_registry: LLMRegistry = PrivateAttr()
    _agent_pool_config: Optional[AgentPoolConfig] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize LLMRegistry with API keys from user config
        self._llm_registry = LLMRegistry.create_default(
            auto_update_capabilities=True, api_keys=self.user_config.api_keys
        )

        # Register any additional provider configurations from system config
        for name, config in self.system_config.providers.items():
            if isinstance(config, dict):
                config = Provider(**config)
            elif not isinstance(config, Provider):
                # Get capability detector class if specified
                detector: Optional[Type[BaseCapabilityDetector]] = None
                if hasattr(config, "capabilities_detector"):
                    # This should be imported and resolved by the system config
                    detector = config.capabilities_detector

                # Convert ProviderSystemConfig to Provider
                config = Provider(
                    name=config.name,
                    api_base=config.api_base,
                    capabilities_detector=detector,
                    rate_limits=config.rate_limits,
                    features=set(),
                )
            self._llm_registry.provider_registry.register_provider(name, config)

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
        if self._agent_pool_config is None:
            self._agent_pool_config = self._create_agent_pool_config()
        return self._agent_pool_config

    @property
    def llm_registry(self) -> LLMRegistry:
        """Get the LLM registry."""
        return self._llm_registry

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

        return await create_llm_interface(config=config, llm_registry=self._llm_registry)

    def get_agent_config(self, agent_type: Optional[str] = None) -> AgentTypeConfig:
        """Get the configuration for a specific agent type."""
        return self.user_config.agents.get_agent_config(agent_type)
