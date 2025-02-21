"""Configuration manager for the application."""

from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, PrivateAttr

from llmaestro.config.agent import AgentPoolConfig, AgentTypeConfig
from llmaestro.config.system import SystemConfig
from llmaestro.config.user import UserConfig
from llmaestro.llm import LLMRegistry, ProviderRegistry
from llmaestro.llm.provider_registry import Provider


class ConfigurationManager(BaseModel):
    """Configuration manager for the application."""

    user_config: UserConfig
    system_config: SystemConfig
    _provider_registry: ProviderRegistry = PrivateAttr()
    _llm_registry: LLMRegistry = PrivateAttr()
    _agent_pool_config: Optional[AgentPoolConfig] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize registries
        self._provider_registry = ProviderRegistry()
        for name, config in self.system_config.providers.items():
            if isinstance(config, dict):
                config = Provider(**config)
            elif not isinstance(config, Provider):
                # Convert ProviderSystemConfig to Provider
                config = Provider(
                    name=config.name,
                    api_base=config.api_base,
                    capabilities_detector=config.capabilities_detector,
                    rate_limits=config.rate_limits,
                    features=set(),
                    models={},
                )
            self._provider_registry.register_provider(name, config)

        self._llm_registry = LLMRegistry(self._provider_registry)
        self._register_models()

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

    def _register_models(self) -> None:
        """Register models from configuration."""
        # Register default model
        default_model = self.user_config.default_model
        self._llm_registry.register_from_provider(default_model.provider, default_model.name)

        # Register agent models
        for agent_config in self.user_config.agents.agent_types.values():
            self._llm_registry.register_from_provider(agent_config.provider, agent_config.model)

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
    def provider_registry(self) -> ProviderRegistry:
        """Get the provider registry."""
        return self._provider_registry

    @property
    def llm_registry(self) -> LLMRegistry:
        """Get the model registry."""
        return self._llm_registry

    def get_model_config(self, provider: str, model_name: str) -> dict:
        """Get the combined configuration for a specific model."""
        base_config = self.provider_registry.get_provider_api_config(
            provider=provider, model_name=model_name, api_key=self.user_config.api_keys.get(provider)
        )

        # Add user settings if this is the default model
        default_model = self.user_config.default_model
        if default_model.provider == provider and default_model.name == model_name:
            base_config.update(default_model.settings)

        return base_config

    def get_agent_config(self, agent_type: Optional[str] = None) -> AgentTypeConfig:
        """Get the configuration for a specific agent type."""
        return self.user_config.agents.get_agent_config(agent_type)
