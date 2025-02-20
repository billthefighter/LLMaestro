"""Configuration management for the application."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from llmaestro.llm import ModelRegistry, ProviderConfig, ProviderRegistry
from llmaestro.llm.models import ModelCapabilities


class StorageConfig(BaseModel):
    """Configuration for storage."""

    path: str = Field(default="chain_storage")
    format: str = Field(default="json")

    model_config = ConfigDict(validate_assignment=True)


class VisualizationConfig(BaseModel):
    """Configuration for visualization."""

    host: str = Field(default="localhost")
    port: int = Field(default=8765)
    enabled: bool = Field(default=True)
    debug: bool = Field(default=False)

    model_config = ConfigDict(validate_assignment=True)


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field(default="INFO")
    file: Optional[str] = Field(default="orchestrator.log")

    model_config = ConfigDict(validate_assignment=True)


class DefaultModelConfig(BaseModel):
    """Configuration for default model."""

    provider: str
    name: str
    settings: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)


class AgentTypeConfig(BaseModel):
    """Configuration for a specific agent type."""

    provider: str = Field(default="anthropic")
    model: str = Field(default="claude-3-sonnet-20240229")
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.7)
    description: Optional[str] = None
    settings: Dict[str, Any] = Field(default_factory=dict)
    capabilities: Optional[ModelCapabilities] = None

    model_config = ConfigDict(validate_assignment=True)


class AgentPoolConfig(BaseModel):
    """Configuration for the agent pool."""

    max_agents: int = Field(default=10, ge=1, le=100)
    default_agent_type: str = Field(default="general")
    agent_types: Dict[str, AgentTypeConfig] = Field(
        default_factory=lambda: {
            "general": AgentTypeConfig(
                provider="anthropic", model="claude-3-sonnet-20240229", description="General purpose agent"
            ),
            "fast": AgentTypeConfig(
                provider="anthropic",
                model="claude-3-haiku-20240229",
                description="Fast, lightweight agent for simple tasks",
            ),
            "specialist": AgentTypeConfig(
                provider="anthropic", model="claude-3-opus-20240229", description="Specialist agent for complex tasks"
            ),
        }
    )

    model_config = ConfigDict(validate_assignment=True)

    def get_agent_config(self, agent_type: Optional[str] = None) -> AgentTypeConfig:
        """Get configuration for a specific agent type."""
        type_to_use = agent_type or self.default_agent_type
        if type_to_use not in self.agent_types:
            raise ValueError(f"Agent type '{type_to_use}' not found in configuration")
        return self.agent_types[type_to_use]


class UserConfig(BaseModel):
    """User-specific configuration."""

    api_keys: Dict[str, str]
    default_model: DefaultModelConfig
    agents: AgentPoolConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    def from_env(cls) -> "UserConfig":
        """Create configuration from environment variables."""
        # Load system config to get provider information
        root_dir = Path(__file__).parent.parent.parent
        system_config_path = root_dir / "config" / "system_config.yml"

        if not system_config_path.exists():
            raise FileNotFoundError(f"System configuration not found at {system_config_path}")

        with open(system_config_path) as f:
            system_data = yaml.safe_load(f)

        # Collect API keys from environment
        api_keys = {}
        default_provider = None
        default_model = None

        for provider_name in system_data.get("providers", {}):
            if not isinstance(provider_name, str):
                continue

            env_prefix = provider_name.upper()
            if api_key := os.getenv(f"{env_prefix}_API_KEY"):
                api_keys[provider_name] = api_key
                # Use the first provider with an API key as default if not set
                if default_provider is None:
                    default_provider = provider_name
                    provider_config = system_data["providers"][provider_name]
                    # Get default model from environment or use first available model
                    default_model = os.getenv(
                        f"{env_prefix}_MODEL", next(iter(provider_config.get("models", {}).keys()), None)
                    )

        if not api_keys:
            raise ValueError("No API keys found in environment. Set {PROVIDER}_API_KEY for at least one provider.")

        if default_provider is None or default_model is None:
            raise ValueError(
                "Could not determine default provider and model. Ensure at least one provider has models configured."
            )

        # Get settings for default model
        env_prefix = default_provider.upper()
        model_settings = {
            "max_tokens": int(os.getenv(f"{env_prefix}_MAX_TOKENS", "1024")),
            "temperature": float(os.getenv(f"{env_prefix}_TEMPERATURE", "0.7")),
        }

        # Create agent pool config from environment
        agent_pool = AgentPoolConfig(
            max_agents=int(os.getenv("LLM_MAX_AGENTS", "10")),
            default_agent_type=os.getenv("LLM_DEFAULT_AGENT_TYPE", "general"),
            agent_types={
                "general": AgentTypeConfig(
                    provider=default_provider,
                    model=default_model,
                    max_tokens=int(os.getenv("LLM_AGENT_MAX_TOKENS", "8192")),
                    temperature=float(os.getenv("LLM_AGENT_TEMPERATURE", "0.7")),
                    description="Default general-purpose agent",
                ),
                "fast": AgentTypeConfig(
                    provider=os.getenv("LLM_FAST_AGENT_PROVIDER", default_provider),
                    model=os.getenv("LLM_FAST_AGENT_MODEL", "claude-3-haiku-20240229"),
                    max_tokens=int(os.getenv("LLM_FAST_AGENT_MAX_TOKENS", "4096")),
                    temperature=float(os.getenv("LLM_FAST_AGENT_TEMPERATURE", "0.7")),
                    description="Fast, lightweight agent for simple tasks",
                ),
                "specialist": AgentTypeConfig(
                    provider=os.getenv("LLM_SPECIALIST_AGENT_PROVIDER", default_provider),
                    model=os.getenv("LLM_SPECIALIST_AGENT_MODEL", "claude-3-opus-20240229"),
                    max_tokens=int(os.getenv("LLM_SPECIALIST_AGENT_MAX_TOKENS", "16384")),
                    temperature=float(os.getenv("LLM_SPECIALIST_AGENT_TEMPERATURE", "0.7")),
                    description="Specialist agent for complex tasks",
                ),
            },
        )

        return cls(
            api_keys=api_keys,
            default_model=DefaultModelConfig(
                provider=default_provider,
                name=default_model,
                settings=model_settings,
            ),
            agents=agent_pool,
            storage=StorageConfig(
                path=os.getenv("LLM_STORAGE_PATH", "chain_storage"),
                format=os.getenv("LLM_STORAGE_FORMAT", "json"),
            ),
            visualization=VisualizationConfig(
                enabled=os.getenv("LLM_VISUALIZATION_ENABLED", "true").lower() == "true",
                host=os.getenv("LLM_VISUALIZATION_HOST", "localhost"),
                port=int(os.getenv("LLM_VISUALIZATION_PORT", "8501")),
                debug=os.getenv("LLM_VISUALIZATION_DEBUG", "false").lower() == "true",
            ),
            logging=LoggingConfig(
                level=os.getenv("LLM_LOG_LEVEL", "INFO"),
            ),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "UserConfig":
        """Create configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class SystemConfig(BaseModel):
    """System-wide configuration for all providers and models."""

    providers: Dict[str, ProviderConfig]

    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SystemConfig":
        """Create configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class ConfigurationManager(BaseModel):
    """Configuration manager for the application."""

    user_config: UserConfig
    system_config: SystemConfig
    _provider_registry: ProviderRegistry = PrivateAttr()
    _model_registry: ModelRegistry = PrivateAttr()
    _agent_pool_config: Optional[AgentPoolConfig] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize registries
        self._provider_registry = ProviderRegistry()
        for name, config in self.system_config.providers.items():
            self._provider_registry.register_provider(name, config)

        self._model_registry = ModelRegistry(self._provider_registry)
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
        self._model_registry.register_from_provider(default_model.provider, default_model.name)

        # Register agent models
        for agent_config in self.user_config.agents.agent_types.values():
            self._model_registry.register_from_provider(agent_config.provider, agent_config.model)

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
    def model_registry(self) -> ModelRegistry:
        """Get the model registry."""
        return self._model_registry

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
