"""Configuration management for the application."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, overload

import yaml
from pydantic import BaseModel, ConfigDict, Field

from llmaestro.llm.llm_registry import ModelRegistry, ProviderConfig


class AgentTypeConfig(BaseModel):
    """Configuration for a specific agent type."""

    provider: str = Field(default="anthropic")
    model: str = Field(default="claude-3-sonnet-20240229")
    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=0.7)
    description: Optional[str] = None
    settings: Dict[str, Any] = Field(default_factory=dict)

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
        """
        Get the configuration for a specific agent type.

        Args:
            agent_type: The type of agent. If None, uses the default agent type.

        Returns:
            AgentTypeConfig for the specified or default agent type
        """
        type_to_use = agent_type or self.default_agent_type
        if type_to_use not in self.agent_types:
            raise ValueError(f"Agent type '{type_to_use}' not found in configuration")
        return self.agent_types[type_to_use]


class UserConfig(BaseModel):
    """User-specific configuration."""

    api_keys: Dict[str, str]
    default_model: Dict[str, Any]
    agents: Dict[str, Any]
    storage: Dict[str, Any]
    visualization: Dict[str, Any]
    logging: Dict[str, Any]

    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    def from_env(cls) -> "UserConfig":
        """Create configuration from environment variables.

        This method reads the system configuration to determine which provider API keys
        to look for in the environment. For each provider, it looks for:
        - {PROVIDER_NAME}_API_KEY: The API key
        - {PROVIDER_NAME}_MODEL: The default model (optional)
        - {PROVIDER_NAME}_MAX_TOKENS: Max tokens setting (optional)
        - {PROVIDER_NAME}_TEMPERATURE: Temperature setting (optional)

        Returns:
            UserConfig: Configuration loaded from environment variables

        Raises:
            ValueError: If required API keys are not found in environment
            FileNotFoundError: If system configuration file is not found
        """
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

        # Extend agent configuration to support multiple agent types
        agents_config = {
            "max_agents": int(os.getenv("LLM_MAX_AGENTS", "10")),
            "default_agent_type": os.getenv("LLM_DEFAULT_AGENT_TYPE", "general"),
            "agent_types": {
                "general": {
                    "provider": default_provider,
                    "model": default_model,
                    "max_tokens": int(os.getenv("LLM_AGENT_MAX_TOKENS", "8192")),
                    "temperature": float(os.getenv("LLM_AGENT_TEMPERATURE", "0.7")),
                    "description": "Default general-purpose agent",
                },
                # Optional additional agent types from environment
                "fast": {
                    "provider": os.getenv("LLM_FAST_AGENT_PROVIDER", default_provider),
                    "model": os.getenv("LLM_FAST_AGENT_MODEL", "claude-3-haiku-20240229"),
                    "max_tokens": int(os.getenv("LLM_FAST_AGENT_MAX_TOKENS", "4096")),
                    "temperature": float(os.getenv("LLM_FAST_AGENT_TEMPERATURE", "0.7")),
                    "description": "Fast, lightweight agent for simple tasks",
                },
                "specialist": {
                    "provider": os.getenv("LLM_SPECIALIST_AGENT_PROVIDER", default_provider),
                    "model": os.getenv("LLM_SPECIALIST_AGENT_MODEL", "claude-3-opus-20240229"),
                    "max_tokens": int(os.getenv("LLM_SPECIALIST_AGENT_MAX_TOKENS", "16384")),
                    "temperature": float(os.getenv("LLM_SPECIALIST_AGENT_TEMPERATURE", "0.7")),
                    "description": "Specialist agent for complex tasks",
                },
            },
        }

        return cls(
            api_keys=api_keys,
            default_model={
                "provider": default_provider,
                "name": default_model,
                "settings": model_settings,
            },
            agents=agents_config,
            storage={
                "path": os.getenv("LLM_STORAGE_PATH", "chain_storage"),
                "format": os.getenv("LLM_STORAGE_FORMAT", "json"),
            },
            visualization={
                "enabled": os.getenv("LLM_VISUALIZATION_ENABLED", "true").lower() == "true",
                "host": os.getenv("LLM_VISUALIZATION_HOST", "localhost"),
                "port": int(os.getenv("LLM_VISUALIZATION_PORT", "8501")),
                "debug": os.getenv("LLM_VISUALIZATION_DEBUG", "false").lower() == "true",
            },
            logging={
                "level": os.getenv("LLM_LOG_LEVEL", "INFO"),
            },
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "UserConfig":
        """Create configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            UserConfig: Configuration loaded from YAML

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the YAML is invalid
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def _register_user_models(self) -> None:
        """Register models specified in user configuration."""
        if not self._user_config or not self._model_registry:
            return

        # Register default model
        self._model_registry.register_model(
            self._user_config.default_model["provider"], self._user_config.default_model["name"]
        )

        # Register agent models
        for _, agent_config in self._user_config.agents.get("agent_types", {}).items():
            self._model_registry.register_model(agent_config["provider"], agent_config["model"])


class SystemConfig(BaseModel):
    """System-wide configuration for all providers and models."""

    providers: Dict[str, ProviderConfig]

    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SystemConfig":
        """Create configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            SystemConfig: Configuration loaded from YAML

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the YAML is invalid
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class ConfigurationManager:
    """Global configuration manager.

    This class manages both system-wide and user-specific configurations.
    It can be initialized in several ways:
    1. From YAML files (default)
    2. From environment variables
    3. From UserConfig and SystemConfig objects

    Example:
        ```python
        # From YAML files
        config = ConfigurationManager()
        config.load_configs("user_config.yml", "system_config.yml")

        # From environment variables
        config = ConfigurationManager()
        config.load_from_env()

        # From objects
        user_config = UserConfig(...)
        system_config = SystemConfig(...)
        config = ConfigurationManager()
        config.initialize(user_config, system_config)
        ```
    """

    def __init__(self):
        self._user_config: Optional[UserConfig] = None
        self._system_config: Optional[SystemConfig] = None
        self._model_registry: Optional[ModelRegistry] = None
        self._agent_pool_config: Optional[AgentPoolConfig] = None

    def initialize(
        self,
        user_config: UserConfig,
        system_config: SystemConfig,
    ) -> None:
        """Initialize the configuration manager with config objects.

        Args:
            user_config: User-specific configuration
            system_config: System-wide configuration

        Raises:
            ValueError: If the configurations are invalid
        """
        self._user_config = user_config
        self._system_config = system_config
        self._model_registry = ModelRegistry(self._system_config.providers)
        self._register_user_models()

    def load_from_env(self, system_config_path: Optional[Union[str, Path]] = None) -> None:
        """Load configuration from environment variables.

        Args:
            system_config_path: Optional path to system config. If None, uses default location.

        Raises:
            FileNotFoundError: If system config file is not found
            ValueError: If required environment variables are missing
        """
        # Load system config first
        if system_config_path is None:
            system_config_path = Path(__file__).parent.parent.parent / "config" / "system_config.yml"

        self._system_config = SystemConfig.from_yaml(system_config_path)
        self._model_registry = ModelRegistry(self._system_config.providers)

        # Load user config from environment
        self._user_config = UserConfig.from_env()
        self._register_user_models()

    @overload
    def load_configs(
        self,
        user_config: UserConfig,
        system_config: SystemConfig,
    ) -> None:
        ...

    @overload
    def load_configs(
        self,
        user_config: Optional[Union[str, Path]] = None,
        system_config: Optional[Union[str, Path]] = None,
    ) -> None:
        ...

    def load_configs(
        self,
        user_config: Union[UserConfig, str, Path, None] = None,
        system_config: Union[SystemConfig, str, Path, None] = None,
    ) -> None:
        """Load both user and system configurations.

        This method supports loading configurations either from files or from objects.
        If paths are provided, loads from YAML files. If objects are provided, uses them directly.
        If nothing is provided, attempts to load from default locations.

        Args:
            user_config: User configuration object or path to user config YAML
            system_config: System configuration object or path to system config YAML

        Raises:
            FileNotFoundError: If configuration files are not found
            ValueError: If configuration is invalid
        """
        # Get project root directory for default paths
        root_dir = Path(__file__).parent.parent.parent

        # Handle system config
        if isinstance(system_config, (str, Path)):
            system_path = Path(system_config)
            self._system_config = SystemConfig.from_yaml(system_path)
        elif isinstance(system_config, SystemConfig):
            self._system_config = system_config
        else:
            system_path = root_dir / "config" / "system_config.yml"
            if not system_path.exists():
                raise FileNotFoundError(f"System configuration file not found at {system_path}")
            self._system_config = SystemConfig.from_yaml(system_path)

        # Initialize model registry
        self._model_registry = ModelRegistry(self._system_config.providers)

        # Handle user config
        if isinstance(user_config, (str, Path)):
            user_path = Path(user_config)
            self._user_config = UserConfig.from_yaml(user_path)
        elif isinstance(user_config, UserConfig):
            self._user_config = user_config
        else:
            # Try environment variables first
            try:
                self._user_config = UserConfig.from_env()
            except ValueError:
                # Try default config file
                user_path = root_dir / "config" / "user_config.yml"
                if not user_path.exists():
                    raise FileNotFoundError(
                        f"User configuration file not found at {user_path}. "
                        "Please copy user_config.yml.example to user_config.yml "
                        "or set required environment variables."
                    ) from None
                self._user_config = UserConfig.from_yaml(user_path)

        # Register models from user config
        self._register_user_models()

    def _create_agent_pool_config(self) -> AgentPoolConfig:
        """Create AgentPoolConfig from user configuration."""
        if not self._user_config:
            return AgentPoolConfig()

        # Convert user config agent types to AgentTypeConfig
        agent_types = {}
        for type_name, agent_config in self._user_config.agents.get("agent_types", {}).items():
            agent_types[type_name] = AgentTypeConfig(
                provider=agent_config.get("provider", "anthropic"),
                model=agent_config.get("model", "claude-3-sonnet-20240229"),
                max_tokens=agent_config.get("max_tokens", 8192),
                temperature=agent_config.get("temperature", 0.7),
                description=agent_config.get("description", f"{type_name} purpose agent"),
                settings=agent_config.get("settings", {}),
            )

        return AgentPoolConfig(
            max_agents=self._user_config.agents.get("max_agents", 10),
            default_agent_type=self._user_config.agents.get("default_agent_type", "general"),
            agent_types=agent_types,
        )

    @property
    def agents(self) -> AgentPoolConfig:
        """Get the agent pool configuration."""
        if self._agent_pool_config is None:
            # Ensure configs are loaded
            if self._user_config is None:
                self.load_configs()

            # Create agent pool config
            self._agent_pool_config = self._create_agent_pool_config()

        return self._agent_pool_config

    @property
    def user_config(self) -> UserConfig:
        """Get the user configuration."""
        if self._user_config is None:
            self.load_configs()
        if self._user_config is None:  # This should never happen after load_configs
            raise RuntimeError("Failed to load user configuration")
        return self._user_config

    @property
    def system_config(self) -> SystemConfig:
        """Get the system configuration."""
        if self._system_config is None:
            self.load_configs()
        if self._system_config is None:  # This should never happen after load_configs
            raise RuntimeError("Failed to load system configuration")
        return self._system_config

    @property
    def model_registry(self) -> ModelRegistry:
        """Get the model registry."""
        if self._model_registry is None:
            self.load_configs()
        if self._model_registry is None:  # This should never happen after load_configs
            raise RuntimeError("Failed to initialize model registry")
        return self._model_registry

    def get_model_config(self, provider: str, model_name: str) -> dict:
        """Get the combined configuration for a specific model.

        Args:
            provider: The provider name (e.g., "anthropic")
            model_name: The model name (e.g., "claude-3-sonnet-20240229")

        Returns:
            Dictionary containing both system and user config for the model

        Raises:
            ValueError: If provider or model is not found
        """
        # Get base model config from registry
        base_config = self.model_registry.get_model_config(
            provider=provider, model_name=model_name, api_key=self.user_config.api_keys.get(provider)
        )

        # Add user settings if this is the default model
        if (
            self.user_config.default_model["provider"] == provider
            and self.user_config.default_model["name"] == model_name
        ):
            base_config.update(self.user_config.default_model.get("settings", {}))

        return base_config

    def get_agent_config(self, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the configuration for a specific agent type.

        Args:
            agent_type: The type of agent. If None, uses the default agent type.

        Returns:
            Dictionary with agent configuration
        """
        if not self._user_config:
            self.load_configs()

        agent_types = self._user_config.agents.get("agent_types", {})
        default_agent_type = self._user_config.agents.get("default_agent_type", "general")

        type_to_use = agent_type or default_agent_type

        if type_to_use not in agent_types:
            raise ValueError(f"Agent type '{type_to_use}' not found in configuration")

        return agent_types[type_to_use]


# Global configuration instance
_config_manager: Optional[ConfigurationManager] = None


def get_config() -> ConfigurationManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager

    return _config_manager
