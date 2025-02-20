"""User-specific configuration models."""

import os
from pathlib import Path
from typing import Dict, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

from llmaestro.config.agent import AgentPoolConfig, AgentTypeConfig
from llmaestro.config.base import (
    DefaultModelConfig,
    LoggingConfig,
    StorageConfig,
    VisualizationConfig,
)


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
