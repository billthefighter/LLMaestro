"""Configuration management for the project."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import yaml
from jsonschema import ValidationError, validate


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    provider: str
    model: str
    api_key: str
    max_tokens: int = 1024
    temperature: float = 0.7


@dataclass
class StorageConfig:
    """Configuration for storage."""

    path: str = "chain_storage"
    format: str = "json"


@dataclass
class VisualizationConfig:
    """Configuration for visualization server."""

    host: str = "localhost"
    port: int = 8765


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    file: Optional[str] = None


@dataclass
class Config:
    """Global configuration container."""

    llm: LLMConfig
    storage: StorageConfig = field(default_factory=StorageConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None) -> "Config":
        """Load configuration from file or environment variables.

        Args:
            config_path: Path to config.yaml file. If None, will look in config directory.

        Returns:
            Loaded configuration object

        Raises:
            ValidationError: If configuration doesn't match schema
            FileNotFoundError: If config file not found
        """
        # Get project root directory
        root_dir = Path(__file__).parent.parent.parent

        # Load schema
        schema_path = root_dir / "config" / "schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found at {schema_path}")

        with open(schema_path) as f:
            schema = json.load(f)

        # Try environment variables first
        if api_key := os.getenv("ANTHROPIC_API_KEY"):
            config_data = {
                "llm": {
                    "provider": "anthropic",
                    "model": os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                    "api_key": api_key,
                    "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024")),
                    "temperature": float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
                }
            }
        else:
            # Try config file
            if config_path is None:
                config_path = root_dir / "config" / "config.yaml"
            elif isinstance(config_path, str):
                config_path = Path(config_path)

            if not config_path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found at {config_path}. "
                    "Please copy config/example_config.yaml to config/config.yaml "
                    "or set ANTHROPIC_API_KEY environment variable."
                )

            with open(config_path) as f:
                config_data = yaml.safe_load(f)

        # Validate against schema
        try:
            validate(instance=config_data, schema=schema)
        except ValidationError as e:
            raise ValidationError(f"Configuration validation failed: {e.message}") from e

        # Create config object
        return cls(
            llm=LLMConfig(**config_data["llm"]),
            storage=StorageConfig(**(config_data.get("storage", {}))),
            visualization=VisualizationConfig(**(config_data.get("visualization", {}))),
            logging=LoggingConfig(**(config_data.get("logging", {}))),
        )


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.load(config_path)
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
