# Configuration

This directory contains the configuration files and schema for LLMaestro.

## Configuration Files

- `system_config.yml`: System-wide provider and model configurations
- `user_config.yml`: User-specific settings and preferences (copy from `user_config.yml.example`)
- `schema.json`: JSON Schema for configuration validation

## Configuration Models

The YAML configuration files map to Pydantic models in the codebase:

### System Configuration (`system_config.yml`)

Maps to these models in `src/llm/llm_registry.py`:
```python
class ModelConfig(BaseModel):
    family: str
    context_window: int
    typical_speed: float
    features: Set[str]
    cost: Dict[str, float]

class ProviderConfig(BaseModel):
    api_base: str
    capabilities_detector: str
    models: Dict[str, ModelConfig]
    rate_limits: Dict[str, int]
    features: Optional[Set[str]]
```

Example YAML:
```yaml
providers:
  anthropic:
    api_base: https://api.anthropic.com/v1
    capabilities_detector: llm.models.ModelCapabilitiesDetector._detect_anthropic_capabilities
    models:
      claude-3-sonnet-20240229:
        family: claude
        context_window: 200000
        typical_speed: 100.0
        features:
          - streaming
          - function_calling
        cost:
          input_per_1k: 0.015
          output_per_1k: 0.024
```

### User Configuration (`user_config.yml`)

Maps to this model in `src/core/config.py`:
```python
class UserConfig(BaseModel):
    api_keys: Dict[str, str]
    default_model: Dict[str, Any]
    agents: Dict[str, Any]
    storage: Dict[str, Any]
    visualization: Dict[str, Any]
    logging: Dict[str, Any]
```

Example YAML:
```yaml
api_keys:
  anthropic: sk-ant-xxxx...
default_model:
  provider: anthropic
  name: claude-3-sonnet-20240229
  settings:
    max_tokens: 1024
    temperature: 0.7
```

## Configuration Loading

The configuration system is managed by the `ConfigurationManager` class in `src/core/config.py`. For detailed usage instructions, see:
- [Core Configuration Documentation](../src/core/README.md#configuration-management)
- [Getting Started Guide](../README.md#getting-started)

### Quick Start

1. Copy the example user configuration:
```bash
cp config/user_config.yml.example config/user_config.yml
```

2. Edit `user_config.yml` with your settings:
```yaml
api_keys:
  anthropic: your-api-key-here
```

3. In your code:
```python
from core.config import get_config

config = get_config()
```

## Security Notes

- Never commit `user_config.yml` to version control (it's in `.gitignore`)
- Use environment variables for API keys in production
- The system configuration file should be reviewed and committed with code changes
- Keep API keys secure and rotate them if exposed
