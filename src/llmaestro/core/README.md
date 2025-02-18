# Core Components

This directory contains the core components of the LLMaestro system. These components provide the foundational functionality for configuration management, task handling, and logging.

## Configuration Management

The configuration system in LLMaestro is designed to handle both system-wide and user-specific settings in a type-safe and maintainable way.

### Configuration Structure

The configuration system is split into two main parts:

1. **System Configuration** (`system_config.yml`):
   - Provider definitions and capabilities
   - Model specifications and features
   - Rate limits and API endpoints
   - Located at `src/llm/system_config.yml`

2. **User Configuration** (`config.yaml`):
   - API keys
   - Default model settings
   - Agent pool configuration
   - Storage and visualization preferences
   - Located at `config/config.yaml`

### Usage Examples

```python
from core.config import get_config

# Get the configuration manager
config = get_config()

# Access user settings
api_key = config.user_config.api_keys["anthropic"]
storage_path = config.user_config.storage["path"]

# Access system settings
provider_config = config.system_config.providers["anthropic"]
model_capabilities = provider_config.models["claude-3-sonnet-20240229"]

# Get combined model config
model_config = config.get_model_config("anthropic", "claude-3-sonnet-20240229")
```

### Configuration Sources

The system supports multiple ways to provide configuration:

1. **Environment Variables**:
   ```bash
   export ANTHROPIC_API_KEY=your-api-key
   export ANTHROPIC_MODEL=claude-3-sonnet-20240229
   export ANTHROPIC_MAX_TOKENS=1024
   export ANTHROPIC_TEMPERATURE=0.7
   ```

2. **Configuration Files**:
   ```yaml
   # config.yaml
   api_keys:
     anthropic: your-api-key
   default_model:
     provider: anthropic
     name: claude-3-sonnet-20240229
     settings:
       max_tokens: 1024
       temperature: 0.7
   ```

### Configuration Manager

The `ConfigurationManager` class provides a centralized interface for:

- Loading and validating configurations
- Managing the model registry
- Combining system and user settings
- Providing type-safe access to configuration values

Key features:
- Automatic configuration loading
- Environment variable support
- Type validation using Pydantic
- Lazy initialization
- Thread-safe singleton pattern

### Model Registry

The configuration system integrates with the Model Registry to:

- Track available models and their capabilities
- Manage provider configurations
- Handle model registration and validation
- Provide access to model-specific settings

### Best Practices

1. **API Key Security**:
   - Never commit API keys to version control
   - Use environment variables in production
   - Keep `config.yaml` in `.gitignore`

2. **Configuration Updates**:
   - System configuration changes require code review
   - User configuration can be modified without code changes
   - Use the example config as a template

3. **Type Safety**:
   - Always use the configuration manager interface
   - Avoid direct dictionary access
   - Leverage Pydantic validation

## Other Core Components

### Task Manager (`task_manager.py`)
Handles task decomposition, scheduling, and result aggregation.

### Logging Configuration (`logging_config.py`)
Configures logging behavior and output formatting.

### Core Models (`models.py`)
Base models and data structures used throughout the system.
