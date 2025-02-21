# Configuration System

The LLMaestro configuration system is organized into modular components that handle different aspects of the application. The system uses Pydantic models for type safety and validation.

## Directory Structure

```
llmaestro/config/
├── __init__.py          # Exports main configuration interfaces
├── base.py             # Core configuration models
├── agent.py            # Agent-specific configuration
├── model.py            # Model-specific configuration
├── provider.py         # Provider-specific configuration
├── system.py           # System-wide configuration
└── user.py             # User-specific configuration
```

## Configuration Components

### Base Configuration (`base.py`)
Core configuration components used across the system:
```python
from pydantic import BaseModel

class StorageConfig(BaseModel):
    path: str
    format: str

class VisualizationConfig(BaseModel):
    host: str
    port: int
    enabled: bool
    debug: bool

class LoggingConfig(BaseModel):
    level: str
    file: Optional[str]
```

### Model Configuration (`model.py`)
Model-specific configuration including capabilities and runtime settings:
```python
class LLMCapabilities(BaseModel):
    supports_streaming: bool
    supports_function_calling: bool
    max_context_window: int
    # ... other capabilities

class ModelConfig(BaseModel):
    provider: str
    name: str
    capabilities: LLMCapabilities
    settings: Dict[str, Any]
```

### Provider Configuration (`provider.py`)
Provider-specific settings and API configurations:
```python
class Provider(BaseModel):
    api_base: str
    capabilities_detector: str
    models: Dict[str, ModelConfig]
    rate_limits: Dict[str, int]
```

### Agent Configuration (`agent.py`)
Agent-specific configuration including runtime settings:
```python
class AgentTypeConfig(BaseModel):
    provider: str
    model: str
    max_tokens: int
    temperature: float
    description: Optional[str]
    settings: Dict[str, Any]
    capabilities: Optional[LLMCapabilities]

class AgentPoolConfig(BaseModel):
    max_agents: int
    default_agent_type: str
    agent_types: Dict[str, AgentTypeConfig]
```

### System Configuration (`system.py`)
System-wide settings and provider configurations:
```python
class SystemConfig(BaseModel):
    providers: Dict[str, Provider]
```

### User Configuration (`user.py`)
User-specific settings including API keys and preferences:
```python
class UserConfig(BaseModel):
    api_keys: Dict[str, str]
    default_model: DefaultModelConfig
    agents: AgentPoolConfig
    storage: StorageConfig
    visualization: VisualizationConfig
    logging: LoggingConfig
```

## Usage Examples

### 1. Loading Configuration from Files
```python
from llmaestro.config import ConfigurationManager

# Load from default locations
config = ConfigurationManager.from_yaml_files()

# Load from specific paths
config = ConfigurationManager.from_yaml_files(
    user_config_path="path/to/user_config.yml",
    system_config_path="path/to/system_config.yml"
)
```

### 2. Loading from Environment Variables
```python
from llmaestro.config import ConfigurationManager

config = ConfigurationManager.from_env()
```

### 3. Direct Configuration
```python
from llmaestro.config import (
    ConfigurationManager,
    UserConfig,
    SystemConfig,
    AgentPoolConfig
)

user_config = UserConfig(
    api_keys={"anthropic": "your-key"},
    default_model=DefaultModelConfig(
        provider="anthropic",
        name="claude-3-sonnet-20240229"
    ),
    agents=AgentPoolConfig(...)
)

system_config = SystemConfig(...)
config = ConfigurationManager.from_configs(user_config, system_config)
```

## Best Practices

1. **Configuration Validation**
   - Use Pydantic validators for complex validation rules
   - Add custom validators for domain-specific logic
   - Validate configurations at startup

2. **Security**
   - Never commit API keys to version control
   - Use environment variables for sensitive data
   - Validate and sanitize all configuration inputs

3. **Extensibility**
   - Keep configuration models modular
   - Use inheritance for shared configuration patterns
   - Add documentation for all configuration options

4. **Testing**
   - Create test configurations for different scenarios
   - Mock configuration for unit tests
   - Test configuration validation rules

## Environment Variables

The following environment variables are supported:

### Provider Configuration
```bash
ANTHROPIC_API_KEY=your-api-key
OPENAI_API_KEY=your-api-key
ANTHROPIC_MODEL=claude-3-sonnet-20240229
ANTHROPIC_MAX_TOKENS=8192
ANTHROPIC_TEMPERATURE=0.7
```

### Global Settings
```bash
LLM_MAX_AGENTS=10
LLM_STORAGE_PATH=chain_storage
LLM_STORAGE_FORMAT=json
LLM_LOG_LEVEL=INFO
LLM_VISUALIZATION_ENABLED=true
LLM_VISUALIZATION_HOST=localhost
LLM_VISUALIZATION_PORT=8501
```

### Agent Settings
```bash
LLM_DEFAULT_AGENT_TYPE=general
LLM_AGENT_MAX_TOKENS=8192
LLM_AGENT_TEMPERATURE=0.7
```
