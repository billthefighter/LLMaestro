# Configuration System

The LLMaestro configuration system is organized into modular components that handle different aspects of the application. The system uses Pydantic models for type safety and validation.

## Directory Structure

```
llmaestro/config/
├── __init__.py           # Exports main configuration interfaces
├── base.py              # Core configuration models
├── agent.py             # Agent-specific configuration
├── system.py            # System-wide configuration
├── user.py              # User-specific configuration
├── security_manager.py  # Security policy management
├── credential_manager.py # Secure credential handling
└── default_configs/     # Default configuration templates
    ├── security_config.yml
    ├── system_config.yml
    ├── provider_config.yml
    └── user_config.yml.example
```

## Configuration Components

### Security Management (`security_manager.py`)
Centralized security policy management:
```python
class SecurityPolicy(BaseModel):
    """Security policy configuration for a provider."""
    require_encryption: bool
    allowed_domains: Set[str]
    allowed_endpoints: Set[str]
    max_token_limit: Optional[int]
    require_ssl: bool

class SecurityManager(BaseModel):
    """Centralizes security policies and validation."""
    allowed_api_domains: Set[str]
    require_api_key_encryption: bool
    allowed_providers: Set[str]
    provider_policies: Dict[str, SecurityPolicy]
```

### Credential Management (`credential_manager.py`)
Secure handling of API keys and sensitive data:
```python
class CredentialManager(BaseModel):
    """Manages API keys and credentials securely."""
    def add_credential(self, provider: str, api_key: str, encrypt: bool = True) -> None:
        """Add a credential with optional encryption."""
        
    def get_credential(self, provider: str) -> Optional[str]:
        """Get a credential, decrypting if necessary."""
```

### Base Configuration (`base.py`)
Core configuration components used across the system:
```python
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

### System Configuration (`system.py`)
System-wide settings and provider configurations:
```python
class LLMSystemConfig(BaseModel):
    """Global system configuration for LLM functionality."""
    global_rate_limits: Dict[str, int]
    max_parallel_requests: int
    max_retries: int
    retry_delay: float
    default_request_timeout: float
    default_stream_timeout: float
    enable_response_cache: bool
    cache_ttl: int
    log_level: str
    enable_telemetry: bool
    require_api_key_encryption: bool
    allowed_api_domains: Set[str]

class SystemConfig(BaseModel):
    """Root system configuration."""
    llm: LLMSystemConfig
```

### User Configuration (`user.py`)
User-specific settings including API keys and preferences:
```python
class UserConfig(BaseModel):
    api_keys: Dict[str, str]
    default_model: LLMProfileReference
    agents: AgentPoolConfig
    storage: StorageConfig
    visualization: VisualizationConfig
    logging: LoggingConfig
```

## Loading Configurations

The system supports multiple ways to load configurations:

### 1. From YAML Files
```python
config_manager = ConfigurationManager.from_yaml_files(
    user_config_path="path/to/user_config.yml",
    system_config_path="path/to/system_config.yml"
)
```

### 2. From Environment Variables
```python
config_manager = ConfigurationManager.from_env(
    system_config_path="path/to/system_config.yml"  # optional
)
```

### 3. Directly from Config Objects
```python
config_manager = ConfigurationManager.from_configs(
    user_config=user_config,
    system_config=system_config
)
```

## Configuration Examples

### 1. Security Configuration (security_config.yml)
```yaml
require_api_key_encryption: true
allowed_api_domains:
  - api.openai.com
  - api.anthropic.com
  - api.cohere.ai
  - api.mistral.ai

provider_policies:
  openai:
    require_encryption: true
    allowed_domains:
      - api.openai.com
    allowed_endpoints:
      - /v1/chat/completions
      - /v1/completions
    max_token_limit: 32768
    require_ssl: true
```

### 2. System Configuration (system_config.yml)
```yaml
llm:
  global_rate_limits:
    requests_per_minute: 5000
    tokens_per_minute: 500000
  max_parallel_requests: 10
  max_retries: 3
  retry_delay: 1.0
  enable_response_cache: true
  cache_ttl: 3600
  require_api_key_encryption: true
  allowed_api_domains:
    - api.openai.com
    - api.anthropic.com
```

### 3. User Configuration (user_config.yml)
```yaml
api_keys:
  anthropic: ${ANTHROPIC_API_KEY}
  openai: ${OPENAI_API_KEY}

default_model:
  provider: anthropic
  name: claude-3-sonnet-20240229
  settings:
    temperature: 0.7
    max_tokens: 4096
```

## Environment Variables

### Provider API Keys
```bash
ANTHROPIC_API_KEY=your-api-key
OPENAI_API_KEY=your-api-key
COHERE_API_KEY=your-api-key
MISTRAL_API_KEY=your-api-key
```

### Global Settings
```bash
LLM_MAX_PARALLEL_REQUESTS=10
LLM_LOG_LEVEL=INFO
LLM_ENABLE_CACHE=true
LLM_CACHE_TTL=3600
```

### Agent Settings
```bash
LLM_DEFAULT_AGENT_TYPE=general
LLM_AGENT_MAX_TOKENS=4096
LLM_AGENT_TEMPERATURE=0.7
```

## Best Practices

1. **Security First**
   - Use the SecurityManager for domain and endpoint validation
   - Enable API key encryption for sensitive providers
   - Validate API key formats before use
   - Use SSL/TLS for all API communications

2. **Credential Management**
   - Never store API keys in plain text
   - Use the CredentialManager for all API key operations
   - Implement proper key rotation practices
   - Monitor credential usage and access

3. **Configuration Separation**
   - Keep security policies in `security_config.yml`
   - Store provider configurations in the model library
   - Use environment variables for sensitive data
   - Keep user preferences in `user_config.yml`

4. **Provider Management**
   - Configure providers through the model library
   - Validate provider configurations against security policies
   - Implement proper rate limiting
   - Monitor API usage and costs

5. **Testing**
   - Create test configurations for different scenarios
   - Mock provider responses for testing
   - Validate security policies in tests
   - Test credential encryption/decryption

6. **Monitoring and Logging**
   - Enable appropriate logging levels
   - Monitor API usage and rate limits
   - Track security policy violations
   - Audit credential access and usage
