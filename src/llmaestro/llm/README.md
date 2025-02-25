# LLM Module

This module provides a comprehensive system for managing and interacting with Large Language Models (LLMs) from various providers. It handles model registration, capability detection, provider configuration, and runtime management.

## Core Components

### Registry System

The module implements a registry system through the LLMRegistry which manages both provider configurations and model-specific functionality:

#### LLM Registry (`llm_registry.py`)
- **Purpose**: Core registry for managing provider configurations, credentials, and model-specific functionality
- **Key Features**:
  - Provider registration and initialization
  - Credential management integration
  - API configuration storage
  - Rate limit configurations
  - Provider state management (registered vs. initialized)
  - Model registration and querying
  - Capability-based model filtering
  - Model validation and deprecation handling
  - Automatic capability updates
- **Usage**: High-level interface for:
  - Managing provider lifecycles
  - Handling provider credentials
  - Validating provider states
  - Storing provider configurations
  - Finding models with specific capabilities
  - Validating model availability
  - Managing model configurations
  - Auto-updating model capabilities

### Interface Factory System

The module uses a factory pattern to create provider-specific interfaces:

```
┌─────────────────────┐
│    LLMRegistry     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Interface Factory  │
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌─────────┐ ┌─────────┐
│ OpenAI  │ │Anthropic│
│Interface│ │Interface│
└─────────┘ └─────────┘
```

#### Factory Integration
```python
# Create registry with configurations
registry = LLMRegistry.create_default(
    auto_update_capabilities=True,
    credential_manager=credential_manager
)

# Initialize provider with API key
registry.initialize_provider("anthropic", "key")

# Create interface using registry
interface = await create_llm_interface(
    config=agent_config,
    llm_registry=registry
)

# Interface automatically configured with:
# - Model capabilities from registry
# - Provider settings from registry
# - Rate limits from provider
# - API configuration
```

### Registry Structure

```
┌─────────────────────────────────────┐
│           LLMRegistry              │
├─────────────────────────────────────┤
│ - Manages provider lifecycles      │
│ - Handles credentials              │
│ - Stores provider configurations   │
│ - Validates provider states        │
│ - Manages model profiles          │
│ - Handles capabilities            │
│ - Model validation               │
└─────────────────────────────────────┘
```

### Initialization Flow

1. **Registry Setup and Provider Initialization**:
   ```python
   # Create and initialize registry
   registry = LLMRegistry.create_default()
   
   # Initialize provider with credentials
   registry.initialize_provider(
       provider_name="anthropic",
       api_key="sk-ant-..."
   )
   ```

2. **Interface Creation**:
   ```python
   # Create interface using LLM registry
   interface = await create_llm_interface(
       config=agent_config,
       llm_registry=registry
   )
   ```

### Configuration Flow

1. **Provider Configuration**:
   ```yaml
   # provider.yaml
   provider:
     name: "anthropic"
     api_base: "https://api.anthropic.com/v1"
     rate_limits:
       requests_per_minute: 1000
     models:
       claude-3-opus:
         capabilities:
           max_context_window: 200000
           # ... other capabilities
   ```

2. **Registry Initialization**:
   ```python
   # Create and initialize registry
   registry = LLMRegistry.create_default()
   
   # Initialize with credentials
   registry.initialize_provider(
       provider_name="anthropic",
       api_key="your-key"
   )
   ```

### Model Management

#### Models (`models.py`)
- **Purpose**: Core domain models and data structures
- **Key Components**:
  - `ModelFamily`: Enumeration of supported model families (e.g., CLAUDE, GPT)
  - `LLMCapabilities`: Model capability specification
  - `LLMProfile`: Complete model configuration
  - `Provider`: Provider configuration with API settings

#### Provider State Management
The system maintains clear provider states:
1. **Registered**: Provider configuration loaded but not initialized
2. **Initialized**: Provider has valid credentials and is ready for use

```python
# Check provider state
if registry.is_provider_initialized("anthropic"):
    # Provider is ready to use
    model = registry.get_model("claude-3-opus")
```

## Usage Examples

### Basic Registry Usage

```python
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.models import ModelFamily

# Create and initialize registry
registry = LLMRegistry.create_default()
registry.initialize_provider(
    provider_name="anthropic",
    api_key="your-anthropic-key"
)

# Create interface
interface = await create_llm_interface(
    config=agent_config,
    llm_registry=registry
)

# Use interface
response = await interface.generate("Your prompt here")
```

### Manual Capability Updates

```python
# Create registry
registry = LLMRegistry.create_default(
    auto_update_capabilities=False
)

# Update specific model
model = await registry.detect_and_register_model(
    provider_name="anthropic",
    model_name="claude-3-opus",
    api_key="your-key"
)
```

## Design Goals

1. **Unified Registry**: Single registry managing both provider and model functionality
2. **State Management**: Clear provider lifecycle states
3. **Type Safety**: Using ModelFamily enum instead of strings
4. **Just-in-Time Initialization**: Providers can be initialized when credentials become available
5. **Configuration Management**: Flexible, persistent configuration

## Contributing

When contributing to this module:

1. Follow the unified registry pattern
2. Use ModelFamily enum for provider identification
3. Maintain clear provider states
4. Update YAML configurations for new models
5. Implement capability detectors for new providers
6. Add comprehensive tests
7. Update documentation
