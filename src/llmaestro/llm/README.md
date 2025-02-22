# LLM Module

This module provides a comprehensive system for managing and interacting with Large Language Models (LLMs) from various providers. It handles model registration, capability detection, provider configuration, and runtime management.

## Core Components

### Registry System

The module implements a two-tier registry system where the LLMRegistry acts as the primary interface, while the ProviderRegistry manages provider-specific configurations:

#### LLM Registry (`llm_registry.py`)
- **Purpose**: Central repository for all available LLM models and their capabilities
- **Key Features**:
  - Model registration and querying
  - Capability-based model filtering
  - Model validation and deprecation handling
  - Automatic capability updates
  - YAML-based configuration management
- **Usage**: Primary interface for:
  - Finding models with specific capabilities
  - Validating model availability
  - Managing model configurations
  - Auto-updating model capabilities

#### Provider Registry (`provider_registry.py`)
- **Purpose**: Internal registry used by LLMRegistry to manage provider configurations
- **Key Features**:
  - Provider API configuration storage
  - Rate limit configurations
  - Provider-specific model mappings
- **Usage**: Typically accessed through LLMRegistry for:
  - Provider API settings
  - Rate limit configurations
  - Provider-specific features

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
    api_keys={"anthropic": "key"}
)

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

### Registry Relationship

```
┌─────────────────────────────────────┐
│            LLMRegistry             │
├─────────────────────────────────────┤
│ - Manages model profiles           │
│ - Handles capability detection     │
│ - Provides primary API interface   │
│                                   │
│  ┌───────────────────────────┐    │
│  │    ProviderRegistry       │    │
│  ├───────────────────────────┤    │
│  │ - Stores provider configs │    │
│  │ - Manages rate limits    │    │
│  │ - API settings          │    │
│  └───────────────────────────┘    │
└─────────────────────────────────────┘
```

### Configuration Flow

1. **YAML Configuration**:
   ```yaml
   # provider.yaml
   provider:
     name: "anthropic"
     api_base: "https://api.anthropic.com/v1"
     rate_limits:
       requests_per_minute: 1000
   models:
     - name: "claude-3-opus"
       capabilities:
         max_context_window: 200000
         # ... other capabilities
   ```

2. **Registry Initialization**:
   ```python
   # Initialize with automatic capability updates
   registry = LLMRegistry.create_default(
       auto_update_capabilities=True,
       api_keys={
           "anthropic": "your-key",
           "openai": "your-key"
       }
   )
   ```

3. **Interface Creation**:
   ```python
   # Create interface with registry configuration
   interface = await create_llm_interface(
       config=agent_config,
       llm_registry=registry
   )
   ```

### Model Management

#### Models (`models.py`)
- **Purpose**: Core domain models and data structures
- **Key Components**:
  - `ModelFamily`: Enumeration of supported model families
  - `LLMCapabilities`: Model capability specification
  - `LLMProfile`: Complete model configuration
  - `Provider`: Provider configuration with API settings

#### Capability Detection (`capability_detector.py`)
- **Purpose**: Dynamic detection and verification of model capabilities
- **Key Features**:
  - Provider-specific capability detection
  - Automatic capability updates
  - Fallback to YAML configurations
  - Flexible detector specification (string paths or class references)

#### Detector Resolution
The system supports two ways to specify capability detectors:

1. **String Path**:
   ```yaml
   # In provider.yaml
   provider:
     name: "openai"
     capabilities_detector: "llmaestro.providers.openai.OpenAICapabilitiesDetector"
   ```

2. **Direct Class Reference**:
   ```python
   from llmaestro.providers.openai import OpenAICapabilitiesDetector

   provider_config = Provider(
       name="openai",
       capabilities_detector=OpenAICapabilitiesDetector,
       # ... other config
   )
   ```

The resolution system will:
- Import and validate string paths at runtime
- Verify proper inheritance from `BaseCapabilityDetector`
- Handle both forms transparently in the configuration system
- Provide clear error messages for invalid detectors

## Usage Examples

### Basic Registry Usage

```python
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.interfaces.factory import create_llm_interface

# Create registry with automatic updates
registry = LLMRegistry.create_default(
    auto_update_capabilities=True,
    api_keys={
        "anthropic": "your-anthropic-key",
        "openai": "your-openai-key"
    }
)

# Create interface for a model
interface = await create_llm_interface(
    config=agent_config,
    llm_registry=registry
)

# Use interface
response = await interface.generate("Your prompt here")
```

### Manual Capability Updates

```python
# Create without auto-updates
registry = LLMRegistry.create_default(auto_update_capabilities=False)

# Update specific model
model = await registry.detect_and_register_model(
    provider="anthropic",
    model_name="claude-3-opus",
    api_key="your-key"
)
```

### Saving Updated Configurations

```python
# Save updated configurations back to YAML
registry.to_provider_files("path/to/model_library")
```

## Design Goals

1. **Single Source of Truth**: YAML files provide base configurations
2. **Dynamic Updates**: Automatic capability verification and updates
3. **Separation of Concerns**: LLMRegistry for external interface, ProviderRegistry for internal management
4. **Type Safety**: Strong typing throughout
5. **Configuration Management**: Flexible, persistent configuration
6. **Runtime Efficiency**: Concurrent capability updates

## Contributing

When contributing to this module:

1. Follow the existing architecture patterns
2. Update YAML configurations for new models
3. Implement capability detectors for new providers
4. Add comprehensive tests
5. Update documentation
