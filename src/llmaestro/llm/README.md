# LLM Module

This module provides a comprehensive system for managing and interacting with Large Language Models (LLMs) from various providers. It handles model registration, capability detection, provider configuration, and runtime management.

## Core Components

### Registry System

The module implements a multi-layered registry system:

#### Model Registry (`llm_registry.py`)
- **Purpose**: Central repository for all available LLM models and their capabilities
- **Key Features**:
  - Model registration and querying
  - Capability-based model filtering
  - Model validation and deprecation handling
  - Persistence (JSON/YAML/Database storage)
- **Usage**: Use this when you need to:
  - Find models with specific capabilities
  - Validate model availability
  - Load/save model configurations

#### Provider Registry (`provider_registry.py`)
- **Purpose**: Manages LLM provider configurations and their available models
- **Key Features**:
  - Provider API configuration management
  - Model-specific provider settings
  - Rate limit configurations
- **Usage**: Use this when you need to:
  - Configure provider API settings
  - Access provider-specific model configurations
  - Manage provider rate limits

### Model Management

#### Models (`models.py`)
- **Purpose**: Core domain models and data structures
- **Key Components**:
  - `ModelFamily`: Enumeration of supported model families
  - `LLMCapabilities`: Comprehensive model capability specification
  - `LLMProfile`: Model metadata and configuration
  - `MediaType`: Supported media type handling

#### Capability Detection (`capability_detector.py`)
- **Purpose**: Dynamic detection and verification of model capabilities
- **Key Features**:
  - Provider-specific capability detection
  - Automatic capability registration
  - Extensible detector framework
- **Usage**: Use this when you need to:
  - Detect capabilities of new models
  - Register models with their detected capabilities
  - Add support for new providers

### Runtime Support

#### Rate Limiting (`rate_limiter.py`)
- **Purpose**: Enforces provider-specific rate limits and quotas
- **Key Features**:
  - Token-based rate limiting
  - Request throttling
  - Quota management

#### Token Utilities (`token_utils.py`)
- **Purpose**: Token counting and management utilities
- **Key Features**:
  - Token counting for different models
  - Cost calculation
  - Token optimization

### Interface Layer (`interfaces/`)
- **Purpose**: Provider-specific implementations of the LLM interface
- **Key Features**:
  - Standardized interface for all providers
  - Provider-specific optimizations
  - Error handling and retries

## Architecture Overview

```
LLM Module
├── Registry Layer
│   ├── Model Registry (model management)
│   └── Provider Registry (provider configuration)
├── Capability Layer
│   ├── Capability Detection
│   └── Model Capabilities
├── Runtime Layer
│   ├── Rate Limiting
│   └── Token Management
└── Interface Layer
    └── Provider Implementations
```

## Usage Examples

### Registering a New Model

```python
from llmaestro.llm.capability_detector import ModelCapabilityDetectorFactory

# Register a specific model
model = await ModelCapabilityDetectorFactory.register_claude_3_5_sonnet_latest(api_key)

# Register all available models
registry = await ModelCapabilityDetectorFactory.register_all_models(
    anthropic_api_key="...",
    openai_api_key="..."
)
```

### Finding Models by Capability

```python
from llmaestro.llm.llm_registry import LLMRegistry

registry = LLMRegistry()
vision_models = registry.get_models_by_capability(
    capability="supports_vision",
    min_context_window=16000,
    max_cost_per_1k=0.01
)
```

### Configuring a Provider

```python
from llmaestro.llm.provider_registry import ProviderRegistry, Provider

provider_registry = ProviderRegistry()
provider_registry.register_provider(
    "anthropic",
    Provider(
        api_base="https://api.anthropic.com",
        rate_limits={"requests_per_minute": 60},
        ...
    )
)
```

## Design Goals

1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Extensibility**: Easy to add new providers and capabilities
3. **Type Safety**: Strong typing and validation throughout
4. **Configuration Management**: Flexible configuration at all levels
5. **Runtime Efficiency**: Optimized for high-throughput production use

## Adding New Providers

To add support for a new LLM provider:

1. Create a new capability detector in `capability_detector.py`
2. Add provider configuration in `provider_registry.py`
3. Implement the provider interface in `interfaces/`
4. Register the provider's models using the capability detector

## Contributing

When contributing to this module:

1. Follow the existing architecture patterns
2. Maintain separation of concerns
3. Add comprehensive tests for new functionality
4. Update documentation for significant changes
5. Consider backward compatibility
