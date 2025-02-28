# LLM Module Architecture

This module handles LLM (Large Language Model) interactions, response formatting, and provider interfaces.

## Key Components

### Response Handling (`responses.py`)

The response handling system uses two main classes with distinct responsibilities:

1. **ResponseFormat**
   - User-facing configuration class
   - Handles validation, parsing, and retry logic
   - Creates provider-agnostic configuration for LLM interfaces

2. **StructuredOutputConfig**
   - Data Transfer Object (DTO) between ResponseFormat and LLM interfaces
   - Encapsulates minimal configuration needed by providers
   - Maintains clean separation between configuration and implementation

This separation follows key design principles:
- Interface segregation (providers only see what they need)
- Separation of concerns (configuration vs validation/parsing)
- Loose coupling (providers don't depend on validation logic)

### Component Flow

```mermaid
graph TD
    A[User Code] --> B[ResponseFormat]
    B -- creates --> C[StructuredOutputConfig]
    C --> D[LLM Provider Interface]
    D -- configures --> E[Provider-specific output settings]
```

### Provider Interfaces

Provider interfaces consume StructuredOutputConfig through the `configure_structured_output()` method, which:
1. Handles both Pydantic models and JSON schemas
2. Converts configuration to provider-specific settings
3. Maintains abstraction between generic configuration and specific implementation

## Directory Structure

- `responses.py` - Response formatting and validation
- `interfaces/` - Provider-specific implementations
- `schema_utils.py` - Schema handling utilities
- `models.py` - Core data models
- `enums.py` - Shared enumerations
- `capabilities.py` - Provider capability definitions
- `llm_registry.py` - Provider registration and management
- `credentials.py` - Credential management
- `rate_limiter.py` - Rate limiting implementation
- `types.py` - Common type definitions

## Best Practices

1. Use Pydantic models for structured output when possible
2. Leverage StructuredOutputConfig for provider interface implementation
3. Keep provider-specific logic isolated in interface implementations
4. Use ResponseFormat for user-facing configuration

## Implementation Notes

- ResponseFormat uses StructuredOutputConfig's effective_schema for validation
- Provider interfaces should handle both direct Pydantic models and converted JSON schemas
- Configuration and implementation concerns are kept separate through the DTO pattern
