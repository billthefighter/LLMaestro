# Model Configurations

This directory contains YAML configurations for different LLM model families. The configurations define the capabilities, limitations, and features of each model.

## Structure

Each YAML file follows this structure:

```yaml
models:
  - name: string               # Model identifier
    family: string            # Model family (gpt, claude, etc.)
    capabilities:
      # Core Features
      supports_streaming: bool
      supports_function_calling: bool
      supports_vision: bool
      supports_embeddings: bool

      # Context and Performance
      max_context_window: int
      max_output_tokens: int
      typical_speed: float    # Tokens per second

      # Cost and Quotas
      input_cost_per_1k_tokens: float
      output_cost_per_1k_tokens: float
      daily_request_limit: int

      # Advanced Features
      supports_json_mode: bool
      supports_system_prompt: bool
      supports_message_role: bool
      supports_tools: bool
      supports_parallel_requests: bool

      # Input/Output Capabilities
      supported_languages: list[str]  # ISO language codes
      supported_mime_types: list[str]
      max_image_size: int            # For vision models
      max_audio_length: float        # For audio models

      # Quality and Control
      temperature:
        min_value: float
        max_value: float
        default_value: float
      top_p:
        min_value: float
        max_value: float
        default_value: float
      supports_frequency_penalty: bool
      supports_presence_penalty: bool
      supports_stop_sequences: bool

      # Specialized Features
      supports_semantic_search: bool
      supports_code_completion: bool
      supports_chat_memory: bool
      supports_few_shot_learning: bool

    is_preview: bool
    is_deprecated: bool
    min_api_version: string
    release_date: string
    end_of_life_date: string      # Optional
    recommended_replacement: string # Optional
```

## Usage

The configurations can be loaded in several ways:

```python
from src.llm.interfaces.models import ModelRegistry

# Load from YAML
registry = ModelRegistry.from_yaml("models/gpt.yaml")

# Load from database
registry = ModelRegistry.from_database(session)

# Query models
code_models = registry.get_models_by_capability(
    "supports_code_completion",
    min_context_window=8000,
    required_languages={"en", "python"}
)
```

## Available Models

### GPT Models (`gpt.yaml`)
- gpt-3.5-turbo
- gpt-4
- gpt-4-vision-preview

### Claude Models (`claude.yaml`)
- claude-2
- claude-instant-1

## Adding New Models

1. Create or modify the appropriate YAML file
2. Follow the structure above
3. Ensure all required fields are present
4. Add model-specific capabilities and limitations
5. Include documentation for any special features

## Validation

The configuration is validated using Pydantic models:
- Required fields must be present
- Values must be within specified ranges
- Enums must match defined values
- Dates must be in ISO format
