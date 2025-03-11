# Capability-Based Model Selection

## Overview

The LLMaestro library now supports capability-based model selection, allowing you to dynamically select the most appropriate model based on required capabilities rather than hardcoding specific model names. This approach offers several advantages:

1. **Cost Optimization**: Automatically select the cheapest model that meets your requirements
2. **Future-Proofing**: Your code won't break when model names change or new models are introduced
3. **Flexibility**: Easily adapt to different environments and available models
4. **Testability**: Tests can run with any model that supports the required capabilities

## Core Features

The capability-based model selection system provides two main methods:

1. `find_cheapest_model_with_capabilities`: Returns the cheapest model that supports all specified capabilities
2. `find_models_with_capabilities`: Returns all models that support the specified capabilities

## Usage Examples

### Finding the Cheapest Model with Required Capabilities

```python
from llmaestro.llm.llm_registry import LLMRegistry

# Initialize your LLM registry
llm_registry = LLMRegistry()

# Find the cheapest model that supports vision capabilities
required_capabilities = {"supports_vision"}
model_name = llm_registry.find_cheapest_model_with_capabilities(required_capabilities)

# Create an instance of the selected model
llm_instance = await llm_registry.create_instance(model_name)

# Use the instance
response = await llm_instance.interface.process(prompt)
```

### Finding All Models with Required Capabilities

```python
# Find all models that support function calling and JSON mode
required_capabilities = {"supports_function_calling", "supports_json_mode"}
matching_models = llm_registry.find_models_with_capabilities(required_capabilities)

print(f"Models supporting function calling and JSON mode: {matching_models}")
```

### Using in Tests

For tests, we provide a utility function that handles the case where no model meets the requirements:

```python
from tests.test_llm.conftest import find_cheapest_model_with_capabilities

# This will automatically skip the test if no suitable model is available
model_name = find_cheapest_model_with_capabilities(llm_registry, {"supports_vision"})
```

## Available Capabilities

The following capabilities can be used as requirements:

- `supports_streaming`: Model supports streaming responses
- `supports_function_calling`: Model supports function calling
- `supports_vision`: Model supports processing images
- `supports_embeddings`: Model supports generating embeddings
- `supports_json_mode`: Model supports JSON mode output
- `supports_system_prompt`: Model supports system prompts
- `supports_tools`: Model supports tools/plugins
- `supports_parallel_requests`: Model supports parallel requests
- `supports_frequency_penalty`: Model supports frequency penalty
- `supports_presence_penalty`: Model supports presence penalty
- `supports_stop_sequences`: Model supports stop sequences
- `supports_message_role`: Model supports message roles
- `supports_direct_pydantic_parse`: Model supports direct Pydantic parsing

## Implementation Details

The capability-based model selection is implemented in the `LLMRegistry` class. The system works by:

1. Validating that all requested capabilities are valid
2. Iterating through all registered models
3. Checking if each model supports all required capabilities
4. For the cheapest model finder, tracking the model with the lowest total cost

The cost calculation is based on the sum of input and output costs per 1,000 tokens.

## Best Practices

1. **Be Specific**: Only request the capabilities you actually need
2. **Handle Missing Models**: Always handle the case where no model meets your requirements
3. **Test Fallbacks**: For critical features, implement fallback strategies
4. **Document Requirements**: Clearly document which capabilities your code requires
