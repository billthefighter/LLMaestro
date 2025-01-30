# LLM Provider Interfaces

This directory contains implementations for different LLM providers (OpenAI, Anthropic, etc.). Each implementation inherits from the `BaseLLMInterface` class and provides standardized access to the underlying LLM service.

## Current Providers

- **OpenAI** (`openai.py`): Implementation for GPT models via OpenAI's API
- **Anthropic** (`anthropic.py`): Implementation for Claude models via Anthropic's API

## Adding a New Provider

To add support for a new LLM provider:

1. Create a new file in this directory (e.g., `your_provider.py`)
2. Implement a class that inherits from `BaseLLMInterface`
3. Add the class to `__init__.py`
4. Update the factory function in `base.py`

### Required Implementation

Your provider class must implement:

```python
class YourProviderLLM(BaseLLMInterface):
    async def process(self, input_data: Any, system_prompt: Optional[str] = None) -> LLMResponse:
        try:
            # 1. Format messages using base class method
            messages = self._format_messages(input_data, system_prompt)

            # 2. Handle task reminders if needed
            if await self._maybe_add_reminder():
                messages.append({
                    "role": "system",
                    "content": f"Remember the initial task: {self._context.initial_task}"
                })

            # 3. Make the API call (using litellm or provider's SDK)
            response = await your_api_call(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                api_key=self.config.api_key
            )

            # 4. Process response using base class method
            return await self._handle_response(response, messages)

        except Exception as e:
            return self._handle_error(e)
```

### Base Class Features

The `BaseLLMInterface` provides several utilities:

- `_format_messages()`: Formats input data and system prompts into message format
- `_handle_response()`: Processes API responses and updates conversation context
- `_handle_error()`: Standardizes error responses
- `_maybe_add_reminder()`: Manages task reminders
- `_maybe_summarize_context()`: Handles context summarization
- `_update_metrics()`: Tracks token usage and context metrics

### Configuration

Your provider needs to be added to the factory function in `base.py`:

```python
def create_llm_interface(config: AgentConfig) -> BaseLLMInterface:
    if "your_provider" in config.model_name.lower():
        return YourProviderLLM(config)
    # ... existing providers ...
```

### Provider Requirements

Each provider implementation should:

1. Use `litellm` for API calls when possible for standardization
2. Handle provider-specific message formatting if needed
3. Properly map provider responses to the `LLMResponse` format
4. Track token usage and context metrics
5. Handle provider-specific errors appropriately

### Testing

When adding a new provider:

1. Create unit tests in `tests/llm/interfaces/`
2. Test with both sync and async calls
3. Verify error handling
4. Check token tracking and context management
5. Validate response format standardization
