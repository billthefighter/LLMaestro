# LLM Provider Interfaces

This directory contains implementations for different LLM providers (OpenAI, Anthropic, Google, etc.). Each implementation inherits from the `BaseLLMInterface` class and provides standardized access to the underlying LLM service.

## Current Providers

- **OpenAI** (`openai.py`): Implementation for GPT models via OpenAI's API
- **Anthropic** (`anthropic.py`): Implementation for Claude models via Anthropic's API
- **Google** (`gemini.py`): Implementation for Gemini models via Google's API

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

### Core Component Integration

The interface system integrates with two core components for resource management:

#### Rate Limiter Integration

The rate limiter (`rate_limiter.py`) is integrated into the base interface to manage API usage:

1. **Initialization**:
   ```python
   self.rate_limiter = RateLimiter(
       config=RateLimitConfig(
           requests_per_minute=config.rate_limit.requests_per_minute,
           requests_per_hour=config.rate_limit.requests_per_hour,
           max_daily_tokens=config.rate_limit.max_daily_tokens,
           alert_threshold=config.rate_limit.alert_threshold,
       )
   )
   ```

2. **Usage Flow**:
   - Request received → Token estimation → Rate limit check → Process or reject
   - Automatic tracking of token usage and request frequency
   - Alert system for approaching limits

#### Token Utilities Integration

The token utilities (`token_utils.py`) provide token counting and management:

1. **Token Counter Usage**:
   ```python
   self._token_counter = TokenCounter()
   estimates = self._token_counter.estimate_messages(messages, self.model_family, self.config.model_name)
   ```

2. **Context Management**:
   - Automatic token counting for context windows
   - Smart context summarization based on token limits
   - Token-aware cost estimation

#### Integration Flow

The system integrates these components in the following way:

```
Request → BaseLLMInterface
  ↓
1. Token Estimation (TokenCounter)
  ↓
2. Rate Limit Check (RateLimiter)
  ↓
3. Context Management
  ↓
4. Process Request
  ↓
5. Update Usage Stats
```

This integration ensures:
- Accurate token usage tracking
- Enforced rate limits
- Respected context windows
- Accurate cost calculation
- Provider-specific limitation enforcement

The architecture allows for flexible scaling and addition of new models while maintaining consistent token counting and rate limiting across the system.
