# LLMaestro Prompt System

The prompt system provides a structured way to create, manage, and execute prompts for LLM interactions. It includes support for versioning, template variables, structured outputs, and tool integration.

## Core Components

### [base.py](./base.py)

Defines the core prompt classes and functionality:

- `BasePrompt`: Foundation for all prompts with:
  - Template rendering with variable substitution
  - Attachment handling for files and images
  - Response format specification
  - Tool integration
  - Validation of templates and variables

- `VersionedPrompt`: Extends BasePrompt with version control capabilities

### [types.py](./types.py)

Defines data structures used throughout the prompt system:

- `VersionInfo`: Tracks version metadata (number, timestamp, author, etc.)
- `PromptMetadata`: Stores additional prompt information (type, tags, model requirements)

### [mixins.py](./mixins.py)

Provides reusable functionality for prompt classes:

- `VersionMixin`: Adds version control capabilities with:
  - Version history tracking
  - Creation and update timestamps
  - Author information

### [tools.py](./tools.py)

Implements function calling capabilities for prompts:

- `FunctionGuard`: Abstract base class for safely executing functions
- `BasicFunctionGuard`: Simple implementation of function safety checks
- `ToolParams`: Represents a tool/function that can be used by an LLM with:
  - Parameter schema generation
  - Function execution handling
  - Conversion utilities for different formats (OpenAI, etc.)

### [loader.py](./loader.py)

Provides mechanisms for loading and saving prompts:

- `FilePrompt`: Implementation that loads from and saves to files
- `PromptLoader`: Manages prompt loading from different storage backends
- Additional implementations for S3 and Git repositories

### [memory.py](./memory.py)

Implements in-memory prompt storage:

- `MemoryPrompt`: Non-persistent prompt implementation for testing and temporary use

## Integration with ResponseFormat

The prompt system integrates closely with the `ResponseFormat` class from `llmaestro.llm.responses` to enable structured output handling.

### How ResponseFormat Works

The `ResponseFormat` class defines the expected structure and validation rules for LLM responses:

```python
from llmaestro.llm.responses import ResponseFormat, ResponseFormatType
from pydantic import BaseModel

# Define a response model
class WeatherResponse(BaseModel):
    temperature: float
    conditions: str
    forecast: list[str]

# Create a response format from the model
response_format = ResponseFormat.from_pydantic_model(
    model=WeatherResponse,
    format_type=ResponseFormatType.JSON_SCHEMA
)
```

### Prompt and ResponseFormat Interaction

When a `BasePrompt` includes a `ResponseFormat`, it:

1. **Instructs the LLM**: The response format is included in the system prompt to guide the LLM's output
2. **Validates Responses**: Responses are validated against the schema or Pydantic model
3. **Handles Retries**: Failed validations can trigger retries with error feedback
4. **Converts Formats**: Responses can be automatically converted to the appropriate format

Example:

```python
from llmaestro.prompts.base import BasePrompt
from llmaestro.llm.responses import ResponseFormat

# Create a prompt with structured output
weather_prompt = BasePrompt(
    name="Weather Forecast",
    description="Get weather forecast for a location",
    system_prompt="You are a weather assistant. Provide accurate weather information.",
    user_prompt="What's the weather like in {location}?",
    expected_response=ResponseFormat.from_pydantic_model(WeatherResponse)
)

# The response will be validated against the WeatherResponse model
response = await agent.execute(weather_prompt, location="New York")
```

### Response Validation Flow

1. LLM generates a response based on the prompt
2. `ResponseFormat.validate_response()` checks if the response matches the expected format
3. If valid, the response is returned (potentially converted to a Pydantic model)
4. If invalid, the system can:
   - Return the validation errors
   - Automatically retry with error feedback
   - Apply fallback strategies

## Usage Examples

### Basic Prompt with Variables

```python
from llmaestro.prompts.base import BasePrompt

prompt = BasePrompt(
    name="Introduction",
    description="Introduce a topic to the user",
    system_prompt="You are a helpful assistant.",
    user_prompt="Explain {topic} in simple terms."
)

# Render the prompt with variables
system, user, examples, tools = prompt.render(topic="quantum computing")
```

### Prompt with Structured Output

```python
from llmaestro.prompts.base import BasePrompt
from llmaestro.llm.responses import ResponseFormat, ResponseFormatType
from pydantic import BaseModel

class SummaryResponse(BaseModel):
    main_points: list[str]
    conclusion: str

prompt = BasePrompt(
    name="Article Summary",
    description="Summarize an article",
    system_prompt="You are a summarization assistant.",
    user_prompt="Summarize the following article:\n{article}",
    expected_response=ResponseFormat.from_pydantic_model(
        SummaryResponse,
        format_type=ResponseFormatType.JSON_SCHEMA
    )
)
```

### Prompt with Tool Integration

```python
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.tools import ToolParams

def search_database(query: str) -> list[dict]:
    # Implementation
    return [{"title": "Result 1", "content": "..."}]

prompt = BasePrompt(
    name="Research Assistant",
    description="Research assistant with database access",
    system_prompt="You are a research assistant with database access.",
    user_prompt="Research the following topic: {topic}",
    tools=[ToolParams.from_function(search_database)]
)
```

## Integration Points

- **LLM Interfaces**: Prompts are executed through LLM interfaces
- **Agent System**: Agents use prompts to interact with LLMs
- **Orchestrator**: The orchestrator manages prompt execution and dependencies
- **Storage**: Prompts can be stored and loaded from various backends
