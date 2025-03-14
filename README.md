# LLMaestro

A system for programmatically interacting with LLMs, with support for conversations, chains, and task orchestration.

It's like LiteLLM and LangChain, but worse!

## Quickstart

```python
import asyncio
from llmaestro.default_library.default_llm_factory import LLMDefaultFactory
from llmaestro.agents.agent_pool import AgentPool
from llmaestro.prompts.base import PromptVariable
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.tools.core import ToolParams, BasicFunctionGuard
from llmaestro.prompts.types import SerializableType

# Define a simple weather function
def get_weather(location: str) -> str:
    """Get current temperature for a given location."""
    return f"The weather in {location} is currently sunny with a temperature of 72°F."

async def main():
    # Initialize the LLM factory and get the registry
    factory = LLMDefaultFactory()
    registry = await factory.DefaultLLMRegistryFactory()

    # Create an agent pool with the default LLMs
    agent_pool = AgentPool(llm_registry=registry)

    # Create a prompt with a tool
    weather_prompt = MemoryPrompt(
        name="weather_query",
        description="Query weather information using tools",
        system_prompt="You are a helpful weather assistant. Use the provided tools to get weather information.",
        user_prompt="What is the weather like in {location} today?",
        variables=[
            PromptVariable(name="location", description="The location to get weather for", expected_input_type=SerializableType.STRING)
        ],
        tools=[
            ToolParams.from_function(get_weather)
        ]
    )

    # Get user input for location
    location = "San Francisco"

    # Render the prompt with the location variable
    system_prompt, user_prompt, _, tools = weather_prompt.render(location=location)

    # Create a new prompt with the rendered content
    formatted_prompt = MemoryPrompt(
        name=weather_prompt.name,
        description=weather_prompt.description,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=tools
    )

    # Execute the prompt with the agent pool
    response = await agent_pool.execute_prompt(
        prompt=formatted_prompt
    )

    # Handle the response
    print(f"Response: {response.content}")
    print(f"Token usage: {response.token_usage}")

    # If the LLM used the tool, the result will be in the metadata
    if response.metadata and "tool_results" in response.metadata:
        print("Tool results:")
        for tool_result in response.metadata["tool_results"]:
            print(f"  Tool: {tool_result.get('name', 'unknown')}")
            print(f"  Result: {tool_result.get('result', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(main())
```

For a complete working example with API key handling and model selection, see [examples/weather_tool_example.py](examples/weather_tool_example.py). To run the example, you'll need to set the `OPENAI_API_KEY` environment variable with your OpenAI API key.

At this point, it's been most robustly tested with openAI models, and the [default library](src/llmaestro/default_library/defined_providers/openai/provider.py) has the most models support - working on claude and google support.

## Capability-Based Model Selection

LLMaestro provides a powerful capability-based model selection system that allows you to dynamically select the most appropriate model based on required capabilities rather than hardcoding specific model names.

### Key Benefits

1. **Cost Optimization**: Automatically select the cheapest model that meets your requirements
2. **Future-Proofing**: Your code won't break when model names change or new models are introduced
3. **Flexibility**: Easily adapt to different environments and available models
4. **Testability**: Tests can run with any model that supports the required capabilities

### Usage Examples

```python
from llmaestro.llm.llm_registry import LLMRegistry

# Find the cheapest model that supports vision capabilities
required_capabilities = {"supports_vision"}
model_name = llm_registry.find_cheapest_model_with_capabilities(required_capabilities)

# Create an instance of the selected model
llm_instance = await llm_registry.create_instance(model_name)
```

You can also pass capabilities directly to the AgentPool:

```python
# Execute a prompt with specific capability requirements
response = await agent_pool.execute_prompt(
    prompt=formatted_prompt,
    required_capabilities={"supports_function_calling", "supports_tools"}
)
```

For more details, see the [Capability-Based Model Selection](docs/capability_based_model_selection.md) documentation.

## Model Status

### Anthropic Models
![claude-3-5-sonnet-latest](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/claude-3-5-sonnet-latest.json) - Balanced Claude 3 model offering strong performance at a lower cost than Opus
![claude-3-haiku-latest](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/claude-3-haiku-latest.json) - Fastest and most cost-effective Claude 3 model, optimized for simpler tasks
![claude-3-opus-latest](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/claude-3-opus-latest.json) - Most capable Claude 3 model, best for complex tasks requiring deep analysis, coding, and reasoning

### OpenAI Models
![gpt-3.5-turbo](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/gpt-3.5-turbo.json) - Fast and cost-effective GPT-3.5 model for most tasks
![gpt-4-turbo-preview](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/gpt-4-turbo-preview.json) - Most capable and up-to-date GPT-4 model with 128k context

### Google Models
![gemini-1.5-flash](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/gemini-1.5-flash.json)
![gemini-pro](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/gemini-pro.json)

## Getting Started

## Documentation Index

The following README files provide detailed documentation for different components of the system:

### Core Components
- [Core System](src/core/README.md) - Core system components including configuration management
- [Agent System](src/agents/README.md) - Documentation for the agent system and agent pool
- [LLM Core](src/llm/README.md) - Core LLM functionality and chain implementations
- [LLM Interfaces](src/llm/interfaces/README.md) - LLM provider interface implementations
- [LLM Models](src/llm/models/README.md) - Model configurations and capabilities
- [Prompts](src/prompts/README.md) - Prompt template system and management
- [Conversations](src/llmaestro/core/conversations.py) - Graph-based conversation management
- [Chains](src/llmaestro/chains/chains.py) - Graph-based chain orchestration system
- [Tools](src/llmaestro/prompts/tools.py) - Function execution and tool management

### Applications
- [Applications Overview](src/applications/README.md) - Guide to creating and structuring applications
- [PDFReader](src/applications/pdfreader/README.md) - PDF data extraction application

### Visualization and Configuration
- [Visualization](src/visualization/README.md) - Chain visualization tools and patterns
- [Configuration](config/README.md) - User configuration guide and examples

## Architecture

The system is built around these main concepts:

1. **Conversations**: Graph-based representation of LLM interactions with token tracking
2. **Chains**: Orchestration of complex LLM workflows with different node types
3. **Tools**: Type-safe function execution with safety guards
4. **Agent Pool**: Manages a pool of LLM agents that can be assigned subtasks
5. **Storage Manager**: Handles efficient storage and retrieval of intermediate results
6. **Persistence**: Automatic database persistence for models and graph structures

### Core Components

- `ConversationGraph`: Manages conversation history with token usage tracking
- `ChainGraph`: Orchestrates complex workflows with different node types
- `ToolParams`: Provides type-safe function execution with safety checks
- `ConfigurationManager`: Manages system and user configuration with type safety
- `TaskManager`: Decomposes large tasks into smaller, manageable chunks
- `Agent`: Represents an LLM instance that can process subtasks
- `StorageManager`: Manages intermediate results using disk-based storage
- `Validator`: Ensures data consistency using Pydantic models
- `PersistentModel`: Base class providing automatic database persistence for models

### Key Features

- Graph-based conversation management with token usage tracking
- Flexible chain orchestration with sequential, parallel, and conditional execution
- Type-safe function execution with safety guards
- Task decomposition strategies for different types of tasks
- Efficient resource management for parallel processing
- Model-agnostic design supporting different LLM providers
- Disk-based storage for handling large datasets
- Strong type validation using Pydantic
- Automatic database persistence for models and graph structures
- Resumable operations through persistent state management

## Persistence

The system uses a persistence layer built on Pydantic models to automatically handle database storage and retrieval:

```python
from llmaestro.core.persistence import PersistentModel
from llmaestro.chains.chains import ChainGraph, ChainMetadata

# Create a chain with metadata
chain = ChainGraph()
metadata = ChainMetadata(
    description="Example chain",
    tags={"example", "demo"},
    version="1.0"
)

# All models inheriting from PersistentModel are automatically persisted
# This includes:
# - Chain graphs and their components
# - Conversation graphs and nodes
# - Chain metadata and state
# - Execution history and results

# The persistence layer allows for:
# - Resuming interrupted operations
# - Loading saved chain configurations
# - Tracking execution history
# - Managing long-running conversations
```

## Conversations

The conversation system provides a graph-based approach to managing LLM interactions:

```python
from llmaestro.core.conversations import ConversationContext
from llmaestro.prompts.base import BasePrompt

# Create a conversation context
context = ConversationContext()

# Add a prompt node
prompt = BasePrompt(
    system_prompt="You are a helpful assistant.",
    user_prompt="What is the capital of France?"
)
prompt_id = context.add_node(prompt, "prompt")

# Add a response node
response_id = context.add_node(llm_response, "response")

# Track token usage
print(f"Total tokens: {context.total_tokens}")
print(f"Prompt tokens: {context.prompt_tokens}")
print(f"Response tokens: {context.response_tokens}")
```

## Chains

The chain system enables complex LLM workflows with different node types:

```python
from llmaestro.chains.chains import ChainGraph, ChainStep, NodeType
from llmaestro.prompts.base import BasePrompt

# Create chain steps
step1 = ChainStep(
    prompt=BasePrompt(
        system_prompt="You are a helpful assistant.",
        user_prompt="Summarize this text: {text}"
    )
)

step2 = ChainStep(
    prompt=BasePrompt(
        system_prompt="You are a helpful assistant.",
        user_prompt="Translate this summary to French: {summary}"
    )
)

# Create chain graph
chain = ChainGraph()

# Add nodes
node1_id = chain.add_node(ChainNode(step=step1, node_type=NodeType.SEQUENTIAL))
node2_id = chain.add_node(ChainNode(step=step2, node_type=NodeType.SEQUENTIAL))

# Add edge
chain.add_edge(ChainEdge(source_id=node1_id, target_id=node2_id))

# Execute chain
results = await chain.execute(text="This is a sample text to summarize.")
```

## Tools

The tools system provides type-safe function execution with safety guards:

```python
from llmaestro.tools.core import ToolParams, BasicFunctionGuard

# Define a function
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle"""
    return length * width

# Create a tool
tool = ToolParams.from_function(calculate_area)

# Execute the tool
result = await tool.execute(length=5, width=3)
print(f"Area: {result}")  # Output: Area: 15.0

# Create a custom function guard
class SafeFileGuard(BasicFunctionGuard):
    def __init__(self, func, allowed_paths=None):
        super().__init__(func)
        self.allowed_paths = allowed_paths or []

    def is_safe_to_run(self, **kwargs):
        if "path" in kwargs:
            return any(kwargs["path"].startswith(p) for p in self.allowed_paths)
        return super().is_safe_to_run(**kwargs)
```

The tools system is tightly integrated with the [prompt system](src/llmaestro/prompts/README.md), allowing LLMs to execute functions safely through prompts. This integration enables natural language interfaces to your code while maintaining type safety and security.

### Tool-Prompt Integration

Tools can be attached to prompts, allowing LLMs to decide when and how to call them:

```python
from llmaestro.prompts.base import PromptVariable
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.tools.core import ToolParams, BasicFunctionGuard
from llmaestro.prompts.types import SerializableType

# Define a function
def get_weather(location: str) -> str:
    """Get current temperature for a given location."""
    return f"The weather in {location} is always sunny."

# Create a prompt with the tool
weather_prompt = MemoryPrompt(
    name="weather_query",
    description="Query weather information using tools",
    system_prompt="You are a weather assistant.",
    user_prompt="What is the weather like in {location} today?",
    variables=[
        PromptVariable(
            name="location",
            description="The location to get weather for",
            expected_input_type=SerializableType.STRING
        )
    ],
    tools=[
        ToolParams.from_function(get_weather)
    ]
)

# Render the prompt with variables
location = "New York"
system_prompt, user_prompt, _, tools = weather_prompt.render(location=location)

# Create a new prompt with the rendered content
formatted_prompt = MemoryPrompt(
    name=weather_prompt.name,
    description=weather_prompt.description,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    tools=tools
)

# Execute the prompt with an LLM
response = await agent.execute_prompt(prompt=formatted_prompt)
```

### Key Features of the Tools System

- **Type Safety**: Automatic parameter validation based on type hints
- **Security**: Function guards prevent unsafe operations
- **Automatic Schema Generation**: JSON schemas are generated from Python type hints
- **Pydantic Integration**: Seamless support for Pydantic models as tools
- **Provider Compatibility**: Tools work with OpenAI, Anthropic, and other providers
- **Custom Guards**: Create custom guards for specific security requirements

See the [prompts/README.md](src/llmaestro/prompts/README.md) for more details on the prompt system and its integration with tools.

## Applications

The framework includes several example applications demonstrating common use cases and best practices. See the [applications directory](src/applications/) for detailed documentation on creating and structuring applications.

| Application | Description | Key Features |
|------------|-------------|--------------|
| [PDFReader](src/applications/pdfreader/) | Extract structured data from PDFs using vision capabilities | - Multi-page processing<br>- Schema-based extraction<br>- Confidence scoring |

Each application follows a standardized structure and includes:
- Comprehensive documentation
- YAML-based prompt templates
- Pydantic data models
- Unit and integration tests
