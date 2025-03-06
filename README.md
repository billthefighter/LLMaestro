# LLMaestro

A system for programmatically interacting with LLMs, with support for conversations, chains, and task orchestration.

It's like LiteLLM and LangChain, but worse!

At this point, it's been most robustly tested with openAI models, and the [default library](src/llmaestro/default_library/defined_providers/openai/provider.py) has the most models support - working on claude and google support.

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

### Core Components

- `ConversationGraph`: Manages conversation history with token usage tracking
- `ChainGraph`: Orchestrates complex workflows with different node types
- `ToolParams`: Provides type-safe function execution with safety checks
- `ConfigurationManager`: Manages system and user configuration with type safety
- `TaskManager`: Decomposes large tasks into smaller, manageable chunks
- `Agent`: Represents an LLM instance that can process subtasks
- `StorageManager`: Manages intermediate results using disk-based storage
- `Validator`: Ensures data consistency using Pydantic models

### Key Features

- Graph-based conversation management with token usage tracking
- Flexible chain orchestration with sequential, parallel, and conditional execution
- Type-safe function execution with safety guards
- Task decomposition strategies for different types of tasks
- Efficient resource management for parallel processing
- Model-agnostic design supporting different LLM providers
- Disk-based storage for handling large datasets
- Strong type validation using Pydantic

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
from llmaestro.prompts.tools import ToolParams, BasicFunctionGuard

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
