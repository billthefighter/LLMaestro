# Core Components

This directory contains the core components of the LLMaestro system. These components provide the foundational functionality for orchestration, conversation management, and resource coordination.

## Architecture Overview

The core system is built around several key components that work together:

1. **Orchestration** (`orchestrator.py`)
   - Manages conversation execution flow
   - Coordinates parallel processing
   - Handles dependency resolution
   - Manages resource allocation

2. **Conversation Management** (`conversations.py`)
   - Graph-based conversation representation
   - Tracks prompts and responses
   - Maintains conversation history
   - Supports conversation summarization

3. **Core Models** (`models.py`)
   - Base data structures and types
   - Execution metadata tracking
   - Token usage and metrics tracking
   - Response standardization

4. **Configuration Management** (`config.py`)
   - System and user configuration
   - Model registry integration
   - Environment variable support
   - Type-safe configuration access

## Conversation-Centric System

The system is designed around conversations as the primary unit of organization, with built-in support for parallel execution and dependency management.

### Conversation Management

```python
from llmaestro.core.orchestrator import Orchestrator
from llmaestro.core.conversations import ConversationGraph
from llmaestro.prompts.base import BasePrompt

# Create an orchestrator
orchestrator = Orchestrator(agent_pool)

# Start a conversation
conversation = await orchestrator.create_conversation(
    name="analysis_session",
    initial_prompt=setup_prompt
)

# Execute prompts with dependencies
response1_id = await orchestrator.execute_prompt(
    conversation_id=conversation.id,
    prompt=analysis_prompt
)

response2_id = await orchestrator.execute_prompt(
    conversation_id=conversation.id,
    prompt=summary_prompt,
    dependencies=[response1_id]
)

# Execute prompts in parallel
response_ids = await orchestrator.execute_parallel(
    conversation_id=conversation.id,
    prompts=[prompt1, prompt2, prompt3],
    max_parallel=2
)
```

### Execution Features

The orchestration system provides:
- Parallel execution with controlled concurrency
- Automatic dependency resolution
- Execution status tracking
- Resource management and allocation
- Error handling and recovery

### Conversation Integration

All interactions are managed through the conversation system:
- Each prompt creates a conversation node
- Responses are linked to their prompts
- Dependencies are tracked as edges
- Full conversation history is maintained
- Support for summarization and context management

```python
# Access conversation history
history = conversation.get_node_history(node_id)
summary = conversation.get_conversation_summary()

# Check execution status
status = orchestrator.get_execution_status(conversation_id, node_id)
```

## Configuration System

The configuration system remains split into system and user configurations:

1. **System Configuration** (`system_config.yml`):
   - Provider definitions and capabilities
   - Model specifications and features
   - Rate limits and API endpoints

2. **User Configuration** (`config.yaml`):
   - API keys and credentials
   - Default model settings
   - Agent pool configuration
   - Storage preferences

### Usage Examples

```python
from llmaestro.core.config import get_config

# Get the configuration manager
config = get_config()

# Access user settings
api_key = config.user_config.api_keys["anthropic"]
storage_path = config.user_config.storage["path"]

# Access system settings
provider_config = config.system_config.providers["anthropic"]
model_capabilities = provider_config.models["claude-3-sonnet-20240229"]

# Get combined model config
model_config = config.get_model_config("anthropic", "claude-3-sonnet-20240229")
```

## Async Support

The system is built with async support throughout:

```python
async def process_conversation():
    # Create conversation
    conversation = await orchestrator.create_conversation(...)

    # Execute prompts
    response_id = await orchestrator.execute_prompt(
        conversation_id=conversation.id,
        prompt=analysis_prompt
    )

    # Execute parallel prompts
    response_ids = await orchestrator.execute_parallel(
        conversation_id=conversation.id,
        prompts=[prompt1, prompt2]
    )

    # Get execution status
    status = orchestrator.get_execution_status(conversation.id, response_id)
```

## Best Practices

1. **Conversation Management**:
   - Use conversations to organize related prompts
   - Track dependencies between prompts
   - Monitor execution status
   - Handle errors appropriately

2. **Resource Management**:
   - Control parallel execution limits
   - Monitor resource utilization
   - Clean up resources properly
   - Use appropriate retry strategies

3. **Configuration**:
   - Never commit API keys to version control
   - Use environment variables in production
   - Keep `config.yaml` in `.gitignore`
   - Use type-safe configuration access

4. **Type Safety**:
   - Use proper type hints throughout
   - Validate inputs at boundaries
   - Handle async operations correctly
   - Use standardized response types

## Module Dependencies

```
core/
├── orchestrator.py    # Execution coordination
├── conversations.py   # Conversation graph management
├── models.py         # Core data structures
├── config.py         # Configuration management
└── logging_config.py # Logging configuration
```

Key relationships:
- `orchestrator.py` manages execution flow and resource coordination
- `conversations.py` provides the conversation graph structure
- All modules use types from `models.py`
- Configuration from `config.py` is used throughout
- Logging is configured via `logging_config.py`
