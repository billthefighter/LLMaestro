# Core Components

This directory contains the core components of the LLMaestro system. These components provide the foundational functionality for configuration management, task handling, conversation management, and logging.

## Architecture Overview

The core system is built around several key components that work together:

1. **Task Management** (`task_manager.py`)
   - Handles task decomposition and execution
   - Integrates with conversation tracking
   - Supports multiple decomposition strategies
   - Manages async task processing

2. **Conversation Management** (`conversations.py`)
   - Graph-based conversation representation
   - Tracks prompts and responses
   - Maintains conversation history
   - Supports conversation summarization

3. **Core Models** (`models.py`)
   - Base data structures and types
   - Task and subtask definitions
   - Token usage and metrics tracking
   - Response standardization

4. **Configuration Management** (`config.py`)
   - System and user configuration
   - Model registry integration
   - Environment variable support
   - Type-safe configuration access

## Task Management System

The task management system is designed to handle complex, decomposable tasks with conversation tracking.

### Task Decomposition Strategies

```python
from llmaestro.core.task_manager import TaskManager, Task
from llmaestro.core.conversations import ConversationGraph

# Create a task manager
task_manager = TaskManager()

# Create and execute a task with conversation tracking
task = task_manager.create_task(
    task_type="code_review",
    input_data=review_prompt,
    config={"max_parallel": 3}
)

# Execute with conversation tracking
conversation = ConversationGraph(id="review-session")
result = await task_manager.execute(task, conversation)
```

Available strategies:
- **Chunk Strategy**: Breaks down large inputs into manageable chunks
- **File Strategy**: Processes multiple files in parallel
- **Error Strategy**: Handles multiple error cases
- **Dynamic Strategy**: Generates custom strategies at runtime

### Conversation Integration

Tasks are automatically integrated with the conversation system:
- Each subtask creates prompt and response nodes
- Relationships between nodes are tracked
- Full conversation history is maintained
- Support for summarization and context management

```python
# Access conversation history
history = conversation.get_node_history(node_id)
summary = conversation.get_conversation_summary()
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
async def process_task():
    # Create and configure task
    task = task_manager.create_task(...)

    # Execute with conversation tracking
    conversation = ConversationGraph(...)
    result = await task_manager.execute(task, conversation)

    # Wait for specific subtask results
    subtask_result = await task_manager.wait_for_result(subtask_id)
```

## Best Practices

1. **Task Management**:
   - Use appropriate decomposition strategies
   - Always provide conversation tracking
   - Handle task results asynchronously
   - Clean up resources properly

2. **Conversation Handling**:
   - Use conversation graphs for all interactions
   - Implement proper error handling
   - Consider context window limitations
   - Use summarization when appropriate

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
├── task_manager.py     # Task handling and decomposition
├── conversations.py    # Conversation graph management
├── models.py          # Core data structures
├── config.py          # Configuration management
└── logging_config.py  # Logging configuration
```

Key relationships:
- `task_manager.py` depends on `conversations.py` for tracking
- All modules use types from `models.py`
- Configuration from `config.py` is used throughout
- Logging is configured via `logging_config.py`
