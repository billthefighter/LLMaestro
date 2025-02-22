# Session Management Module

## Overview

The `Session` class provides a centralized, conversation-centric management system for LLM (Large Language Model) interactions, designed to streamline complex AI workflows through orchestrated conversations, parallel execution, and dependency management.

## Key Features

### 1. Conversation Management
- Conversation-based workflow organization
- Dependency tracking between prompts
- Parallel execution support
- Rich conversation history and metadata

### 2. Orchestration
- Centralized execution coordination
- Resource management and allocation
- Parallel processing with controlled concurrency
- Execution status tracking

### 3. Model Management
- Model capability validation
- Dynamic interface creation
- Automatic resource allocation

## Session Creation

### Using the Factory Method (Recommended)
The recommended way to create a session is using the `create_default` factory method:

```python
from llmaestro.session.session import Session

# Create a fully initialized session
session = await Session.create_default(
    api_key="your-api-key",
    storage_path="./custom_storage"
)
# Session is ready to use immediately
```

The factory method supports several customization options:
- `api_key`: API key for the default provider
- `storage_path`: Custom storage location
- `config`: Custom configuration manager
- `llm_registry`: Custom LLM registry
- `prompt_loader`: Custom prompt loader
- `session_id`: Custom session identifier

Benefits of using `create_default`:
- Ensures proper initialization of all components
- Handles both synchronous and asynchronous setup
- Provides clear parameter documentation
- Returns a fully initialized session ready for use
- Maintains flexibility while providing convenience

### Manual Creation (Advanced)
For advanced use cases, you can create a session manually:

```python
session = Session(api_key="your-api-key")
await session.initialize()  # Don't forget this step!
```

Note: When creating a session manually, you MUST call `initialize()` before using the session.

## Basic Usage

### Starting a Conversation
```python
from llmaestro.prompts.base import BasePrompt

# Create initial prompt
setup_prompt = BasePrompt(
    name="initial_setup",
    system_prompt="You are a helpful assistant.",
    user_prompt="Let's begin our conversation."
)

# Start a conversation
conversation_id = await session.start_conversation(
    name="Analysis Session",
    initial_prompt=setup_prompt
)
```

### Executing Prompts with Dependencies
```python
# Execute sequential prompts with dependencies
response1_id = await session.execute_prompt(analysis_prompt)
response2_id = await session.execute_prompt(
    summary_prompt,
    dependencies=[response1_id]
)

# Execute prompts in parallel
response_ids = await session.execute_parallel(
    prompts=[prompt1, prompt2, prompt3],
    max_parallel=2
)

# Check execution status
status = session.get_execution_status(response1_id)

# Get conversation history
history = session.get_conversation_history(
    node_id=response2_id,
    max_depth=5
)
```

### Storing and Retrieving Artifacts
```python
# Store an artifact
artifact = session.store_artifact(
    name="analysis_result",
    data={"summary": "Analysis findings..."},
    content_type="json",
    metadata={"type": "analysis"}
)

# Retrieve artifacts
retrieved = session.get_artifact(artifact.id)
all_artifacts = session.list_artifacts()
```

## Advanced Features

### Parallel Processing
- Controlled concurrent execution
- Automatic resource management
- Progress tracking for parallel tasks
- Group status monitoring

### Dependency Management
- Automatic dependency resolution
- Execution order optimization
- Failure handling and recovery
- Conditional execution paths

### Conversation History
- Complete conversation tracking
- Rich metadata support
- Hierarchical history views
- Efficient storage and retrieval

## Best Practices
1. Use `create_default` for session creation unless you need advanced customization
2. Always provide API keys during session creation for seamless initialization
3. Use conversations to organize related prompts
4. Leverage dependencies for complex workflows
5. Use parallel execution for independent tasks
6. Monitor execution status for long-running operations
7. Store important artifacts for persistence

## Error Handling
- Comprehensive error tracking
- Automatic retry strategies
- Dependency failure management
- Resource cleanup on errors

## Performance Considerations
- Efficient parallel processing
- Smart resource allocation
- Minimal overhead for coordination
- Optimized conversation storage

## Dependencies
- Requires `pydantic`
- Uses `asyncio` for concurrency
- Integrates with `LLMRegistry`
- Depends on `FileSystemArtifactStorage`

## Contributing
Please read the project's contribution guidelines before submitting pull requests or reporting issues.
