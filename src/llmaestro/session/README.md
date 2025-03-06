# Session Management Module

## Overview

The `Session` class provides a centralized, conversation-centric management system for LLM (Large Language Model) interactions, designed to streamline complex AI workflows through orchestrated conversations, parallel execution, and dependency management.

## Session vs. Orchestrator: When to Use Each

### Session
The `Session` class is a high-level interface that:
- Provides a complete, ready-to-use environment for LLM interactions
- Manages the entire lifecycle of LLM conversations
- Handles initialization of all required components (LLM registry, agent pool, storage, etc.)
- Offers artifact storage and retrieval capabilities
- Validates model capabilities against task requirements
- Provides both synchronous and asynchronous APIs

**Use Session when:**
- You need a complete, self-contained system for LLM interactions
- You want simplified setup with minimal configuration
- You need artifact storage and retrieval
- You're building end-user applications
- You want a higher-level abstraction that manages all components

### Orchestrator
The `Orchestrator` class is a lower-level component that:
- Focuses specifically on managing conversation execution
- Handles the mechanics of prompt execution and dependencies
- Provides fine-grained control over conversation flow
- Manages parallel execution and dependencies between prompts
- Requires an agent pool to be provided

**Use Orchestrator when:**
- You need direct control over conversation execution
- You're building custom orchestration logic
- You want to integrate with your own storage or agent management
- You're extending the core functionality
- You need fine-grained control over execution flow

The Session internally uses an Orchestrator to manage conversations, providing a higher-level interface that handles initialization, storage, and other concerns.

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

### 4. Artifact Storage
- Persistent storage of conversation artifacts
- Structured metadata and retrieval
- Support for various content types

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
- `llm_registry`: Custom LLM registry
- `default_model`: Specify a default model to use
- `default_capabilities`: Set required capabilities
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

### Model Validation
- Capability-based model selection
- Requirement validation against model capabilities
- Automatic fallback strategies

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
