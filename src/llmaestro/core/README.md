# LLMaestro Core

The core module provides the fundamental building blocks for the LLMaestro system, implementing the essential components for LLM orchestration, conversation management, and data handling.

## Core Components

### [models.py](./models.py)

Defines the base data models used throughout the system:

- `TokenUsage`: Tracks token consumption for LLM requests (prompt, completion, total tokens, cost)
- `ContextMetrics`: Monitors context window usage and limits
- `BaseResponse`: Abstract base class for all response types
- `LLMResponse`: Represents a response from an LLM model with content and usage metrics

### [graph.py](./graph.py)

Implements a generic graph data structure for representing relationships between components:

- `BaseNode`: Foundation for all graph nodes with ID, creation timestamp, and metadata
- `BaseEdge`: Represents directed connections between nodes with source, target, and relationship type
- `BaseGraph`: Generic graph implementation with methods for:
  - Adding/retrieving nodes and edges
  - Analyzing dependencies and execution order
  - Pruning nodes based on age or count
  - Generating graph summaries

### [conversations.py](./conversations.py)

Provides models for representing and managing conversation structures:

- `ConversationNode`: Represents a single node in a conversation (prompt or response)
- `ConversationEdge`: Directed edge between conversation nodes
- `ConversationGraph`: Graph-based representation of an LLM conversation with:
  - Token usage tracking and aggregation
  - Conversation history management
  - Node type filtering
- `ConversationContext`: Manages the current conversation state with history tracking

### [orchestrator.py](./orchestrator.py)

Manages the execution of LLM conversations and coordinates resources:

- `ExecutionMetadata`: Tracks execution status of nodes (pending, running, completed, failed)
- `Orchestrator`: Central controller that:
  - Manages active conversations
  - Handles prompt execution (sequential and parallel)
  - Coordinates dependencies between conversation nodes
  - Provides event callbacks for visualization
  - Tracks execution status and history

### [attachments.py](./attachments.py)

Handles file and image content for LLM interactions:

- `BaseAttachment`: Abstract base class for all file attachments
- `ImageAttachment`: Specialized attachment for images with validation
- `FileAttachment`: General-purpose file attachment
- `AttachmentConverter`: Utility for converting between attachment formats

### [storage.py](./storage.py)

Provides persistent storage for artifacts and conversation data:

- `Artifact`: Base model for all storable artifacts
- `ArtifactModel`: SQLAlchemy model for database storage
- `ArtifactStorage`: Abstract interface for storage implementations
- `DatabaseArtifactStorage`: SQL-based implementation
- `FileSystemArtifactStorage`: File system-based implementation with sync/async methods

### [logging_config.py](./logging_config.py)

Centralizes logging configuration for consistent formatting:

- `configure_logging`: Sets up logging with customizable levels, file output, and module naming
- Handles special cases for third-party libraries

## Usage Examples

```python
# Create an orchestrator with an agent pool
orchestrator = Orchestrator(agent_pool)

# Create a new conversation
conversation = await orchestrator.create_conversation(
    name="Example Conversation",
    initial_prompt=BasePrompt(content="Hello, how can I help you?")
)

# Execute a prompt and get the response
node_id = await orchestrator.execute_prompt(
    conversation=conversation,
    prompt=BasePrompt(content="Tell me about LLMs")
)

# Execute multiple prompts in parallel
node_ids = await orchestrator.execute_parallel(
    conversation=conversation,
    prompts=[
        BasePrompt(content="What are the benefits of LLMs?"),
        BasePrompt(content="What are the limitations of LLMs?")
    ],
    max_parallel=2
)

# Get conversation history
history = orchestrator.get_conversation_history(conversation=conversation)
```

## Integration Points

- **Agent System**: Core components integrate with the [agent system](../agents/README.md) for LLM interactions
- **Prompt System**: Uses the [prompt template system](../prompts/README.md) for structured LLM interactions
- **Visualization**: Provides hooks for visualizing conversation graphs and execution flow
- **Applications**: Serves as the foundation for [higher-level applications](../applications/README.md)
