# LLM Interface

This module provides the base interface and implementations for LLM interactions, including automatic context management and summarization.

## Context Management

### Task Reminders

The system can periodically remind the LLM of the initial task context to maintain focus and consistency throughout long conversations. This is particularly useful for complex tasks that require multiple interactions.

Configure reminders through the `SummarizationConfig`:

```python
config = AgentConfig(
    model_name="gpt-4",
    max_tokens=1000,
    summarization=SummarizationConfig(
        reminder_frequency=5,  # Remind every 5 messages
        reminder_template="Initial Task Context: {initial_task}"
    )
)

# Create LLM interface
llm = create_llm_interface(config)

# Set the initial task
llm.set_initial_task("Refactor the user authentication system to use JWT tokens")
```

#### Reminder Settings

- `reminder_frequency`: Number of messages between reminders (default: 5, set to 0 to disable)
- `reminder_template`: Template for formatting the reminder message
- Reminders are automatically inserted as system messages
- The initial task is always preserved during context summarization

## Context Summarization

The system automatically manages conversation context and handles token limits through intelligent summarization. When the context window approaches its limit, the system will automatically summarize previous interactions while preserving key information.

### Configuration

Configure summarization through the `AgentConfig`:

```python
config = AgentConfig(
    model_name="gpt-4",
    max_tokens=1000,
    max_context_tokens=8192,
    summarization=SummarizationConfig(
        enabled=True,
        target_utilization=0.8,  # Summarize at 80% context usage
        min_tokens_for_summary=1000,
        preserve_last_n_messages=3
    )
)
```

### Summarization Settings

- `enabled`: Toggle automatic summarization (default: True)
- `target_utilization`: Context window usage threshold for triggering summarization (default: 0.8)
- `min_tokens_for_summary`: Minimum tokens before considering summarization (default: 1000)
- `preserve_last_n_messages`: Number of recent messages to preserve without summarization (default: 3)

### How It Works

1. **Context Monitoring**
   - Tracks token usage for each interaction
   - Monitors context window utilization
   - Preserves conversation history
   - Maintains initial task context

2. **Automatic Summarization**
   - Triggers when context utilization exceeds target threshold
   - Preserves specified number of recent messages
   - Summarizes earlier context while maintaining key information
   - Updates context with summary + preserved messages
   - Retains initial task context for reminders

3. **Summary Format**
   The summary includes:
   - Concise overview of key points
   - Important decisions and their rationale
   - Current task progress and state
   - Planned next steps
   - Token usage metrics

### Usage Example

```python
# Create LLM interface with summarization enabled
llm = create_llm_interface(config)

# Set initial task
llm.set_initial_task("Implement a new feature X")

# Process requests normally - summarization and reminders happen automatically
response = await llm.process("Your prompt")

# Access summarization info if available
if response.metadata.get("context_summary"):
    print("Context was summarized:")
    print(f"Summary: {response.metadata['context_summary']['summary']}")
    print(f"Progress: {response.metadata['context_summary']['state']['progress']}")
    print(f"Next steps: {response.metadata['context_summary']['state']['next_steps']}")

# Monitor context metrics
if response.context_metrics:
    print(f"Context utilization: {response.context_metrics.context_utilization * 100:.1f}%")
    print(f"Available tokens: {response.context_metrics.available_tokens}")
    print(f"Messages processed: {response.metadata['message_count']}")
```

### Token Usage Tracking

The system also tracks token usage and optionally calculates costs:

```python
# Enable cost tracking
config = AgentConfig(
    model_name="gpt-4",
    max_tokens=1000,
    cost_per_1k_tokens=0.03  # $0.03 per 1k tokens
)

# Access token usage
if response.token_usage:
    print(f"Used {response.token_usage.total_tokens} tokens")
    print(f"Estimated cost: ${response.token_usage.estimated_cost}")

# Get cumulative usage
total_used = llm.total_tokens_used
print(f"Total tokens used in this session: {total_used}")
``` 