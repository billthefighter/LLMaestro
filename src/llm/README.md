# LLM Module Documentation

This module provides the core LLM functionality, including the base interface for LLM interactions, automatic context management, summarization, and various chain implementations for orchestrating complex LLM tasks.

## Table of Contents
1. [Context Management](#context-management)
2. [Context Summarization](#context-summarization)
3. [Chain Implementations](#chain-implementations)
   - [ReminderChain](#reminderchain)
   - [RecursiveChain](#recursivechain)

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

## Prompt Integration

The LLM module now provides tight integration with the prompt system through the `process_prompt` method. This integration enables:
- Type-safe prompt handling
- Automatic media type validation
- Token estimation and rate limiting
- Structured file attachments

### Using Structured Prompts

```python
from src.prompts.base import BasePrompt
from src.llm.models import MediaType

# Create a prompt with file attachments
prompt = BasePrompt(
    name="image_analysis",
    description="Analyze an image with specific focus areas",
    system_prompt="You are an image analysis assistant.",
    user_prompt="Please analyze this image, focusing on: {focus_areas}",
    metadata=PromptMetadata(...)
)

# Add file attachments
with open("image.jpg", "rb") as f:
    prompt.add_attachment(
        content=f.read(),
        media_type=MediaType.JPEG,
        file_name="image.jpg",
        description="Input image to analyze"
    )

# Process the prompt with variables
response = await llm.process_prompt(
    prompt,
    variables={
        "focus_areas": "lighting, composition, subject placement"
    }
)
```

### Media Type Support

The system automatically validates media types against model capabilities:

```python
# Check if model supports a media type
if llm.supports_media_type(MediaType.PDF):
    # Add PDF attachment
    prompt.add_attachment(
        content=pdf_content,
        media_type=MediaType.PDF,
        file_name="document.pdf"
    )

# Validation happens automatically in process_prompt
try:
    response = await llm.process_prompt(prompt)
except ValueError as e:
    print(f"Media type not supported: {e}")
```

### Token Estimation

Prompts include built-in token estimation:

```python
# Estimate tokens before processing
estimates = prompt.estimate_tokens(
    model_family=llm.model_family,
    model_name=llm.config.model_name,
    variables={"focus_areas": "lighting, composition"}
)

print(f"Estimated tokens: {estimates['total_tokens']}")
print(f"Has variables: {estimates['has_variables']}")
print(f"Is estimate: {estimates['is_estimate']}")

# Validate context window
is_valid, error = prompt.validate_context(
    model_family=llm.model_family,
    model_name=llm.config.model_name,
    max_completion_tokens=1000
)
if not is_valid:
    print(f"Context too large: {error}")
```

### Handling Multiple Attachments

Prompts can handle multiple attachments with different media types:

```python
# Add multiple attachments
prompt.add_attachment(
    content=image1_content,
    media_type=MediaType.JPEG,
    file_name="image1.jpg"
)
prompt.add_attachment(
    content=image2_content,
    media_type=MediaType.PNG,
    file_name="image2.png"
)

# Clear attachments if needed
prompt.clear_attachments()
```

### Integration with Model Capabilities

The system automatically checks model capabilities:

```python
# Model capabilities are checked automatically
if llm.capabilities and llm.capabilities.supports_vision:
    # Add image attachments
    for image in images:
        prompt.add_attachment(
            content=image.content,
            media_type=MediaType.from_file_extension(image.path),
            file_name=image.name
        )

# Process with automatic capability validation
response = await llm.process_prompt(prompt)
```

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

## Chain Implementations

This section covers various chain implementations for orchestrating LLM interactions. Each chain type is designed for specific use cases and provides different features for managing complex LLM tasks.

### ReminderChain

The `ReminderChain` is designed to maintain context and consistency across multi-step LLM tasks. It automatically injects previous context and the original task description into each subsequent prompt, while intelligently managing the context size through pruning.

#### Key Features

- **Context Maintenance**: Automatically tracks and injects relevant context from previous steps
- **Smart Pruning**: Removes completed steps and maintains a manageable context size
- **Flexible Configuration**: Customizable reminder templates and pruning strategies
- **Type Safety**: Generic type parameters ensure type safety across the chain

#### Basic Usage Example

Here's a complete example of using a ReminderChain for a multi-step document analysis task:

```python
from llm_orchestrator.llm.chains import ReminderChain, ChainStep
from llm_orchestrator.llm.base import LLMResponse
from your_llm_implementation import YourLLM  # Replace with actual LLM implementation

# 1. Define the steps for document analysis
steps = [
    ChainStep[LLMResponse](
        task_type="extract_key_concepts",
        # Optional: Add custom input/output transforms
    ),
    ChainStep[LLMResponse](
        task_type="analyze_relationships",
    ),
    ChainStep[LLMResponse](
        task_type="generate_summary",
    )
]

# 2. Initialize the chain with custom settings
chain = ReminderChain[LLMResponse](
    steps=steps,
    llm=YourLLM(),  # Your LLM implementation
    # Customize how context is maintained
    reminder_template=(
        "Original Task: {initial_prompt}\n\n"
        "Progress so far:\n{context}\n\n"
        "Current Focus: {current_prompt}"
    ),
    # Enable context pruning
    prune_completed=True,
    completion_markers=["completed", "finished", "done"],
    max_context_items=2  # Keep only 2 most recent non-completed steps
)

# 3. Execute the chain
async def analyze_document():
    result = await chain.execute(
        prompt="Analyze this technical document about quantum computing, "
               "focusing on key concepts, relationships, and providing a summary. "
               "Maintain consistent terminology throughout the analysis."
    )

    # Access results
    print(f"Final Summary: {result.content}")

    # Access intermediate results if needed
    for i, step_result in enumerate(chain.context_history):
        print(f"Step {i+1}: {step_result.content}")
```

#### How It Works

1. **Step Execution**:
   - Each step in the chain is executed sequentially
   - The original prompt is preserved and injected into each step
   - Previous step results are maintained in the context history

2. **Context Management**:
   - The chain automatically tracks completed steps
   - Steps marked as "completed" are pruned from the context
   - Only the most recent N steps are kept (configurable)
   - Original task description is always preserved

3. **Customization Options**:
   - `reminder_template`: Format how context is presented in prompts
   - `prune_completed`: Enable/disable automatic context pruning
   - `completion_markers`: Words that indicate a step is complete
   - `max_context_items`: Limit on number of previous steps to maintain

### RecursiveChain

The `RecursiveChain` supports dynamic step generation and recursive processing, allowing chains to create new steps based on intermediate results. It includes built-in safeguards against infinite recursion and maintains a clear execution path.

#### Key Features

- **Dynamic Step Generation**: Generate new steps based on LLM responses
- **Recursion Control**: Built-in depth tracking and limits
- **Path Tracking**: Maintains clear execution path for debugging
- **Safe Execution**: Automatic cleanup of recursion state
- **Context Preservation**: Maintains context across recursive calls

#### Basic Usage Example

Here's an example of using a RecursiveChain for a task that may require follow-up steps:

```python
from llm_orchestrator.llm.chains import RecursiveChain, ChainStep
from llm_orchestrator.llm.base import LLMResponse
from typing import List

# 1. Define a step generator function
def generate_followup_steps(response: LLMResponse, context: ChainContext) -> List[ChainStep[LLMResponse]]:
    # Check if the response indicates need for follow-up
    if "needs_clarification" in response.content.lower():
        return [
            ChainStep[LLMResponse](
                task_type="clarification",
                input_transform=lambda ctx, **kwargs: {
                    "original_response": response.content,
                    "prompt": "Please clarify the following points..."
                }
            )
        ]
    return []  # No follow-up needed

# 2. Create the initial step
initial_step = ChainStep[LLMResponse](
    task_type="analysis",
    input_transform=lambda ctx, **kwargs: {
        "prompt": "Analyze this topic in detail..."
    }
)

# 3. Initialize the chain
chain = RecursiveChain[LLMResponse](
    initial_step=initial_step,
    step_generator=generate_followup_steps,
    max_recursion_depth=3  # Limit recursion depth
)

# 4. Execute the chain
async def analyze_with_followup():
    result = await chain.execute(
        topic="Complex topic that might need clarification"
    )
    print(f"Final result: {result.content}")

    # Access intermediate results if needed
    for step_id, step_result in chain.context.artifacts.items():
        print(f"Step {step_id}: {step_result.content}")
```

#### How It Works

1. **Initial Execution**:
   - The chain executes the initial step
   - Response is stored in artifacts
   - Step generator is called with the response

2. **Dynamic Step Generation**:
   - Step generator analyzes the response
   - Can return new steps based on the result
   - New steps are executed in a sequential chain

3. **Recursion Control**:
   - Depth is tracked in the context
   - Path is maintained for debugging
   - Automatic cleanup in finally block

4. **Context Management**:
   - Context is shared across all recursive calls
   - Artifacts are preserved throughout execution
   - State can be accessed between steps

## Future Improvements

1. **Additional Chain Types**
   - Sequential chains for step-by-step processing
   - Parallel chains for concurrent task execution
   - Chord chains for parallel processing with aggregation
   - Map chains for processing multiple inputs
   - Group chains for organizing complex workflows

2. **Advanced Use Cases**
   - Multi-document analysis with shared context
   - Dynamic step generation based on intermediate results
   - Integration with different LLM providers
   - Complex data transformation pipelines
   - Hybrid chains combining different chain types

3. **Developer Tools**
   - Chain visualization utilities
   - Execution monitoring and logging
   - Performance profiling tools
   - Testing utilities and fixtures
   - Chain composition helpers
