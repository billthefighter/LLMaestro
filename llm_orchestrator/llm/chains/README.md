# Chain Examples

This directory contains implementations of various prompt chain types for orchestrating LLM interactions. Below are examples of how to use them effectively.

## ReminderChain

The `ReminderChain` is designed to maintain context and consistency across multi-step LLM tasks. It automatically injects previous context and the original task description into each subsequent prompt, while intelligently managing the context size through pruning.

### Key Features

- **Context Maintenance**: Automatically tracks and injects relevant context from previous steps
- **Smart Pruning**: Removes completed steps and maintains a manageable context size
- **Flexible Configuration**: Customizable reminder templates and pruning strategies
- **Type Safety**: Generic type parameters ensure type safety across the chain

### Basic Usage Example

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

### How It Works

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

### Best Practices

1. **Context Management**:
   - Set appropriate `max_context_items` based on your token limits
   - Use clear completion markers in your prompts
   - Consider task-specific reminder templates

2. **Step Design**:
   - Break complex tasks into focused steps
   - Use consistent terminology across steps
   - Consider adding input/output transforms for complex data

3. **Error Handling**:
   - Implement retry strategies for unreliable steps
   - Store intermediate results using the built-in artifact storage
   - Monitor context size to avoid token limits

### Advanced Features

The ReminderChain supports several advanced features:

1. **Custom Transforms**:
   ```python
   def input_transform(context: ChainContext, **kwargs) -> Dict[str, Any]:
       return {
           "prompt": kwargs["prompt"],
           "additional_context": process_context(context)
       }

   step = ChainStep[LLMResponse](
       task_type="analysis",
       input_transform=input_transform
   )
   ```

2. **Artifact Storage**:
   ```python
   # Results are automatically stored
   chain.get_artifact(f"step_{step.id}")  # Retrieve step result
   ```

3. **Custom Pruning**:
   ```python
   chain = ReminderChain(
       steps=steps,
       completion_markers=["analysis complete", "section finished"],
       max_context_items=3
   )
   ```

### Future Improvements

The following enhancements are planned for future releases:

1. **Additional Chain Examples**
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

3. **Troubleshooting Guide**
   - Common error patterns and solutions
   - Performance optimization tips
   - Token limit management strategies
   - Debugging techniques for chain execution
   - Best practices for error recovery

4. **Developer Tools**
   - Chain visualization utilities
   - Execution monitoring and logging
   - Performance profiling tools
   - Testing utilities and fixtures
   - Chain composition helpers

## RecursiveChain

The `RecursiveChain` supports dynamic step generation and recursive processing, allowing chains to create new steps based on intermediate results. It includes built-in safeguards against infinite recursion and maintains a clear execution path.

### Key Features

- **Dynamic Step Generation**: Generate new steps based on LLM responses
- **Recursion Control**: Built-in depth tracking and limits
- **Path Tracking**: Maintains clear execution path for debugging
- **Safe Execution**: Automatic cleanup of recursion state
- **Context Preservation**: Maintains context across recursive calls

### Basic Usage Example

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

### How It Works

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

### Best Practices

1. **Step Generation**:
   - Keep generator logic simple and deterministic
   - Include clear conditions for termination
   - Consider using response metadata for control

2. **Depth Control**:
   - Set appropriate max_recursion_depth
   - Use context state to track progress
   - Implement fallbacks for depth limits

3. **Error Handling**:
   - Handle RecursionError appropriately
   - Preserve partial results on failure
   - Log execution paths for debugging

### Advanced Features

1. **Custom Step Generation**:
   ```python
   def complex_generator(response: LLMResponse, context: ChainContext) -> List[ChainStep]:
       # Access previous results
       previous_steps = [
           result for result in context.artifacts.values()
           if isinstance(result, LLMResponse)
       ]
       
       # Make decisions based on history
       if len(previous_steps) < 3 and needs_refinement(response):
           return [create_refinement_step(previous_steps)]
       return []
   ```

2. **State Management**:
   ```python
   # Store state in context
   context.state["visited_topics"] = set()
   
   def state_aware_generator(response: LLMResponse, context: ChainContext):
       # Use state to avoid revisiting topics
       topics = extract_topics(response)
       new_topics = topics - context.state["visited_topics"]
       context.state["visited_topics"].update(new_topics)
       
       return [create_step_for_topic(topic) for topic in new_topics]
   ```

3. **Conditional Recursion**:
   ```python
   def conditional_generator(response: LLMResponse, context: ChainContext):
       if context.recursion_depth >= 2:
           # Switch to simpler strategy at deeper levels
           return create_simple_steps(response)
       # Use complex strategy at shallow levels
       return create_complex_steps(response)
   ```

### Future Improvements

1. **Enhanced Control**:
   - Support for recursion breadth limits
   - Conditional depth limits by branch
   - Priority-based step execution

2. **Performance**:
   - Caching of repeated sub-chains
   - Parallel execution of independent branches
   - Resource-aware step generation

3. **Monitoring**:
   - Detailed execution graphs
   - Performance metrics by depth
   - Resource usage tracking

4. **Error Handling**:
   - Implement error handling for recursive calls
   - Add logging for debugging and monitoring
   - Consider integration with monitoring tools

5. **Resource Management**:
   - Optimize resource usage for deep recursion
   - Implement lazy evaluation for large chains
   - Consider integration with resource management tools 