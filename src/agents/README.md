# Agent System

This directory contains the implementation of the agent system, which handles the execution of LLM tasks through a pool of managed agents.

## Overview

The agent system consists of two main components:
- `Agent`: Handles individual task execution using chain-based processing
- `AgentPool`: Manages a collection of agents and distributes tasks among them

## Agent

The `Agent` class is responsible for processing individual tasks using LLM chains. It supports both typed and untyped tasks, with special handling for JSON responses.

### Features

- Chain-based task processing
- Automatic JSON response parsing
- Error handling and retry logic
- Type-safe transformations
- Flexible prompt template usage

### Usage Example

```python
from llm_orchestrator.agents import Agent
from llm_orchestrator.core.models import AgentConfig, SubTask

# Create an agent
agent = Agent(AgentConfig(
    model_name="gpt-4",
    max_tokens=8192
))

# Create a task
task = SubTask(
    id="task-123",
    type="analysis",
    input_data={
        "text": "Content to analyze",
        "parameters": {"depth": "detailed"}
    }
)

# Process the task
result = await agent.process_task(task)
```

### Task Processing

Tasks are processed using a `SequentialChain` with the following workflow:

1. **Input Validation**:
   - Checks for valid input data format
   - Validates task type against available prompts

2. **Chain Creation**:
   ```python
   chain = SequentialChain[Dict[str, Any]](
       steps=[ChainStep[Dict[str, Any]](
           task_type=subtask.type,
           output_transform=self._create_json_parser() if json_expected else None
       )]
   )
   ```

3. **Response Processing**:
   - Automatic JSON parsing for structured responses
   - Error handling and formatting
   - Result standardization

## AgentPool

The `AgentPool` manages a collection of agents and handles task distribution.

### Features

- Dynamic agent creation
- Task queuing and distribution
- Concurrent task execution
- Resource management
- Automatic cleanup

### Usage Example

```python
from llm_orchestrator.agents import AgentPool

# Create a pool with max 5 agents
pool = AgentPool(max_agents=5)

# Submit tasks
pool.submit_task(task1)
pool.submit_task(task2)

# Get results
result1 = pool.wait_for_result(task1.id)
result2 = pool.wait_for_result(task2.id)
```

### Configuration

The pool can be configured with:
- `max_agents`: Maximum number of concurrent agents
- Agent-specific configurations through `AgentConfig`

Additionally, the pool now supports multiple agent types through the `AgentPoolConfig` system:

```python
# In your config.yaml
agents:
  max_agents: 10
  default_agent_type: "general"
  agent_types:
    general:
      model_name: "gpt-4"
      max_tokens: 8192
      description: "General purpose agent"
    fast:
      model_name: "gpt-3.5-turbo"
      max_tokens: 4096
      description: "Fast, lightweight agent for simple tasks"
    specialist:
      model_name: "claude-3-opus-20240229"
      max_tokens: 16384
      description: "Specialist agent for complex tasks"
```

You can request specific agent types when submitting tasks:

```python
# Get default agent
agent = pool.get_agent(task_id)

# Get specific type of agent
fast_agent = pool.get_agent(task_id, agent_type="fast")
specialist_agent = pool.get_agent(task_id, agent_type="specialist")
```

This configuration system allows you to:
- Define multiple agent types with different capabilities
- Set appropriate resource limits per agent type
- Match agent types to task requirements
- Maintain a default agent type for general tasks

## Best Practices

1. **Task Design**:
   - Use typed tasks with structured input/output when possible
   - Include clear task types that map to prompt templates
   - Consider response format requirements

2. **Resource Management**:
   - Monitor agent pool size
   - Clean up completed tasks
   - Handle long-running tasks appropriately

3. **Error Handling**:
   - Implement proper error handling in task processing
   - Use retry strategies for transient failures
   - Monitor and log task execution status

## Integration with Chains

The agent system now uses the chain system for task processing, providing:

1. **Structured Processing**:
   - Clear execution flow
   - Type-safe transformations
   - Built-in retry logic

2. **Extensibility**:
   - Custom transform support
   - Flexible prompt handling
   - Chain composition

3. **Type Safety**:
   ```python
   # Example of type-safe JSON handling
   class JsonOutputTransform(OutputTransform, Protocol):
       def __call__(self, response: LLMResponse) -> Dict[str, Any]: ...
   ```

## Future Improvements

1. **Enhanced Chain Integration**:
   - Support for more complex chain types
   - Dynamic chain composition
   - Chain result caching

2. **Resource Optimization**:
   - Smart agent pooling with automatic type selection
   - Dynamic load balancing based on agent types
   - Priority queuing with agent type preferences
   - Automatic scaling of agent pools based on task patterns

3. **Monitoring**:
   - Task execution metrics per agent type
   - Performance tracking across different agent configurations
   - Resource utilization stats by agent type
   - Cost optimization recommendations

4. **Agent Type Management**:
   - Dynamic agent type creation based on task patterns
   - Automatic agent type suggestion based on task characteristics
   - A/B testing of different agent configurations
   - Cost-performance optimization across agent types
