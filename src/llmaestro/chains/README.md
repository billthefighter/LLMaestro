# LLMaestro Chains

The chains module provides a flexible, graph-based system for orchestrating complex LLM interactions. It allows you to build pipelines of prompts with validation, retries, and conditional execution.

## Core Components

### Chain Graph
The foundation of the system is the `ChainGraph`, which represents a directed acyclic graph (DAG) of operations:

```python
chain = ChainGraph(agent_pool=agent_pool)
chain.add_node(node)
chain.add_edge(ChainEdge(source_id=node1.id, target_id=node2.id))
```

### Node Types
- `SEQUENTIAL`: Standard sequential execution
- `PARALLEL`: Concurrent execution
- `CONDITIONAL`: Branching based on conditions
- `AGENT`: LLM interaction nodes
- `VALIDATION`: Response validation and retry logic

### Chain Steps
Each node contains a `ChainStep` that defines its behavior:

```python
step = ChainStep(
    prompt=my_prompt,
    input_transform=lambda ctx, data: {"var": f"transformed_{data['var']}"},
    output_transform=lambda response: response.content.upper(),
    retry_strategy=RetryStrategy(max_retries=3)
)
```

## Validation System

The chains module includes a robust validation system for LLM responses:

```python
validation_node = ValidationNode(
    response_format=ResponseFormat(
        format=ResponseFormatType.JSON,
        schema='{"type": "object", ...}'
    ),
    retry_strategy=RetryStrategy(
        max_retries=3,
        exponential_backoff=True
    )
)
```

### Features
- Schema validation for structured outputs
- Automatic retries with customizable strategies
- Context preservation during retries
- Custom error handling

## Common Recipes

### 1. JSON Validation Chain
```python
from llmaestro.chains.recipes import create_validation_chain

chain = await create_validation_chain(
    prompt=json_prompt,
    response_format=ResponseFormat(
        format=ResponseFormatType.JSON,
        schema=my_schema
    ),
    agent_pool=agent_pool
)
```

### 2. Structured Conversation Chain
```python
chain = await create_validation_chain(
    prompt=conversation_prompt,
    response_format=prompt.metadata.expected_response,
    agent_pool=agent_pool,
    preserve_context=True
)

async def execute_turn(user_input: str):
    return await chain.execute(user_input=user_input)
```

## Retry Strategies

Configure how nodes handle failures:

```python
strategy = RetryStrategy(
    max_retries=3,
    delay=1.0,
    exponential_backoff=True,
    max_delay=10.0
)
```

## Context Management

Chains maintain execution context through the `ChainContext`:

```python
context = ChainContext(
    metadata=ChainMetadata(
        description="My chain",
        tags={"tag1", "tag2"}
    ),
    state=ChainState(
        variables={"key": "value"}
    )
)
```

## Advanced Usage

### 1. Custom Node Types
```python
class MyCustomNode(ChainNode):
    def __init__(self, custom_config: Dict[str, Any]):
        super().__init__(
            node_type=NodeType.AGENT,
            step=ChainStep(...)
        )
        self.custom_config = custom_config
```

### 2. Parallel Execution
```python
# Nodes at the same level execute concurrently
chain.add_node(parallel_node1)
chain.add_node(parallel_node2)
chain.add_edge(ChainEdge(source_id=start.id, target_id=parallel_node1.id))
chain.add_edge(ChainEdge(source_id=start.id, target_id=parallel_node2.id))
```

### 3. Conditional Execution
```python
chain.add_edge(ChainEdge(
    source_id=check_node.id,
    target_id=success_node.id,
    condition="result.status == 'success'"
))
```

## Best Practices

1. **Error Handling**
   - Always provide retry strategies for critical nodes
   - Use validation nodes for structured outputs
   - Implement custom error handlers for specific failure modes

2. **Context Management**
   - Preserve context during retries
   - Use input/output transforms for data manipulation
   - Maintain conversation history in context

3. **Performance**
   - Group independent operations for parallel execution
   - Use appropriate retry delays
   - Consider token limits in prompts

4. **Validation**
   - Define clear schemas for structured outputs
   - Use appropriate response formats
   - Handle partial validation success when appropriate

## Examples

### Multi-Stage Processing Chain
```python
async def create_processing_chain(agent_pool: AgentPool):
    chain = ChainGraph(agent_pool=agent_pool)

    # Extract information
    extract_node = ChainNode(
        step=ChainStep(prompt=extraction_prompt),
        node_type=NodeType.AGENT
    )

    # Validate extraction
    validate_node = ValidationNode(
        response_format=ResponseFormat(
            format=ResponseFormatType.JSON,
            schema=extraction_schema
        )
    )

    # Process results
    process_node = ChainNode(
        step=ChainStep(
            prompt=processing_prompt,
            input_transform=lambda ctx, data: {
                "extracted": data["validation_result"]
            }
        )
    )

    # Add nodes and edges
    chain.add_node(extract_node)
    chain.add_node(validate_node)
    chain.add_node(process_node)

    chain.add_edge(ChainEdge(
        source_id=extract_node.id,
        target_id=validate_node.id
    ))
    chain.add_edge(ChainEdge(
        source_id=validate_node.id,
        target_id=process_node.id
    ))

    return chain
```

### Conversation Chain with History
```python
async def create_conversation_chain(agent_pool: AgentPool):
    chain = ChainGraph(agent_pool=agent_pool)

    # Create conversation node
    convo_node = ChainNode(
        step=ChainStep(
            prompt=conversation_prompt,
            input_transform=lambda ctx, data: {
                "input": data["user_input"],
                "history": ctx.variables.get("history", [])
            }
        )
    )

    # Add history tracking
    def update_history(response):
        history = chain.context.variables.get("history", [])
        history.append({
            "role": "assistant",
            "content": response.content
        })
        chain.context.variables["history"] = history
        return response

    convo_node.step.output_transform = update_history

    chain.add_node(convo_node)
    return chain
```

## Contributing

When adding new features to the chains module:

1. Add appropriate tests
2. Update type hints
3. Document new components
4. Add examples to this README
5. Consider backward compatibility

## See Also

- [Prompts Documentation](../prompts/README.md)
- [Agents Documentation](../agents/README.md)
- [Core Documentation](../core/README.md)
