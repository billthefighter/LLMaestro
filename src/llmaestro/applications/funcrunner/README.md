# FunctionRunner

A flexible system for executing functions through natural language requests using LLMs.

## Features

- Natural language to function call translation
- Type-safe function registration and execution
- Confidence scoring for function selection
- Automatic parameter type inference
- Built-in error handling and validation
- Support for async operations

## Usage

1. Create a runner instance:
```python
from src.applications.funcrunner import FunctionRunner

runner = FunctionRunner(
    api_key="your-api-key",  # Optional, defaults to config.yaml
    model_name="claude-3-sonnet-20240229"  # Optional
)
```

2. Register functions:
```python
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle"""
    return length * width

runner.register_function(
    calculate_area,
    "Calculate the area of a rectangle given length and width"
)
```

3. Process natural language requests:
```python
result = await runner.process_llm_request(
    "Calculate the area of a rectangle that is 5 meters by 3 meters"
)

print(f"Result: {result.result}")
if result.error:
    print(f"Error: {result.error}")
```

## Configuration

Configure FunctionRunner using any of these methods:

1. Using config.yaml:
```yaml
llm:
  provider: anthropic
  model: claude-3-sonnet-20240229
  api_key: your-api-key
  max_tokens: 1024
  temperature: 0.7
```

2. Using environment variables:
```bash
export ANTHROPIC_API_KEY=your-api-key
```

3. Direct initialization:
```python
runner = FunctionRunner(
    api_key="your-api-key",
    config_path="path/to/config.yaml",
    model_name="claude-3-sonnet-20240229"
)
```

## Function Registration

Functions can be registered with type hints and descriptions:

```python
from typing import List, Optional

def send_email(
    to: str,
    subject: str,
    body: str,
    cc: Optional[List[str]] = None
) -> bool:
    """Send an email to specified recipients"""
    # Implementation
    return True

runner.register_function(
    send_email,
    "Send an email to specified recipients with optional CC"
)
```

## Error Handling

The system handles various error cases:

1. Function not found:
```python
try:
    result = await runner.process_llm_request("Call nonexistent_function")
except ValueError as e:
    print(f"Function not found: {e}")
```

2. Invalid arguments:
```python
try:
    result = await runner.process_llm_request("Calculate area with invalid input")
    if result.error:
        print(f"Execution error: {result.error}")
except Exception as e:
    print(f"Processing error: {e}")
```

3. Low confidence handling:
```python
result = await runner.process_llm_request("Ambiguous request")
if result.error and "Low confidence" in result.error:
    print("Request was too ambiguous")
```

## Testing

Run the tests with:
```bash
# All tests
pytest tests/test_integration/test_funcrunner.py

# Only integration tests
pytest tests/test_integration/test_funcrunner.py -m integration

# With real LLM
pytest tests/test_integration/test_funcrunner.py --use-llm-tokens
```
