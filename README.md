# LLM Orchestrator

A system for orchestrating large-scale LLM tasks that exceed token limits through task decomposition and parallel processing.

## Architecture

The system is built around three main concepts:

1. **Task Manager**: Handles task decomposition, scheduling, and result aggregation
2. **Agent Pool**: Manages a pool of LLM agents that can be assigned subtasks
3. **Storage Manager**: Handles efficient storage and retrieval of intermediate results

### Core Components

- `TaskManager`: Decomposes large tasks into smaller, manageable chunks
- `Agent`: Represents an LLM instance that can process subtasks
- `StorageManager`: Manages intermediate results using disk-based storage
- `Validator`: Ensures data consistency using Pydantic models

### Key Features

- Task decomposition strategies for different types of tasks (PDF analysis, code refactoring, etc.)
- Efficient resource management for parallel processing
- Model-agnostic design supporting different LLM providers
- Disk-based storage for handling large datasets
- Strong type validation using Pydantic

## Setup

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest
```

## Usage

```python
from llm_orchestrator import TaskManager, Agent

# Initialize the orchestrator
task_manager = TaskManager()

# Create a task
task = task_manager.create_task(
    task_type="pdf_analysis",
    input_data="path/to/pdfs",
    config={"batch_size": 100}
)

# Execute the task
results = task_manager.execute(task)
``` 