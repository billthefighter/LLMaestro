# LLMaestro

A system for orchestrating large-scale LLM tasks that exceed token limits through task decomposition and parallel processing. Also includes a collection of applications that demonstrate common use cases and best practices.

## Model Status

### Anthropic Models
![claude-3-5-sonnet-latest](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lucaswhipple/llm_orchestrator/main/docs/badges/claude-3-5-sonnet-latest.json)

### OpenAI Models
![gpt-4-turbo-preview](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lucaswhipple/llm_orchestrator/main/docs/badges/gpt-4-turbo-preview.json)
![gpt-4](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lucaswhipple/llm_orchestrator/main/docs/badges/gpt-4.json)
![gpt-3.5-turbo](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lucaswhipple/llm_orchestrator/main/docs/badges/gpt-3.5-turbo.json)

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

## Applications

The framework includes several example applications demonstrating common use cases and best practices. See the [applications directory](src/applications/) for detailed documentation on creating and structuring applications.

| Application | Description | Key Features |
|------------|-------------|--------------|
| [PDFReader](src/applications/pdfreader/) | Extract structured data from PDFs using vision capabilities | - Multi-page processing<br>- Schema-based extraction<br>- Confidence scoring |
| [FunctionRunner](src/applications/funcrunner/) | Execute functions through natural language requests | - Natural language processing<br>- Type-safe execution<br>- Confidence scoring |

Each application follows a standardized structure and includes:
- Comprehensive documentation
- YAML-based prompt templates
- Pydantic data models
- Unit and integration tests

## Examples

Each example demonstrates how LLM Orchestrator handles tasks that would typically exceed token limits by:
- Automatically decomposing large tasks into manageable chunks
- Processing subtasks in parallel where possible
- Managing context and resources efficiently
- Aggregating results into a coherent output

## Visualization

The system provides interactive visualizations for different types of task decomposition patterns:

- [Sequential Chain](examples/visualizations/sequential_chain.html) - Simple linear task decomposition
- [Parallel Chain](examples/visualizations/parallel_chain.html) - Tasks processed concurrently
- [Chord Chain](examples/visualizations/chord_chain.html) - Complex dependencies between tasks
- [Group Chain](examples/visualizations/group_chain.html) - Tasks organized in logical groups
- [Map Chain](examples/visualizations/map_chain.html) - Map-reduce style processing
- [Reminder Chain](examples/visualizations/reminder_chain.html) - Tasks with temporal dependencies
- [Recursive Chain](examples/visualizations/recursive_chain.html) - Self-referential task decomposition
- [Nested Chain](examples/visualizations/nested_chain.html) - Hierarchical task organization

Each visualization demonstrates how different types of tasks are broken down and processed by the orchestrator. Click the links above to view the interactive visualizations.

Note: To view the interactive visualizations locally, clone the repository and open the HTML files in your browser.

## Configuration

The project requires configuration for LLM API access. You can set this up in two ways:

### 1. Using a Configuration File

1. Copy the example configuration:
```bash
cp example_config.yaml config.yaml
```

2. Edit `config.yaml` and replace the placeholder API key with your actual key:
```yaml
llm:
  provider: anthropic
  model: claude-3-sonnet-20240229
  api_key: your-api-key-here
```

### 2. Using Environment Variables

Alternatively, you can set the following environment variables:

```bash
export ANTHROPIC_API_KEY=your-api-key-here
export ANTHROPIC_MODEL=claude-3-sonnet-20240229  # optional, defaults to sonnet
```

## Security Notes

- Never commit API keys to version control
- The `config.yaml` file is already added to `.gitignore`
- Use environment variables in production environments
- Rotate API keys if they are ever exposed

## Getting Started

1. Install dependencies:
```bash
poetry install
```

2. Set up configuration using one of the methods above

3. Try the visualization demo:
```bash
python examples/live_visualization_demo.py
```
