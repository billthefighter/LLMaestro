# LLMaestro

A system for orchestrating large-scale LLM tasks that exceed token limits through task decomposition and parallel processing. Also includes a collection of applications that demonstrate common use cases and best practices.

## Model Status

### Anthropic Models
![claude-2.1](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/claude-2.1.json)
![claude-3-5-sonnet-latest](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/claude-3-5-sonnet-latest.json)
![claude-3-haiku-20240229](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/claude-3-haiku-20240229.json)
![claude-3-opus-20240229](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/claude-3-opus-20240229.json)
![claude-3-sonnet-20240229](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/claude-3-sonnet-20240229.json)
![claude-instant-1.2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/claude-instant-1.2.json)

## Getting Started

## Documentation Index

The following README files provide detailed documentation for different components of the system:

### Core Components
- [Core System](src/core/README.md) - Core system components including configuration management
- [Agent System](src/agents/README.md) - Documentation for the agent system and agent pool
- [LLM Core](src/llm/README.md) - Core LLM functionality and chain implementations
- [LLM Interfaces](src/llm/interfaces/README.md) - LLM provider interface implementations
- [LLM Models](src/llm/models/README.md) - Model configurations and capabilities
- [Prompts](src/prompts/README.md) - Prompt template system and management

### Applications
- [Applications Overview](src/applications/README.md) - Guide to creating and structuring applications
- [PDFReader](src/applications/pdfreader/README.md) - PDF data extraction application
- [FunctionRunner](src/applications/funcrunner/README.md) - Natural language function execution

### Visualization and Configuration
- [Visualization](src/visualization/README.md) - Chain visualization tools and patterns
- [Configuration](config/README.md) - User configuration guide and examples

## Architecture

The system is built around three main concepts:

1. **Task Manager**: Handles task decomposition, scheduling, and result aggregation
2. **Agent Pool**: Manages a pool of LLM agents that can be assigned subtasks
3. **Storage Manager**: Handles efficient storage and retrieval of intermediate results

### Core Components

- `ConfigurationManager`: Manages system and user configuration with type safety
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

Each example demonstrates how LLMaestro handles tasks that would typically exceed token limits by:
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

The project uses a flexible configuration system that supports multiple initialization methods:

### 1. Using Configuration Files

1. Copy the example user configuration:
```bash
cp config/user_config.yml.example config/user_config.yml
```

2. Edit `user_config.yml` with your settings:
```yaml
api_keys:
  anthropic: your-api-key-here
default_model:
  provider: anthropic
  name: claude-3-sonnet-20240229
  settings:
    max_tokens: 1024
    temperature: 0.7
```

3. In your code:
```python
from core.config import get_config

config = get_config()
```

### 2. Using Environment Variables

Set environment variables for your providers:
```bash
# Provider API Keys
export ANTHROPIC_API_KEY=your-api-key-here
export OPENAI_API_KEY=your-openai-key  # optional

# Default Model Settings (optional)
export ANTHROPIC_MODEL=claude-3-sonnet-20240229
export ANTHROPIC_MAX_TOKENS=1024
export ANTHROPIC_TEMPERATURE=0.7

# Global Settings (optional)
export LLM_MAX_AGENTS=10
export LLM_STORAGE_PATH=chain_storage
export LLM_LOG_LEVEL=INFO
```

Then in your code:
```python
from core.config import get_config

config = get_config()
config.load_from_env()
```

### 3. Programmatic Configuration

You can also configure the system programmatically:

```python
from core.config import ConfigurationManager, UserConfig, SystemConfig

# Create configurations
user_config = UserConfig(
    api_keys={"anthropic": "your-api-key"},
    default_model={
        "provider": "anthropic",
        "name": "claude-3-sonnet-20240229",
        "settings": {"max_tokens": 1024}
    },
    # ... other settings ...
)

# Load system config from file
system_config = SystemConfig.from_yaml("config/system_config.yml")

# Initialize configuration
config = ConfigurationManager()
config.initialize(user_config, system_config)
```

For detailed configuration documentation, see:
- [Core Configuration](src/core/README.md#configuration-management)
- [Configuration Files](config/README.md)
- [LLM Models](src/llm/README.md)

### Security Notes

- Never commit API keys to version control
- The `user_config.yml` file is already added to `.gitignore`
- Use environment variables in production environments
- Keep API keys secure and rotate them if exposed

## Getting Started

1. Install dependencies:
```
