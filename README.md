# LLMaestro

A system for programmatically interacting with LLMs, with support for conversations, chains, and task orchestration.

It's like LiteLLM and LangChain, but worse!

## Model Status

### Anthropic Models
![claude-3-5-sonnet-latest](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/claude-3-5-sonnet-latest.json) - Balanced Claude 3 model offering strong performance at a lower cost than Opus
![claude-3-haiku-latest](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/claude-3-haiku-latest.json) - Fastest and most cost-effective Claude 3 model, optimized for simpler tasks
![claude-3-opus-latest](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/claude-3-opus-latest.json) - Most capable Claude 3 model, best for complex tasks requiring deep analysis, coding, and reasoning

### OpenAI Models
![gpt-3.5-turbo](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/gpt-3.5-turbo.json) - Fast and cost-effective GPT-3.5 model for most tasks
![gpt-4-turbo-preview](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/gpt-4-turbo-preview.json) - Most capable and up-to-date GPT-4 model with 128k context

### Google Models
![gemini-1.5-flash](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/gemini-1.5-flash.json)
![gemini-pro](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/gemini-pro.json)

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
