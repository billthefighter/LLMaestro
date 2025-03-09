# LLMaestro Tools

This directory contains tools that can be used by LLMs to interact with various systems and services.

## Available Tools

### SQL Tools (`sql_tools.py`)

The SQL tools allow LLMs to execute SQL queries against databases with appropriate safety guards.

#### Features

- **Read-only SQL queries**: Allows only SELECT statements for data retrieval
- **Read-write SQL queries**: Allows both read and write operations with safety checks
- **SQL injection prevention**: Parameterized queries and input validation
- **Dangerous operation detection**: Blocks potentially harmful operations
- **SQLAlchemy integration**: Works with any database supported by SQLAlchemy

#### Components

- `SQLQueryParams`: Pydantic model for SQL query parameters
- `SQLReadOnlyGuard`: Function guard that only allows read-only SQL queries
- `SQLReadWriteGuard`: Function guard that allows both read and write SQL queries
- `create_sql_read_only_tool`: Factory function to create a read-only SQL query tool
- `create_sql_read_write_tool`: Factory function to create a read-write SQL query tool

#### Usage Example

```python
from sqlalchemy import create_engine
from llmaestro.tools.sql_tools import create_sql_read_only_tool, create_sql_read_write_tool

# Create a database engine
engine = create_engine("sqlite:///example.db")

# Create read-only and read-write SQL tools
read_only_tool = create_sql_read_only_tool(engine)
read_write_tool = create_sql_read_write_tool(engine)

# Execute a read-only query
async def example():
    # Execute a SELECT query with parameters
    results = await read_only_tool.execute(
        query="SELECT * FROM users WHERE age > :min_age",
        params={"min_age": 25}
    )
    
    # Execute a write query (only works with read_write_tool)
    rows_affected = await read_write_tool.execute(
        query="INSERT INTO users (name, email) VALUES (:name, :email)",
        params={"name": "Alice", "email": "alice@example.com"}
    )
```

#### Safety Considerations

The SQL tools implement several safety measures:

1. **Read-only guard**: 
   - Only allows SELECT statements
   - Blocks any data modification operations

2. **Read-write guard**:
   - Blocks dangerous operations like:
     - System table access
     - Database structure modifications
     - Command execution
     - Potential SQL injection patterns
     - File operations

3. **Query cleaning**:
   - Removes comments
   - Normalizes whitespace
   - Performs pattern matching for dangerous operations

4. **Connection management**:
   - Properly handles transactions (commit/rollback)
   - Closes connections when created from engines

#### Adding to an LLM Chain

To add SQL tools to an LLM chain:

```python
from llmaestro.llm import OpenAIChat
from llmaestro.chains import Chain
from llmaestro.tools.sql_tools import create_sql_read_only_tool

# Create a database engine
engine = create_engine("sqlite:///example.db")

# Create a read-only SQL tool
sql_tool = create_sql_read_only_tool(engine)

# Create an LLM with the SQL tool
llm = OpenAIChat(
    model="gpt-4",
    tools=[sql_tool]
)

# Create a chain with the LLM
chain = Chain(llm=llm)

# Run the chain
response = await chain.run("How many users are in the database?")
```

### Tool Registry (`registry.py`)

The Tool Registry provides a centralized registry for all tools in LLMaestro, making it easy to discover and use available tools.

#### Features

- **Tool Registration**: Register tools with the registry using decorators or direct registration
- **Tool Discovery**: Discover available tools at runtime
- **Tool Categorization**: Organize tools into categories for better organization
- **Self-Registration**: The registry itself is a tool that can be used to discover other tools

#### Components

- `ToolRegistry`: A singleton class that serves as a registry for tools
- `register_tool`: A method for registering tools with the registry
- `create_tool_discovery_tool`: Factory function to create a tool for discovering available tools
- `get_registry`: Convenience function to get the singleton instance of the registry

#### Usage Example

```python
from llmaestro.tools.registry import get_registry, ToolRegistry

# Get the registry instance
registry = get_registry()

# Register a tool using the decorator
@registry.register(category="file_system")
def read_file(path: str) -> str:
    """Read a file from the file system."""
    with open(path, "r") as f:
        return f.read()

# Register a tool directly
def write_file(path: str, content: str) -> None:
    """Write content to a file."""
    with open(path, "w") as f:
        f.write(content)

registry.register_tool("write_file", write_file, category="file_system")

# Get a tool by name
read_file_tool = registry.get_tool("read_file")

# Get all tools in a category
file_system_tools = registry.get_tools_by_category("file_system")

# Use the registry as a tool to discover available tools
available_tools = registry.list_available_tools()
available_file_tools = registry.list_available_tools(category="file_system")
```

### All Tools (`all_tools.py`)

The All Tools module provides a central registry of all tools available in LLMaestro, making it easy to access and use all tools from a single import.

#### Features

- **Centralized Access**: Access all tools from a single import
- **Automatic Registration**: All tools are automatically registered when the module is imported
- **Category Organization**: Tools are organized by category for easy discovery
- **Placeholder Tools**: Placeholder tools for modules that are not yet fully implemented
- **Default Configurations**: Sensible defaults for tools that require configuration

#### Components

- `register_all_tools`: Function to register all available tools with the registry
- `get_all_tools`: Function to get all registered tools
- `get_tools_by_category`: Function to get all tools in a category
- `get_tool`: Function to get a tool by name
- `get_categories`: Function to get all categories

#### Usage Example

```python
from llmaestro.tools.all_tools import get_all_tools, get_tools_by_category, get_tool

# Get all available tools
all_tools = get_all_tools()
print(f"Total tools available: {len(all_tools)}")

# Get tools by category
sql_tools = get_tools_by_category("database")
file_tools = get_tools_by_category("file_system")

# Get a specific tool
read_only_sql = get_tool("execute_read_only_sql")

# Use a tool
async def example():
    results = await read_only_sql.execute(
        query="SELECT * FROM users WHERE age > :min_age",
        params={"min_age": 25}
    )
```

## Planned Tools

The following tools are planned for future implementation:

### File System Tools (`file_system_tools.py`)

Tools for interacting with the file system safely.

- Read files with content validation and size limits
- Write files with permission checks
- List directory contents with filtering options
- Search for files by pattern or content
- File metadata retrieval (size, creation date, etc.)

### API Integration Tools (`api_integration_tools.py`)

Tools for interacting with external APIs.

- REST API client with request validation
- GraphQL query execution
- OAuth authentication handling
- Rate limiting and retry logic
- Response parsing and validation

### Vector Database Tools (`vector_database_tools.py`)

Tools for working with vector databases and embeddings.

- Store and retrieve vector embeddings
- Similarity search with configurable metrics
- Hybrid search (vector + metadata filtering)
- Batch operations for efficiency
- Index management

### Data Processing Tools (`data_processing_tools.py`)

Tools for processing and transforming structured data.

- CSV/JSON/XML parsing and generation
- Data filtering and transformation
- Statistical analysis
- Data validation against schemas
- Tabular data operations

### Web Scraping Tools (`web_scraping_tools.py`)

Tools for extracting information from websites safely.

- HTML parsing with content extraction
- Navigation through web pages
- Handling of pagination and infinite scroll
- Respect for robots.txt and rate limits
- Extraction of structured data (tables, lists, etc.)

### Natural Language Processing Tools (`nlp_tools.py`)

Tools for enhancing text processing capabilities.

- Text classification and entity extraction
- Sentiment analysis
- Language detection
- Text summarization
- Keyword extraction

### Image Processing Tools (`image_processing_tools.py`)

Tools for working with images.

- Image loading and basic transformations
- OCR (Optical Character Recognition)
- Image description and captioning
- Object detection
- Image metadata extraction

### Caching and State Management Tools (`caching_tools.py`)

Tools for managing state and improving performance.

- Key-value storage with TTL (Time To Live)
- Session state management
- Result caching with invalidation strategies
- Distributed locking for concurrent operations
- Progress tracking for long-running tasks

### Authentication and Authorization Tools (`auth_tools.py`)

Tools for managing security and access control.

- User authentication with various methods
- Role-based access control
- Token management
- Credential validation
- Audit logging

### Workflow Orchestration Tools (`workflow_tools.py`)

Tools for coordinating complex multi-step processes.

- Task scheduling and execution
- Conditional branching
- Error handling and retry logic
- Parallel execution
- Progress monitoring

## Core Tool Components (`core.py`)

The core tool components provide the foundation for all tools in LLMaestro:

- `FunctionGuard`: Abstract base class for guarding function execution with safety checks
- `BasicFunctionGuard`: Default implementation with basic safety checks
- `ToolParams`: Parameters for a tool/function that can be used by an LLM

These components ensure that tools are executed safely and can be properly integrated with LLMs. 