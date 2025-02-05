# Applications

This directory contains example applications built using the LLM Orchestrator framework. Each application demonstrates best practices for structuring and implementing LLM-powered tools.

## Application Structure

A typical application should follow this structure:

```
applications/
└── your_app/
    ├── __init__.py
    ├── app.py          # Main application code
    ├── prompt.yaml     # Prompt template
    ├── models.py       # Pydantic models (optional)
    ├── utils.py        # Helper functions (optional)
    └── README.md       # Application-specific documentation
```

## Core Components

Each application should include:

1. **Main Application Class**
   - Inherits or uses framework components
   - Handles initialization and configuration
   - Provides high-level interface for users

2. **Prompt Template (YAML)**
   - Defines system and user prompts
   - Specifies expected response format
   - Includes metadata and versioning
   - Uses schema validation

3. **Data Models**
   - Uses Pydantic for validation
   - Defines input/output schemas
   - Handles type conversion
   - Documents data structures

4. **Configuration**
   - Uses framework's config system
   - Supports environment variables
   - Allows custom settings

## Best Practices

1. **Initialization**
   ```python
   class YourApp:
       def __init__(
           self,
           output_model: Type[BaseModel],
           api_key: Optional[str] = None,
           config_path: Optional[Path] = None
       ):
           # Initialize components
           self._init_config()
           self._init_llm()
           self._init_prompts()
   ```

2. **Prompt Management**
   ```python
   class YourAppPrompt(BasePrompt):
       def __init__(self, output_model: Type[BaseModel], **data):
           # Load and customize prompt template
           prompt_path = Path(__file__).parent / "prompt.yaml"
           prompt_data = yaml.safe_load(prompt_path.read_text())
           data.update(prompt_data)
           super().__init__(**data)
   ```

3. **Response Handling**
   ```python
   class YourAppResponse(BaseModel):
       result: Dict[str, Any]
       metadata: Dict[str, Any]
       warnings: Optional[List[str]] = None
   ```

4. **Error Handling**
   ```python
   try:
       result = await self.process(input_data)
   except Exception as e:
       logger.error(f"Processing failed: {e}")
       raise YourAppError(f"Failed to process: {str(e)}")
   ```

## Current Applications

1. [PDFReader](pdfreader/): Extract structured data from PDF documents using vision capabilities
   - Demonstrates multi-page processing
   - Shows schema-based extraction
   - Includes confidence scoring

## Creating a New Application

1. Create a new directory:
   ```bash
   mkdir src/applications/your_app
   ```

2. Copy the template structure:
   ```bash
   cp -r src/applications/pdfreader/* src/applications/your_app/
   ```

3. Customize the components:
   - Modify prompt.yaml for your use case
   - Update the main class for your needs
   - Define your data models
   - Add application-specific features

4. Add documentation:
   - Create a README.md
   - Document usage examples
   - Explain configuration options

## Testing

Each application should include:

1. Unit tests for core functionality
2. Integration tests with the LLM
3. Example usage in documentation
4. Test fixtures and mocks

Example:
```python
@pytest.mark.asyncio
async def test_your_app():
    app = YourApp(output_model=YourModel)
    result = await app.process(test_input)
    assert isinstance(result, YourAppResponse)
    assert result.warnings is None
```
