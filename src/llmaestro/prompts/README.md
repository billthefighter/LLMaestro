# LLM Prompts

This directory contains the prompt templates used by the LLMaestro. Each prompt is defined in a structured format with metadata, versioning, and optional git tracking.

## Directory Structure

```
prompts/
├── README.md
├── schema.json     # JSON Schema for prompt validation
└── tasks/          # Task-specific prompts
    ├── pdf_analysis.yaml
    ├── code_refactor.yaml
    └── lint_fix.yaml
```

## Storage Implementations

The system supports multiple storage backends through different implementations:

1. **FilePrompt**: Local file storage with YAML format
   ```python
   prompt = await loader.load_prompt("file", "prompts/tasks/my_prompt.yaml")
   ```

2. **GitRepoPrompt**: Full git repository integration (requires `gitpython`)
   ```python
   prompt = await loader.load_prompt("git", "repo_path:prompts/my_prompt.yaml@main")
   ```

3. **S3Prompt**: AWS S3 storage (requires `boto3`)
   ```python
   prompt = await loader.load_prompt("s3", "bucket-name/path/to/prompt.json")
   ```

Each implementation can be extended with git tracking using the `GitMixin`:

```python
class CustomPrompt(BasePrompt, GitMixin):
    async def save(self) -> bool:
        author, commit = self.get_git_info()
        if author and commit:
            self.update_git_metadata(author, commit)
        # ... custom save logic ...
```

## Prompt Variables

The system supports type-safe variable handling for prompt templates:

1. **Variable Definition**:
   ```yaml
   variables:
     - name: "user_name"
       description: "Name of the user to address"
       expected_input_type: "string"
     - name: "items"
       description: "List of items to process"
       expected_input_type: "list"
       string_conversion_template: "{value:,}"  # Join with commas
   ```

2. **Type-Safe Usage**:
   ```python
   # Get the strongly-typed model for variables
   VariablesModel = prompt.get_variables_model()

   # Create and validate variables
   vars = VariablesModel(
       user_name="Alice",
       items=["item1", "item2"]
   )

   # Render with validated variables
   system, user, _ = prompt.render(variables=vars)
   ```

3. **Supported Types**:
   - `string`: Text values
   - `integer`: Whole numbers
   - `float`: Decimal numbers
   - `boolean`: True/False values
   - `list`: Arrays/sequences
   - `dict`: Key-value mappings
   - `schema`: JSON schema (as string or dict)

4. **Custom Formatting**:
   ```python
   from typing import List

   def format_items(items: List[str]) -> str:
       return "• " + "\n• ".join(items)

   prompt = BasePrompt(
       variables=[
           PromptVariable(
               name="items",
               expected_input_type="list",
               string_conversion_template=format_items
           )
       ],
       user_prompt="Process these items:\n{items}"
   )
   ```

## Version Control

Prompts now support sophisticated version control with:

1. **Semantic Versioning**:
   ```python
   prompt.bump_version("minor", "Updated template structure")
   prompt.bump_version_with_git("major", "Breaking change in response format")
   ```

2. **Version History**:
   ```python
   for version in prompt.version_history:
       print(f"{version.number}: {version.description} by {version.author}")
   ```

3. **Git Integration**:
   - Automatic git metadata tracking
   - Commit hash association with versions
   - Author attribution

## Prompt Format

Each prompt includes the following structure:

```yaml
name: "unique_prompt_name"
description: "Brief description of what this prompt does"
variables:
  - name: "user_name"
    description: "Name of the user to address"
    expected_input_type: "string"
  - name: "items"
    description: "List of items to process"
    expected_input_type: "list"
metadata:
  type: "task_type"  # e.g., pdf_analysis, code_refactor, lint_fix
  model_requirements:
    min_tokens: 1000
    preferred_models: ["gpt-4", "claude-2"]
  expected_response:
    format: "json"  # or "text", "markdown", etc.
    schema: |
      {
        "field1": "type and description",
        "field2": ["array", "of", "items"]
      }
  tags: ["category1", "category2"]
  is_active: true

system_prompt: |
  You are helping {user_name} with their task.
  Follow these instructions carefully.

user_prompt: |
  Process these items for {user_name}:
  {items}

examples:
  - input:
      user_name: "Alice"
      items: ["item1", "item2"]
    expected_output: |
      {
        "processed": ["item1_result", "item2_result"]
      }
```

## Usage Examples

1. **Loading Prompts**:
   ```python
   loader = PromptLoader()
   prompt = await loader.load_prompt("file", "prompts/my_prompt.yaml")
   ```

2. **Using Variables**:
   ```python
   # Get type information
   var_types = prompt.get_variable_types()
   required_vars = prompt.get_required_variables()

   # Get the variables model
   VariablesModel = prompt.get_variables_model()

   # Create type-safe variables
   try:
       vars = VariablesModel(
           user_name="Alice",
           items=["item1", "item2"]
       )
   except ValidationError as e:
       print("Invalid variables:", e)

   # Render the prompt
   system, user, attachments = prompt.render(variables=vars)
   ```

3. **Version Management**:
   ```python
   prompt.bump_version_with_git("minor", "Updated variable types")
   ```

4. **Example Management**:
   ```python
   # Add new example
   prompt.add_example(
       input_vars={"name": "Alice"},
       expected_output='{"greeting": "Hello Alice!"}'
   )

   # Validate examples
   errors = prompt.validate_all_examples()
   ```

## Best Practices

1. **Variable Handling**:
   - Define variables explicitly with types and descriptions
   - Use the variables model for type safety
   - Provide custom formatting when needed
   - Document variable requirements in examples

2. **Storage Selection**:
   - Use `FilePrompt` for simple local storage
   - Use `GitRepoPrompt` for version-controlled prompts
   - Use `S3Prompt` for cloud-based deployments

3. **Version Control**:
   - Use `bump_version_with_git()` to maintain git metadata
   - Document significant changes in version descriptions
   - Use appropriate change types (major/minor/patch)

4. **Examples and Testing**:
   - Include diverse examples covering edge cases
   - Validate examples before saving
   - Use the token estimation for context management

5. **Error Handling**:
   - Validate variables before rendering
   - Handle type conversion errors gracefully
   - Check for missing required variables
   - Validate templates before rendering
