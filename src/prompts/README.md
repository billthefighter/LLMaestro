# LLM Prompts

This directory contains the prompt templates used by the LLM Orchestrator. Each prompt is defined in a structured format with metadata, versioning, and optional git tracking.

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

current_version:
  number: "1.0.0"
  timestamp: "2024-03-20T12:00:00Z"
  author: "Author Name"
  description: "Initial version"
  change_type: "major"
  git_commit: "abc123..."  # Optional

system_prompt: |
  Clear description of the assistant's role and task.
  Can be multiple lines.

user_prompt: |
  Template for the user's input with {variables}.
  Can include multiple lines and formatting.

examples:
  - input:
      variable1: "example value 1"
      variable2: "example value 2"
    expected_output: |
      {
        "field1": "example response",
        "field2": ["item1", "item2"]
      }
```

## Usage Examples

1. **Loading Prompts**:
   ```python
   loader = PromptLoader()

   # Load from file
   file_prompt = await loader.load_prompt("file", "prompts/my_prompt.yaml")

   # Load from git repo
   git_prompt = await loader.load_prompt("git", "repo:path/to/prompt.yaml@main")
   ```

2. **Using Prompts**:
   ```python
   # Render template
   system, user = prompt.render(variable1="value1", variable2="value2")

   # Validate response
   is_valid, error = prompt.validate_response(llm_response)

   # Estimate tokens
   system_tokens, user_tokens = prompt.estimate_tokens()
   ```

3. **Version Management**:
   ```python
   # Update version
   prompt.bump_version_with_git("minor", "Updated prompt structure")

   # Check version info
   print(f"Current version: {prompt.version}")
   print(f"Created: {prompt.created_at}")
   print(f"Last modified: {prompt.updated_at}")
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

1. **Storage Selection**:
   - Use `FilePrompt` for simple local storage
   - Use `GitRepoPrompt` for version-controlled prompts
   - Use `S3Prompt` for cloud-based deployments

2. **Version Control**:
   - Use `bump_version_with_git()` to maintain git metadata
   - Document significant changes in version descriptions
   - Use appropriate change types (major/minor/patch)

3. **Examples and Testing**:
   - Include diverse examples covering edge cases
   - Validate examples before saving
   - Use the token estimation for context management

4. **Git Integration**:
   - Use meaningful commit messages
   - Track breaking changes appropriately
   - Maintain clean git history

5. **Error Handling**:
   - Check for validation errors before using prompts
   - Handle storage backend failures gracefully
   - Validate templates before rendering
