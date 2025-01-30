# LLM Prompts

This directory contains the prompt templates used by the LLM Orchestrator. Each prompt is defined in YAML format with specific metadata and formatting requirements.

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

## Schema Validation

The `schema.json` file provides a JSON Schema for validating prompt templates. To use it:

1. **VS Code**: Add this to your settings.json:
```json
{
  "yaml.schemas": {
    "./llm_orchestrator/prompts/schema.json": "llm_orchestrator/prompts/tasks/*.yaml"
  }
}
```

2. **Command Line**: Use a tool like `yamllint` or `jsonschema`:
```bash
poetry run jsonschema -i prompts/tasks/pdf_analysis.yaml prompts/schema.json
```

The schema enforces:
- Required fields and their types
- Valid task types and response formats
- Proper version number format
- Git metadata structure
- Example format requirements

## Prompt Format

Each prompt file must be in YAML format with the following structure:

```yaml
name: "unique_prompt_name"
version: "1.0.0"  # Automatically updated by pre-commit hook
description: "Brief description of what this prompt does"
author: "Author Name"
git_metadata:      # Automatically managed by pre-commit hook
  created:
    commit: "commit_hash_when_file_was_created"
    author: "commit_author"
  last_modified:
    commit: "commit_hash_of_last_modification"
    author: "commit_author"

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

## Required Fields

- `name`: Unique identifier for the prompt
- `version`: Semantic version of the prompt (automatically managed)
- `description`: Brief explanation of the prompt's purpose
- `metadata.type`: Type of task this prompt is designed for
- `metadata.expected_response.format`: Expected format of the LLM's response
- `system_prompt`: The system-level instructions for the LLM
- `user_prompt`: Template for formatting user input

## Optional Fields

- `author`: Creator of the prompt
- `git_metadata`: Automatically managed Git commit information
- `metadata.model_requirements`: Specific model requirements
- `examples`: Sample inputs and outputs for testing

## Automatic Versioning

The version number is automatically managed by the pre-commit hook following semantic versioning rules:

1. **MAJOR** (x.0.0): Manual update only
   - Requires explicitly changing the version number
   - Use for breaking changes in prompt interface

2. **MINOR** (0.x.0): Automatic update for prompt changes
   - Incremented when `system_prompt` or `user_prompt` changes
   - Resets patch version to 0
   - Use for backward-compatible prompt improvements

3. **PATCH** (0.0.x): Automatic update for other changes
   - Incremented for metadata updates, example additions, etc.
   - Use for documentation and non-functional changes

To manually update the MAJOR version (for breaking changes):
1. Change the version number in the YAML file
2. Commit the change
3. The pre-commit hook will preserve your manual version change

## Git Integration

The prompt templates are automatically versioned using Git hooks. The `git_metadata` field is managed by pre-commit hooks that:
1. Track when a template is first created
2. Update the last modification information when changes are made
3. Record the commit hash and author for both creation and modifications
4. Automatically update the version number based on changes

To set up the Git hooks:

```bash
# Install the pre-commit hooks
poetry run pre-commit install
```

## Best Practices

1. **Versioning**:
   - Let the hook manage MINOR and PATCH versions
   - Only manually update MAJOR version for breaking changes
   - Document significant changes in commit messages
2. **Documentation**: Include clear descriptions and examples
3. **Schema**: Define clear response schemas for structured outputs
4. **Testing**: Include example inputs and expected outputs
5. **Git History**: Use meaningful commit messages when updating prompts
