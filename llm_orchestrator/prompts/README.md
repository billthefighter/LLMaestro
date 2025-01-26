# LLM Prompts

This directory contains the prompt templates used by the LLM Orchestrator. Each prompt is defined in YAML format with specific metadata and formatting requirements.

## Directory Structure

```
prompts/
├── README.md
└── tasks/           # Task-specific prompts
    ├── pdf_analysis.yaml
    ├── code_refactor.yaml
    └── lint_fix.yaml
```

## Prompt Format

Each prompt file must be in YAML format with the following structure:

```yaml
name: "unique_prompt_name"
version: "1.0.0"
description: "Brief description of what this prompt does"
author: "Author Name"
git_metadata:
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
- `version`: Semantic version of the prompt
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

## Git Integration

The prompt templates are automatically versioned using Git hooks. The `git_metadata` field is managed by pre-commit hooks that:
1. Track when a template is first created
2. Update the last modification information when changes are made
3. Record the commit hash and author for both creation and modifications

To set up the Git hooks:

```bash
# Install the pre-commit hooks
poetry run pre-commit install
```

## Best Practices

1. **Versioning**: Use semantic versioning (MAJOR.MINOR.PATCH) for prompts
2. **Documentation**: Include clear descriptions and examples
3. **Schema**: Define clear response schemas for structured outputs
4. **Testing**: Include example inputs and expected outputs
5. **Git History**: Use meaningful commit messages when updating prompts 