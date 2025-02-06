# Changelist Manager

An automated tool for maintaining changelists and ensuring documentation stays up-to-date with code changes.

## Features

- Automatically generates changelists from git diffs
- Maintains a `changelist.md` file with chronological entries
- Identifies README files that may need updates
- Suggests specific documentation updates
- Runs automatically as a pre-push git hook

## Installation

1. The application is included in the LLMaestro package
2. Ensure the pre-commit hook is installed:
   ```bash
   pre-commit install --hook-type pre-push
   ```

## Usage

The changelist manager runs automatically on git push, but you can also run it manually:

```bash
python scripts/update_changelist.py
```

### Changelist Format

The `changelist.md` file follows this format:

```markdown
# Changelist

## 2024-02-05 12:34:56

Summary of changes...

Files changed:
- file1.py
- file2.py

---
```

### Configuration

The application uses the standard LLMaestro configuration system. Relevant settings:

```yaml
agents:
  default_agent_type: "changelist"  # Agent type to use for processing
  max_agents: 1                     # Number of concurrent agents
```

## Prompt Templates

The application uses two main prompt templates:

1. `generate_changelist.yaml`: Analyzes git diffs and generates summaries
2. `validate_readmes.yaml`: Validates if README files need updates

## Development

### Adding New Features

1. Modify `app.py` to add new functionality
2. Update prompt templates in `prompts/` directory
3. Add tests in `tests/applications/changelistmanager/`

### Testing

Run the test suite:

```bash
pytest tests/applications/changelistmanager/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

Same as the parent LLMaestro project.
