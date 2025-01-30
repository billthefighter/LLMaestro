# Configuration

This directory contains configuration files and schemas for the LLM Orchestrator.

## Files

- `example_config.yaml`: Template configuration file with documentation
- `schema.json`: JSON Schema defining the configuration format
- `config.yaml`: Your local configuration file (not tracked in git)

## Usage

1. Copy the example configuration:
```bash
cp config/example_config.yaml config/config.yaml
```

2. Edit `config.yaml` with your settings:
```yaml
llm:
  provider: anthropic
  model: claude-3-sonnet-20240229
  api_key: your-api-key-here
```

## Environment Variables

You can also use environment variables instead of a config file:

```bash
export ANTHROPIC_API_KEY=your-api-key-here
export ANTHROPIC_MODEL=claude-3-sonnet-20240229
export ANTHROPIC_MAX_TOKENS=1024
export ANTHROPIC_TEMPERATURE=0.7
```

## Schema

The configuration schema (`schema.json`) defines:

- **LLM Settings**
  - Provider selection
  - Model selection
  - API key validation
  - Generation parameters

- **Storage Settings**
  - Storage path
  - Data format (json/pickle/yaml)

- **Visualization Settings**
  - Server host/port
  - Display options

- **Logging Settings**
  - Log level
  - Log file path

## Validation

The configuration is automatically validated against the schema when loaded. This ensures:

- Required fields are present
- Values are within allowed ranges
- Enums match allowed values
- Types are correct

## Security

- Never commit `config.yaml` to version control
- Use environment variables in production
- Rotate API keys regularly
- Keep your API keys secure
