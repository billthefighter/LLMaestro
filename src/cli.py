import asyncio
import functools
import json
from typing import Optional

import click

from src.core.config import get_config
from src.core.models import AgentConfig
from src.llm.chains import ChainStep, SequentialChain
from src.llm.interfaces import create_llm_interface
from src.prompts.loader import PromptLoader
from src.utils.storage import StorageManager


def coro(f):
    """Turn an async function into a regular function."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@click.group()
def cli():
    """LLMaestro CLI - Manage and run LLM chains"""
    pass


@cli.group()
def config():
    """Manage configuration settings"""
    pass


@config.command()
@click.option("--openai-key", help="OpenAI API key")
@click.option("--anthropic-key", help="Anthropic API key")
@click.option("--azure-key", help="Azure OpenAI API key")
@click.option("--azure-endpoint", help="Azure OpenAI endpoint")
@click.option("--default-model", help="Default LLM model to use")
@click.option("--storage-path", help="Path to store chain execution data")
def set(
    openai_key: Optional[str],
    anthropic_key: Optional[str],
    azure_key: Optional[str],
    azure_endpoint: Optional[str],
    default_model: Optional[str],
    storage_path: Optional[str],
):
    """Set configuration values"""
    config = get_config()

    if openai_key is not None:
        config.api_credentials.openai_api_key = openai_key
    if anthropic_key is not None:
        config.api_credentials.anthropic_api_key = anthropic_key
    if azure_key is not None:
        config.api_credentials.azure_api_key = azure_key
    if azure_endpoint is not None:
        config.api_credentials.azure_endpoint = azure_endpoint
    if default_model is not None:
        config.default_model = default_model
    if storage_path is not None:
        config.storage_path = storage_path

    config.save_to_file()
    click.echo("Configuration updated successfully")


@config.command()
def show():
    """Show current configuration"""
    config = get_config()
    # Mask API keys in output
    config_dict = config.model_dump()
    creds = config_dict["api_credentials"]
    for key in creds:
        if creds[key] and "key" in key:
            creds[key] = "*" * 8 + creds[key][-4:]
    click.echo(json.dumps(config_dict, indent=2))


@cli.command()
@click.argument("chain_name")
@click.option("--input", "-i", multiple=True, help="Input in the format key=value")
@click.option("--model", help="Override default model")
@coro
async def run(chain_name: str, input: tuple[str, ...], model: Optional[str]):
    """Run a chain with the given inputs"""
    # Parse input parameters
    inputs = {}
    for inp in input:
        try:
            key, value = inp.split("=", 1)
            inputs[key.strip()] = value.strip()
        except ValueError:
            click.echo(f"Error: Invalid input format '{inp}'. Use key=value format.")
            return

    config = get_config()
    storage = StorageManager(config.storage_path)
    prompt_loader = PromptLoader()

    try:
        # Create LLM interface
        llm_config = AgentConfig(
            model_name=model or config.default_model,
            max_tokens=4096,  # Default max tokens
            api_key=config.api_credentials.openai_api_key,
        )
        llm = create_llm_interface(llm_config)

        # Create and run the chain
        chain = SequentialChain(
            steps=[ChainStep(task_type=chain_name)],
            llm=llm,
            storage=storage,
            prompt_loader=prompt_loader,
        )
        result = await chain.execute(**inputs)
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error running chain: {str(e)}", err=True)


@cli.command()
def list_chains():
    """List available chains"""
    prompt_loader = PromptLoader()

    if not prompt_loader.prompts:
        click.echo("No prompt templates found")
        return

    click.echo("Available chains:")
    for prompt_type, prompt in prompt_loader.prompts.items():
        click.echo(f"  - {prompt_type} ({prompt.description})")


if __name__ == "__main__":
    cli()
