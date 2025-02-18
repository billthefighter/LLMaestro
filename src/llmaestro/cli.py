import asyncio
import functools
import json
from pathlib import Path
from typing import Optional

import click

from llmaestro.core.config import get_config
from llmaestro.core.models import AgentConfig
from llmaestro.llm.chains import ChainStep, SequentialChain
from llmaestro.llm.interfaces import create_llm_interface
from llmaestro.prompts.loader import PromptLoader
from llmaestro.utils.storage import StorageManager


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
    config_manager = get_config()
    user_config = config_manager.user_config

    # Update API keys
    if openai_key is not None:
        user_config.api_keys["openai"] = openai_key
    if anthropic_key is not None:
        user_config.api_keys["anthropic"] = anthropic_key
    if azure_key is not None:
        user_config.api_keys["azure"] = azure_key
    if azure_endpoint is not None:
        user_config.api_keys["azure_endpoint"] = azure_endpoint

    # Update default model if specified
    if default_model is not None:
        provider, model = default_model.split("/", 1) if "/" in default_model else ("anthropic", default_model)
        user_config.default_model = {
            "provider": provider,
            "name": model,
            "settings": user_config.default_model.get("settings", {}),
        }

    # Update storage path if specified
    if storage_path is not None:
        user_config.storage["path"] = storage_path

    # Save updated configuration
    config_path = Path.home() / ".llmaestro" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(user_config.model_dump(), f, indent=2)

    click.echo("Configuration updated successfully")


@config.command()
def show():
    """Show current configuration"""
    config_manager = get_config()
    user_config = config_manager.user_config

    # Create a copy of the config for display
    config_dict = user_config.model_dump()

    # Mask API keys
    for provider, key in config_dict["api_keys"].items():
        if isinstance(key, str) and "key" in provider.lower():
            config_dict["api_keys"][provider] = "*" * 8 + key[-4:] if len(key) > 4 else "*" * len(key)

    click.echo(json.dumps(config_dict, indent=2))


@cli.command()
@click.argument("chain_name")
@click.option("--input", "-i", multiple=True, help="Input in the format key=value")
@click.option("--model", help="Override default model (format: provider/model)")
@click.option("--agent-type", help="Agent type to use (e.g., general, fast, specialist)")
@coro
async def run(chain_name: str, input: tuple[str, ...], model: Optional[str], agent_type: Optional[str]):
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

    config_manager = get_config()
    storage = StorageManager(config_manager.user_config.storage["path"])
    prompt_loader = PromptLoader()

    try:
        # Get agent configuration
        agent_config_dict = config_manager.get_agent_config(agent_type)

        # Override model if specified
        if model:
            provider, model_name = model.split("/", 1) if "/" in model else ("anthropic", model)
            # model_config = config_manager.get_model_config(provider, model_name)
            agent_config_dict.update(
                {
                    "provider": provider,
                    "model_name": model_name,
                }
            )

        # Get API key for the provider
        provider = agent_config_dict["provider"]
        api_key = config_manager.user_config.api_keys.get(provider)
        if not api_key:
            raise ValueError(f"No API key found for provider: {provider}")

        # Convert dictionary to AgentConfig
        agent_config = AgentConfig(
            provider=provider,
            model_name=agent_config_dict["model"],
            api_key=api_key,
            max_tokens=agent_config_dict["max_tokens"],
            temperature=agent_config_dict["temperature"],
        )

        # Create LLM interface
        llm = create_llm_interface(agent_config)

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
