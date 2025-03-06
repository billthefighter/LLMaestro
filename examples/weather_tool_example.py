#!/usr/bin/env python3
"""
Weather Tool Example

This example demonstrates:
1. Using the default LLM factory to initialize LLMs
2. Creating a prompt with a tool
3. Executing the prompt and handling the response

Usage:
    python weather_tool_example.py

Environment variables:
    OPENAI_API_KEY: Your OpenAI API key
    ANTHROPIC_API_KEY: Your Anthropic API key (optional)
"""

import asyncio
import os
from typing import Dict, Any, Optional

from llmaestro.default_library.default_llm_factory import LLMDefaultFactory
from llmaestro.agents.agent_pool import AgentPool
from llmaestro.prompts.base import BasePrompt, PromptVariable, SerializableType
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.prompts.tools import ToolParams, BasicFunctionGuard
from llmaestro.llm.credentials import APIKey
from llmaestro.llm.llm_registry import LLMRegistry


def get_weather(location: str) -> str:
    """Get current temperature for a given location."""
    # In a real application, this would call a weather API
    weather_data = {
        "San Francisco": "sunny with a temperature of 72°F",
        "New York": "partly cloudy with a temperature of 65°F",
        "London": "rainy with a temperature of 55°F",
        "Tokyo": "clear skies with a temperature of 78°F",
        "Sydney": "warm with a temperature of 80°F",
    }

    # Default response for unknown locations
    return f"The weather in {location} is {weather_data.get(location, 'currently unavailable')}."


async def setup_llm_registry() -> LLMRegistry:
    """Set up the LLM registry with API keys from environment variables."""
    # Get API keys from environment
    credentials = {}

    # OpenAI (required for this example)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    credentials["openai"] = APIKey(key=openai_api_key)

    # Anthropic (optional)
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        credentials["anthropic"] = APIKey(key=anthropic_api_key)

    # Initialize the factory with credentials
    factory = LLMDefaultFactory(credentials=credentials)
    return await factory.DefaultLLMRegistryFactory()


async def main():
    # Set up the LLM registry
    try:
        registry = await setup_llm_registry()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    # Print available models
    print("Available models:")
    # Get the list of registered models instead of accessing providers directly
    for model_name in registry.get_registered_models():
        print(f"  Model: {model_name}")

    # Create an agent pool with the LLM registry
    agent_pool = AgentPool(llm_registry=registry)

    # Select a model for this example
    model_name = "gpt-3.5-turbo"  # A good default that's cost-effective
    provider_name = "openai"
    print(f"\nUsing model: {provider_name}/{model_name}")

    # Create a prompt with a tool and define the 'location' variable
    weather_prompt = MemoryPrompt(
        name="weather_query",
        description="Query weather information using tools",
        system_prompt="You are a helpful weather assistant. Use the provided tools to get weather information.",
        user_prompt="What is the weather like in {location} today?",
        tools=[
            ToolParams.from_function(get_weather)
        ],
        variables=[
            PromptVariable(
                name="location",
                description="The location to get weather for",
                expected_input_type=SerializableType.STRING
            )
        ]
    )

    # Get user input for location
    location = input("\nEnter a location (e.g., San Francisco, New York, London): ")
    if not location:
        location = "San Francisco"  # Default location

    print(f"\nQuerying weather for: {location}")

    try:
        # Render the prompt with the location variable
        system_prompt, user_prompt, _, tools = weather_prompt.render(location=location)

        # Create a new prompt with the rendered content
        formatted_prompt = MemoryPrompt(
            name=weather_prompt.name,
            description=weather_prompt.description,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools
        )

        # Execute the prompt with the agent pool
        response = await agent_pool.execute_prompt(
            prompt=formatted_prompt
        )

        # Handle the response
        print("\nResponse from LLM:")
        print(f"{response.content}")

        print("\nToken Usage:")
        if response.token_usage:
            print(f"  Prompt tokens: {response.token_usage.prompt_tokens}")
            print(f"  Completion tokens: {response.token_usage.completion_tokens}")
            print(f"  Total tokens: {response.token_usage.total_tokens}")
        else:
            print("  Token usage information not available")

        # Check for tool usage in the metadata instead of accessing tool_results directly
        if response.metadata and "tool_results" in response.metadata:
            print("\nTool Results:")
            for tool_result in response.metadata["tool_results"]:
                print(f"  Tool: {tool_result.get('name', 'unknown')}")
                print(f"  Result: {tool_result.get('result', 'N/A')}")

    except Exception as e:
        print(f"\nError executing prompt: {e}")


if __name__ == "__main__":
    asyncio.run(main())
