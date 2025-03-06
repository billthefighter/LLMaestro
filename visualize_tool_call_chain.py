#!/usr/bin/env python3
"""
Visualization script for the ToolCallChain example.

This script demonstrates how to use the LLMaestro visualization framework
to create a visual representation of the ToolCallChain.
"""

import asyncio
from typing import Optional

from llmaestro.agents.agent_pool import AgentPool
from llmaestro.chains.tool_call_chain_example import ToolCallChain
from llmaestro.core.orchestrator import Orchestrator
from llmaestro.llm.registry import LLMRegistry
from llmaestro.llm.responses import ResponseFormat, ResponseFormatType
from llmaestro.prompts.base import BasePrompt
from llmaestro.visualization.live_visualizer import LiveVisualizer


async def visualize_tool_call_chain_static(output_path: Optional[str] = None) -> str:
    """
    Generate a static HTML visualization of the ToolCallChain.

    Args:
        output_path: Optional path to save the visualization HTML file

    Returns:
        Path to the generated HTML file
    """
    from llmaestro.visualization.visualize import ChainVisualizationManager

    # Create the necessary components
    llm_registry = LLMRegistry()
    agent_pool = AgentPool(llm_registry=llm_registry)
    orchestrator = Orchestrator(agent_pool=agent_pool)
    chain = ToolCallChain(orchestrator)
    chain.agent_pool = agent_pool

    # Create initial prompt
    initial_prompt = BasePrompt(
        name="weather_query",
        description="Query about weather with potential tool calls",
        system_prompt="You are a helpful assistant that can use tools to answer questions.",
        user_prompt="What's the weather like in New York today?",
    )

    # Define expected response format
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["answer", "sources"],
    }
    expected_format = ResponseFormat.from_json_schema(schema=schema, format_type=ResponseFormatType.JSON_SCHEMA)

    # Configure the chain
    await chain.create_tool_call_chain(
        name="Tool Call Example",
        initial_prompt=initial_prompt,
        expected_response_format=expected_format,
    )

    # Create visualization manager
    viz_manager = ChainVisualizationManager()

    # Custom layout for better visualization
    layout = {"name": "dagre", "rankDir": "LR", "nodeSep": 80, "rankSep": 150, "padding": 50}

    # Generate visualization
    if not output_path:
        output_path = "tool_call_chain_visualization.html"

    html_path = viz_manager.visualize_chain(chain=chain, output_path=output_path, layout=layout, method="html")

    print(f"Visualization saved to: {html_path}")
    return html_path


async def visualize_tool_call_chain_live(port: int = 8765) -> None:
    """
    Start a live visualization server for the ToolCallChain.

    Args:
        port: Port number for the WebSocket server
    """
    # Create the necessary components
    llm_registry = LLMRegistry()
    agent_pool = AgentPool(llm_registry=llm_registry)
    orchestrator = Orchestrator(agent_pool=agent_pool)

    # Create initial prompt
    initial_prompt = BasePrompt(
        name="weather_query",
        description="Query about weather with potential tool calls",
        system_prompt="You are a helpful assistant that can use tools to answer questions.",
        user_prompt="What's the weather like in New York today?",
    )

    # Define expected response format
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["answer", "sources"],
    }
    expected_format = ResponseFormat.from_json_schema(schema=schema, format_type=ResponseFormatType.JSON_SCHEMA)

    # Create live visualizer
    live_viz = LiveVisualizer(port=port)

    # Start the WebSocket server
    await live_viz.start_server()
    print(f"Live visualization server started on ws://localhost:{port}")
    print("Open the HTML file in your browser to view the visualization")

    try:
        # Create the chain
        chain = ToolCallChain(orchestrator)
        chain.agent_pool = agent_pool

        # Configure the chain
        await chain.create_tool_call_chain(
            name="Tool Call Example",
            initial_prompt=initial_prompt,
            expected_response_format=expected_format,
        )

        # Notify visualization of chain start
        await live_viz.on_chain_start(chain)

        # Execute the chain
        print("Executing chain...")
        results = await chain.execute_with_tool_handling()

        print("Chain execution completed")
        print(f"Conversation ID: {results['conversation_id']}")

        # Keep the server running for viewing
        print("Press Ctrl+C to stop the server")
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("Stopping server...")
    finally:
        await live_viz.stop_server()
        print("Server stopped")


async def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize the ToolCallChain example")
    parser.add_argument(
        "--mode",
        choices=["static", "live"],
        default="static",
        help="Visualization mode: static HTML or live WebSocket server",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tool_call_chain_visualization.html",
        help="Output path for static HTML visualization",
    )
    parser.add_argument("--port", type=int, default=8765, help="Port number for live visualization server")

    args = parser.parse_args()

    if args.mode == "static":
        await visualize_tool_call_chain_static(args.output)
    else:
        await visualize_tool_call_chain_live(args.port)


if __name__ == "__main__":
    asyncio.run(main())
