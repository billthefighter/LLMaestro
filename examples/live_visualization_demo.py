"""Demo of live visualization capabilities."""

import asyncio
import random
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from llmaestro.chains.chains import ChainNode, ChainStep, NodeType, ChainMetadata
from llmaestro.core.models import LLMResponse
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.types import PromptMetadata
from llmaestro.visualization.live_visualizer import LiveVisualizer


async def simulate_step_execution(step: ChainNode, visualizer: LiveVisualizer) -> bool:
    """Simulate execution of a chain step."""
    # Simulate start
    await visualizer.on_step_start(step)
    await asyncio.sleep(random.uniform(0.5, 2.0))

    # Randomly succeed or fail
    success = random.random() > 0.2
    if success:
        await visualizer.on_step_complete(step)
    else:
        await visualizer.on_step_error(step, Exception("Random failure"))

    return success


async def main():
    """Run the visualization demo."""
    # Create visualizer
    visualizer = LiveVisualizer()

    # Start server
    await visualizer.start_server()

    try:
        # Create some test steps
        steps = []
        for i in range(5):
            prompt = BasePrompt(
                name=f"test_prompt_{i}",
                description=f"Test prompt {i}",
                system_prompt="You are a test system.",
                user_prompt=f"This is test {i}",
                metadata=PromptMetadata(type="test"),
                variables=[],
            )
            step = ChainNode(
                step=ChainStep(prompt=prompt),
                node_type=NodeType.AGENT,
                metadata=ChainMetadata(description=f"Test step {i}")
            )
            steps.append(step)

        # Simulate execution
        for step in steps:
            success = await simulate_step_execution(step, visualizer)
            if not success:
                # Retry failed steps
                await asyncio.sleep(1.0)
                await simulate_step_execution(step, visualizer)

    finally:
        # Stop server
        await visualizer.stop_server()


if __name__ == "__main__":
    asyncio.run(main())
