#!/usr/bin/env python
"""
Dummy Conversation Generator for Testing Visualization

This script creates a dummy conversation with nodes being added every second
to help test the LiveTaxProcessorVisualizer.
"""

import asyncio
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

# Add the parent directory to the path so we can import the llmaestro package
sys.path.append(str(Path(__file__).parent.parent))

# Import from the correct paths
from src.llmaestro.core.conversations import ConversationGraph
from src.llmaestro.prompts.base import BasePrompt
from src.llmaestro.core.models import LLMResponse, TokenUsage
from src.llmaestro.llm.responses import ResponseFormatType
from examples.live_tax_processor_viz import LiveTaxProcessorVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dummy_conversation_generator")

# Sample prompts and responses
SAMPLE_PROMPTS = [
    "Extract tax information from this document.",
    "What is the total income reported?",
    "Find all deductions in this tax form.",
    "Identify the filing status.",
    "Calculate the tax liability.",
    "Verify the social security number format.",
    "Check for any missing information.",
    "Summarize the key tax details.",
]

SAMPLE_RESPONSES = [
    "I found the following tax information: income of $75,000, deductions of $12,000.",
    "The total income reported is $75,000 from all sources.",
    "Deductions include: mortgage interest ($8,000), charitable donations ($2,000), student loan interest ($2,000).",
    "The filing status is 'Single'.",
    "Based on the information provided, the tax liability is approximately $10,500.",
    "The SSN format is valid: XXX-XX-XXXX.",
    "Missing information: employer identification number and some itemized deductions.",
    "Key tax details: $75,000 income, $12,000 deductions, $10,500 tax liability, filing as Single.",
]

class DummyPrompt(BasePrompt):
    """Dummy prompt for testing."""
    user_prompt: str
    name: str = "Dummy Prompt"
    description: str = "A dummy prompt for testing"
    system_prompt: str = "You are a helpful assistant."

    def get_prompt_text(self) -> str:
        return self.user_prompt

    async def save(self) -> bool:
        """Dummy implementation of abstract method."""
        return True

    @classmethod
    async def load(cls, identifier: str) -> Optional["BasePrompt"]:
        """Dummy implementation of abstract method."""
        return None

class DummyConversationGenerator:
    """Generates dummy conversations for testing visualization."""

    def __init__(self, visualizer: LiveTaxProcessorVisualizer, conversation_count: int = 1):
        self.visualizer = visualizer
        self.conversation_count = conversation_count
        self.conversations = {}
        self.running = False

    async def start(self):
        """Start generating dummy conversations."""
        self.running = True

        # Create initial conversations
        for _ in range(self.conversation_count):
            conversation_id = str(uuid4())
            conversation = ConversationGraph(id=conversation_id)
            self.conversations[conversation_id] = conversation

            # Track the conversation in the visualizer
            self.visualizer.track_conversation(conversation)
            logger.info(f"Created conversation {conversation_id}")

        # Start adding nodes to conversations
        while self.running:
            for conversation_id, conversation in self.conversations.items():
                await self._add_random_node(conversation_id, conversation)

                # Update the visualizer
                self.visualizer.update_conversation(conversation_id, conversation)
                logger.info(f"Updated conversation {conversation_id}, now has {len(conversation.nodes)} nodes")

            # Wait before adding more nodes
            await asyncio.sleep(1)

    async def stop(self):
        """Stop generating dummy conversations."""
        self.running = False

    async def _add_random_node(self, conversation_id: str, conversation: ConversationGraph):
        """Add a random node to the conversation."""
        # Determine if this should be a prompt or response
        is_prompt = len(conversation.nodes) % 2 == 0

        if is_prompt:
            # Create a prompt node
            prompt_text = random.choice(SAMPLE_PROMPTS)
            prompt = DummyPrompt(user_prompt=prompt_text)

            node_id = conversation.add_conversation_node(
                content=prompt,
                node_type="prompt",
                metadata={
                    "execution": {"status": "completed"},
                    "source": "dummy_generator"
                }
            )

            logger.debug(f"Added prompt node {node_id} to conversation {conversation_id}")
            return node_id
        else:
            # Create a response node
            response_text = random.choice(SAMPLE_RESPONSES)

            # Get the last node (which should be a prompt)
            last_node_id = list(conversation.nodes.keys())[-1]

            # Create a response
            response = LLMResponse(
                content=response_text,
                success=True,
                token_usage=TokenUsage(
                    prompt_tokens=random.randint(10, 50),
                    completion_tokens=random.randint(20, 100),
                    total_tokens=random.randint(30, 150)
                ),
                execution_time=random.uniform(0.5, 2.0),
                metadata={"source": "dummy_generator"}
            )

            # Add the response node
            node_id = conversation.add_conversation_node(
                content=response,
                node_type="response",
                metadata={
                    "execution": {"status": "completed"},
                    "source": "dummy_generator"
                }
            )

            # Add an edge from the prompt to the response
            conversation.add_conversation_edge(
                source_id=last_node_id,
                target_id=node_id,
                edge_type="prompt_response"
            )

            logger.debug(f"Added response node {node_id} to conversation {conversation_id}")
            return node_id

async def main():
    """Main function to run the dummy conversation generator."""
    # Create the visualizer
    visualizer = LiveTaxProcessorVisualizer(port=8765)

    # Start the visualizer
    await visualizer.start()
    logger.info("Visualizer started")

    try:
        # Create the dummy conversation generator
        generator = DummyConversationGenerator(visualizer, conversation_count=2)

        # Start generating conversations
        await generator.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        # Stop the visualizer
        await visualizer.stop()
        logger.info("Visualizer stopped")

if __name__ == "__main__":
    asyncio.run(main())
