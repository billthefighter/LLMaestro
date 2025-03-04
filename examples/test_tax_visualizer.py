#!/usr/bin/env python3
import asyncio
import json
import logging
import uuid
import sys
import os
from datetime import datetime
from typing import Dict, Any, List
import random

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llmaestro.core.conversations import ConversationGraph, ConversationNode, ConversationEdge
from llmaestro.core.models import LLMResponse, TokenUsage
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.prompts.types import PromptMetadata

# Import the LiveTaxProcessorVisualizer using a relative import
from examples.live_tax_processor_viz import LiveTaxProcessorVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockTaxConversationGenerator:
    """Generates mock tax processing conversations for visualization testing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sample_prompts = [
            "Extract tax information from this receipt.",
            "Identify taxable items in this invoice.",
            "What is the total tax amount in this document?",
            "Categorize the expenses in this receipt.",
            "Find all business expenses in this document.",
        ]
        self.sample_responses = [
            "I've identified the following taxable items: Office supplies ($45.99), Equipment rental ($120.00).",
            "The document contains 3 taxable items with a total tax amount of $18.75.",
            "The expenses are categorized as: Office Supplies, Travel, and Equipment.",
            "Total tax amount: $24.50. Tax rate: 8.25%.",
            "Business expenses found: Computer equipment, Software licenses, Office furniture.",
        ]

    def create_mock_conversation(self, conversation_id: str = "") -> ConversationGraph:
        """Create a mock conversation with random nodes and edges."""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        self.logger.info(f"Creating mock conversation with ID: {conversation_id}")

        # Create conversation graph
        conversation = ConversationGraph(id=conversation_id)

        # Add initial nodes
        root_node_id = str(uuid.uuid4())
        prompt_text = random.choice(self.sample_prompts)

        # Create a proper prompt object
        prompt_content = MemoryPrompt(
            name="tax_extraction",
            description="Extract tax information",
            system_prompt="You are processing a tax document.",
            user_prompt=prompt_text,
            metadata=PromptMetadata(type="tax_extraction")
        )

        # Create root node
        root_node = ConversationNode(
            id=root_node_id,
            node_type="prompt",
            content=prompt_content,
            created_at=datetime.now(),
            metadata={
                "execution": {
                    "status": "completed",
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat()
                }
            }
        )
        conversation.add_node(root_node)

        # Create response node
        response_node_id = str(uuid.uuid4())
        response_text = random.choice(self.sample_responses)
        response_content = LLMResponse(
            content=response_text,
            success=True,
            token_usage=TokenUsage(
                prompt_tokens=len(prompt_text) // 4,
                completion_tokens=len(response_text) // 4,
                total_tokens=(len(prompt_text) + len(response_text)) // 4
            )
        )

        response_node = ConversationNode(
            id=response_node_id,
            node_type="response",
            content=response_content,
            created_at=datetime.now(),
            metadata={
                "execution": {
                    "status": "completed",
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat()
                }
            }
        )
        conversation.add_node(response_node)

        # Add edge between nodes
        edge = ConversationEdge(
            source_id=root_node_id,
            target_id=response_node_id,
            edge_type="response"
        )
        conversation.add_edge(edge)

        return conversation

    async def add_nodes_periodically(self, conversation: ConversationGraph, visualizer: LiveTaxProcessorVisualizer):
        """Add nodes to the conversation periodically and update the visualizer."""
        last_node_id = list(conversation.nodes.keys())[-1]

        for i in range(10):  # Add 10 pairs of nodes
            # Add prompt node
            prompt_node_id = str(uuid.uuid4())
            prompt_text = random.choice(self.sample_prompts)

            # Create a proper prompt object
            prompt_content = MemoryPrompt(
                name="tax_extraction",
                description="Extract tax information",
                system_prompt="You are processing a tax document.",
                user_prompt=prompt_text,
                metadata=PromptMetadata(type="tax_extraction")
            )

            prompt_node = ConversationNode(
                id=prompt_node_id,
                node_type="prompt",
                content=prompt_content,
                created_at=datetime.now(),
                metadata={
                    "execution": {
                        "status": "completed",
                        "start_time": datetime.now().isoformat(),
                        "end_time": datetime.now().isoformat()
                    }
                }
            )
            conversation.add_node(prompt_node)

            # Add edge from last node to new prompt
            edge1 = ConversationEdge(
                source_id=last_node_id,
                target_id=prompt_node_id,
                edge_type="next"
            )
            conversation.add_edge(edge1)

            # Update visualizer
            visualizer.update_conversation(conversation.id, conversation)
            self.logger.info(f"Added prompt node to conversation {conversation.id}, now has {len(conversation.nodes)} nodes")
            await asyncio.sleep(1)

            # Add response node
            response_node_id = str(uuid.uuid4())
            response_text = random.choice(self.sample_responses)
            response_content = LLMResponse(
                content=response_text,
                success=True,
                token_usage=TokenUsage(
                    prompt_tokens=len(prompt_text) // 4,
                    completion_tokens=len(response_text) // 4,
                    total_tokens=(len(prompt_text) + len(response_text)) // 4
                )
            )

            response_node = ConversationNode(
                id=response_node_id,
                node_type="response",
                content=response_content,
                created_at=datetime.now(),
                metadata={
                    "execution": {
                        "status": "completed",
                        "start_time": datetime.now().isoformat(),
                        "end_time": datetime.now().isoformat()
                    }
                }
            )
            conversation.add_node(response_node)

            # Add edge from prompt to response
            edge2 = ConversationEdge(
                source_id=prompt_node_id,
                target_id=response_node_id,
                edge_type="response"
            )
            conversation.add_edge(edge2)

            # Update last node ID
            last_node_id = response_node_id

            # Update visualizer
            visualizer.update_conversation(conversation.id, conversation)
            self.logger.info(f"Added response node to conversation {conversation.id}, now has {len(conversation.nodes)} nodes")
            await asyncio.sleep(1)

async def main():
    """Run the test visualizer with mock conversations."""
    # Initialize visualizer
    visualizer = LiveTaxProcessorVisualizer(port=8765)
    await visualizer.start()

    # Print instructions
    print("\n" + "="*80)
    print("Test Tax Processor Visualization Server is running!")
    print("="*80)
    print(f"\nTo view the visualization:")
    print("1. Open examples/templates/tax_processor_viz.html in your web browser")
    print("2. You should see 'Connected' status in the top-left corner")
    print("\nVisualization features:")
    print("- Node colors indicate type (blue=prompt, green=response, yellow=processing, red=error)")
    print("- Hover over nodes to see more details")
    print("- The top bar shows statistics including token count estimates")
    print("- The graph layout updates automatically as new nodes are added")
    print("="*80)

    # Wait for user confirmation
    while True:
        response = input("\nHave you opened the visualization page and confirmed it's connected? (yes/no): ").lower()
        if response == 'yes':
            break
        elif response == 'no':
            print("\nPlease open the visualization page and ensure it connects before continuing.")
        else:
            print("\nPlease answer 'yes' or 'no'.")

    try:
        # Create mock conversation generator
        generator = MockTaxConversationGenerator()

        # Create and track initial conversation
        conversation = generator.create_mock_conversation()
        visualizer.track_conversation(conversation)

        # Add nodes periodically
        await generator.add_nodes_periodically(conversation, visualizer)

        # Keep the server running
        print("\nMock conversation generation complete. Visualization server is still running.")
        print("Press Ctrl+C to stop the server.")

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean up
        await visualizer.stop()

if __name__ == "__main__":
    asyncio.run(main())
