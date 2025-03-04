#!/usr/bin/env python
"""
Mock Conversation Visualizer for Testing

This script creates a simple visualization server with mock conversation data
to test the visualization without relying on the actual conversation classes.
"""

import asyncio
import json
import logging
import random
import sys
import uuid
import websockets
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mock_conversation_visualizer")

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

class MockConversationVisualizer:
    """A simple visualization server with mock conversation data."""

    def __init__(self, port: int = 8765):
        self.port = port
        self.connected_clients = set()
        self.conversations = {}
        self.server = None
        self.logger = logging.getLogger("mock_conversation_visualizer")

    async def start(self):
        """Start the visualization server."""
        self.server = await websockets.serve(
            self._handle_client, "localhost", self.port
        )
        self.logger.info(f"Visualization server started on ws://localhost:{self.port}")

        # Create a mock conversation
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = {
            "id": conversation_id,
            "nodes": {},
            "edges": []
        }
        self.logger.info(f"Created mock conversation {conversation_id}")

        # Start adding nodes
        asyncio.create_task(self._add_nodes_periodically(conversation_id))

    async def stop(self):
        """Stop the visualization server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("Visualization server stopped")

    async def _handle_client(self, websocket):
        """Handle a client connection."""
        self.connected_clients.add(websocket)
        self.logger.info(f"Client connected, total clients: {len(self.connected_clients)}")

        try:
            # Send initial data
            for conversation_id in self.conversations:
                await self._broadcast_update(conversation_id)

            # Keep the connection open
            async for message in websocket:
                # Process any client messages if needed
                data = json.loads(message)
                self.logger.info(f"Received message from client: {data}")
        except Exception as e:
            self.logger.error(f"Error handling client: {e}")
        finally:
            self.connected_clients.remove(websocket)
            self.logger.info(f"Client disconnected, remaining clients: {len(self.connected_clients)}")

    async def _add_nodes_periodically(self, conversation_id: str):
        """Add nodes to the conversation periodically."""
        conversation = self.conversations[conversation_id]
        node_count = 0

        while True:
            # Add a prompt node
            prompt_id = str(uuid.uuid4())
            prompt_text = random.choice(SAMPLE_PROMPTS)
            conversation["nodes"][prompt_id] = {
                "id": prompt_id,
                "content": prompt_text,
                "node_type": "prompt",
                "created_at": datetime.now().isoformat(),
                "metadata": {
                    "execution": {"status": "completed"},
                    "source": "mock_generator"
                }
            }
            node_count += 1
            self.logger.info(f"Added prompt node to conversation {conversation_id}, now has {node_count} nodes")

            # Broadcast the update
            await self._broadcast_update(conversation_id)

            # Wait a bit
            await asyncio.sleep(1)

            # Add a response node
            response_id = str(uuid.uuid4())
            response_text = random.choice(SAMPLE_RESPONSES)
            conversation["nodes"][response_id] = {
                "id": response_id,
                "content": response_text,
                "node_type": "response",
                "created_at": datetime.now().isoformat(),
                "metadata": {
                    "execution": {"status": "completed"},
                    "source": "mock_generator"
                }
            }
            node_count += 1

            # Add an edge from prompt to response
            conversation["edges"].append({
                "source_id": prompt_id,
                "target_id": response_id,
                "edge_type": "prompt_response"
            })

            self.logger.info(f"Added response node to conversation {conversation_id}, now has {node_count} nodes")

            # Broadcast the update
            await self._broadcast_update(conversation_id)

            # Wait a bit
            await asyncio.sleep(1)

    async def _broadcast_update(self, conversation_id: str):
        """Broadcast a conversation update to all connected clients."""
        if not self.connected_clients:
            self.logger.info("No clients connected, skipping update")
            return

        conversation = self.conversations.get(conversation_id)
        if not conversation:
            self.logger.warning(f"No conversation found with ID {conversation_id}")
            return

        # Convert to cytoscape format
        graph_data = self._conversation_to_cytoscape(conversation)
        self.logger.debug(f"Broadcasting update for conversation {conversation_id}")
        self.logger.debug(f"Graph data: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")

        # Create visualization data
        visualization_data = {
            "type": "graph_update",
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "data": {
                "elements": graph_data,
                "conversations": {conversation_id: True},
                "style": [
                    {
                        "selector": "node",
                        "style": {
                            "label": "data(label)",
                            "text-wrap": "wrap",
                            "text-max-width": "180px"
                        }
                    },
                    {
                        "selector": "edge",
                        "style": {
                            "label": "data(label)",
                            "curve-style": "bezier",
                            "target-arrow-shape": "triangle"
                        }
                    }
                ],
                "layout": {
                    "name": "dagre",
                    "rankDir": "LR",
                    "spacingFactor": 1.2,
                    "rankSep": 200,
                    "nodeSep": 100,
                    "edgeSep": 50,
                    "animate": True
                }
            }
        }

        # Broadcast to all clients
        websockets_clients = [ws for ws in self.connected_clients]
        if websockets_clients:
            self.logger.debug(f"Broadcasting to {len(websockets_clients)} clients")
            try:
                await asyncio.gather(
                    *[client.send(json.dumps(visualization_data)) for client in websockets_clients]
                )
                self.logger.debug("Broadcast completed successfully")
            except Exception as e:
                self.logger.error(f"Error broadcasting update: {e}")
        else:
            self.logger.warning("No websocket clients available for broadcast")

    def _conversation_to_cytoscape(self, conversation: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Convert conversation to Cytoscape format."""
        nodes = []
        edges = []

        # Add nodes
        for node_id, node in conversation["nodes"].items():
            # Create node content snippet
            content_snippet = node["content"][:50] + "..." if len(node["content"]) > 50 else node["content"]

            # Get created time
            created_time = datetime.fromisoformat(node["created_at"]).strftime("%H:%M:%S")

            # Estimate token count (rough estimate: ~4 chars per token)
            estimated_tokens = len(node["content"]) // 4

            # Determine node color based on type
            if node["node_type"] == "prompt":
                bg_color = "#E3F2FD"
                border_color = "#1976D2"
            elif node["node_type"] == "response":
                bg_color = "#E8F5E9"
                border_color = "#43A047"
            elif node["node_type"] == "processing":
                bg_color = "#FFF9C4"
                border_color = "#FBC02D"
            elif node["node_type"] == "error":
                bg_color = "#FFEBEE"
                border_color = "#E53935"
            else:
                bg_color = "#F5F5F5"
                border_color = "#9E9E9E"

            node_data = {
                "id": node_id,
                "type": node["node_type"],
                "label": f"{node['node_type'].title()}\n{content_snippet}\n{created_time}",
                "status": node["metadata"].get("execution", {}).get("status", "unknown"),
                "content": node["content"],
                "tokens": estimated_tokens,
                # Add style properties directly in node data
                "style": {
                    "background-color": bg_color,
                    "border-color": border_color,
                    "border-width": "2px",
                    "width": "200px",
                    "height": "75px",
                    "font-size": "12px",
                    "text-wrap": "wrap",
                    "text-valign": "center",
                    "text-halign": "center"
                }
            }
            nodes.append({"data": node_data})

        # Add edges
        for edge in conversation["edges"]:
            # Create a unique edge ID if not present
            edge_id = f"{edge['source_id']}-{edge['target_id']}"

            # Log the edge being processed
            self.logger.debug(f"Processing edge: {edge_id} from {edge['source_id']} to {edge['target_id']}")

            # Ensure source and target nodes exist
            if edge['source_id'] not in conversation["nodes"]:
                self.logger.warning(f"Source node {edge['source_id']} not found for edge {edge_id}")
                continue

            if edge['target_id'] not in conversation["nodes"]:
                self.logger.warning(f"Target node {edge['target_id']} not found for edge {edge_id}")
                continue

            edge_data = {
                "id": edge_id,
                "source": edge["source_id"],
                "target": edge["target_id"],
                "label": edge["edge_type"],
                "style": {
                    "width": "2px",
                    "line-color": "#757575",
                    "target-arrow-color": "#757575",
                    "curve-style": "bezier",
                    "target-arrow-shape": "triangle"
                }
            }
            edges.append({"data": edge_data})

        return {"nodes": nodes, "edges": edges}

async def main():
    """Main function to run the mock conversation visualizer."""
    visualizer = MockConversationVisualizer(port=8765)

    try:
        await visualizer.start()

        # Keep the server running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        await visualizer.stop()

if __name__ == "__main__":
    asyncio.run(main())
