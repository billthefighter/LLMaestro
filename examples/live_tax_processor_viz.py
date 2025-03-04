#!/usr/bin/env python3
import asyncio
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import websockets
from datetime import datetime

from pdf_tax_processor import PDFTaxProcessor, TaxableItem, TaxableItems
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.core.conversations import ConversationGraph, ConversationContext
from llmaestro.visualization.cytoscape_renderer import CytoscapeRenderer, CytoscapeStyle
from llmaestro.llm.credentials import APIKey
from llmaestro.default_library.default_llm_factory import LLMDefaultFactory
from llmaestro.visualization.base_visualizer import BaseVisualizer, CytoscapeNode, CytoscapeEdge, CytoscapeGraph
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.prompts.types import PromptMetadata
from llmaestro.core.models import LLMResponse
from llmaestro.core.orchestrator import Orchestrator

class LiveTaxProcessorVisualizer:
    """Visualizes PDF tax processor conversations in real-time."""

    def __init__(self, port: int = 8765):
        self.port = port
        self.renderer = CytoscapeRenderer()
        self.active_conversations: Dict[str, ConversationGraph] = {}
        self.connected_clients = set()
        self.server = None

        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Add custom styles for tax processing visualization
        self.custom_styles = [
            CytoscapeStyle(
                "node[type = 'prompt']",
                {
                    "background-color": "#E3F2FD",
                    "border-color": "#1976D2",
                    "border-width": "2px",
                }
            ),
            CytoscapeStyle(
                "node[type = 'response']",
                {
                    "background-color": "#E8F5E9",
                    "border-color": "#43A047",
                    "border-width": "2px",
                }
            ),
            CytoscapeStyle(
                "node[status = 'processing']",
                {
                    "background-color": "#FFF3E0",
                    "border-color": "#F57C00",
                    "border-width": "3px",
                }
            ),
            CytoscapeStyle(
                "node[status = 'completed']",
                {
                    "background-color": "#E8F5E9",
                    "border-color": "#43A047",
                    "border-width": "2px",
                }
            ),
            CytoscapeStyle(
                "node[status = 'error']",
                {
                    "background-color": "#FFEBEE",
                    "border-color": "#D32F2F",
                    "border-width": "2px",
                }
            )
        ]

    def _conversation_to_cytoscape(self, conversation: ConversationGraph) -> Dict[str, Any]:
        """Convert conversation graph to Cytoscape format."""
        nodes = []
        edges = []

        # Add nodes
        for node_id, node in conversation.nodes.items():
            # Determine node status
            status = node.metadata.get("execution", {}).get("status", "unknown")

            # Create node content snippet
            content_snippet = ""
            if isinstance(node.content, LLMResponse):
                content_snippet = node.content.content[:50] + "..." if len(node.content.content) > 50 else node.content.content
                content = node.content.content
            elif hasattr(node.content, "user_prompt"):
                content_snippet = node.content.user_prompt[:50] + "..." if len(node.content.user_prompt) > 50 else node.content.user_prompt
                content = node.content.user_prompt
            else:
                content = str(node.content)
                content_snippet = content[:50] + "..." if len(content) > 50 else content

            # Format created time
            created_time = node.created_at.strftime("%H:%M:%S")

            # Estimate token count (rough estimate: ~4 chars per token)
            estimated_tokens = len(content) // 4

            # Determine node color based on type and status
            if node.node_type == "prompt":
                bg_color = "#E3F2FD"
                border_color = "#1976D2"
            elif node.node_type == "response":
                bg_color = "#E8F5E9"
                border_color = "#43A047"
            elif status == "processing" or status == "running":
                bg_color = "#FFF9C4"
                border_color = "#FBC02D"
            elif status == "error":
                bg_color = "#FFEBEE"
                border_color = "#E53935"
            else:
                bg_color = "#F5F5F5"
                border_color = "#9E9E9E"

            node_data = {
                "id": node_id,
                "type": node.node_type,
                "label": f"{node.node_type.title()}\n{content_snippet}\n{created_time}",
                "status": status,
                "content": content,
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
        for edge in conversation.edges:
            # Create a unique edge ID
            edge_id = f"{edge.source_id}-{edge.target_id}"

            # Log the edge being processed
            self.logger.debug(f"Processing edge: {edge_id} from {edge.source_id} to {edge.target_id}")

            # Ensure source and target nodes exist
            if edge.source_id not in conversation.nodes:
                self.logger.warning(f"Source node {edge.source_id} not found for edge {edge_id}")
                continue

            if edge.target_id not in conversation.nodes:
                self.logger.warning(f"Target node {edge.target_id} not found for edge {edge_id}")
                continue

            edge_data = {
                "id": edge_id,
                "source": edge.source_id,
                "target": edge.target_id,
                "label": edge.edge_type,
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

    async def _broadcast_update(self, conversation_id: str) -> None:
        """Broadcast conversation update to all connected clients."""
        if not self.connected_clients:
            self.logger.info("No clients connected, skipping update")
            return

        conversation = self.active_conversations.get(conversation_id)
        if not conversation:
            self.logger.warning(f"No conversation found with ID {conversation_id}")
            return

        # Prepare visualization data
        graph_data = self._conversation_to_cytoscape(conversation)
        self.logger.debug(f"Broadcasting update for conversation {conversation_id}")
        self.logger.debug(f"Graph data: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")

        # Create complete visualization config
        visualization_data = {
            "type": "graph_update",
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "data": {
                "elements": graph_data,
                "conversations": {conversation_id: True},  # Add conversations field for the template
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

    async def _handle_client(self, websocket):
        """Handle a client connection."""
        self.connected_clients.add(websocket)
        self.logger.info(f"New client connected from {websocket.remote_address}. Total clients: {len(self.connected_clients)}")

        try:
            # Send initial data for all active conversations
            for conversation_id in self.active_conversations:
                await self._broadcast_update(conversation_id)

            # Handle incoming messages
            async for message in websocket:
                self.logger.debug(f"Received message from client: {message}")
                # Handle incoming messages if needed
                try:
                    data = json.loads(message)
                    self.logger.info(f"Received message from client: {data}")
                except json.JSONDecodeError:
                    self.logger.warning(f"Received invalid JSON from client: {message}")
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client connection closed from {websocket.remote_address}")
        except Exception as e:
            self.logger.error(f"Error handling client connection: {e}")
        finally:
            self.connected_clients.remove(websocket)
            self.logger.info(f"Client disconnected. Total clients: {len(self.connected_clients)}")

    async def start(self):
        """Start the visualization server."""
        if self.server is not None:
            self.logger.warning("Server is already running")
            return

        self.server = await websockets.serve(self._handle_client, "localhost", self.port)
        self.logger.info(f"Visualization server started on ws://localhost:{self.port}")

    async def stop(self):
        """Stop the visualization server."""
        if self.server is None:
            self.logger.warning("Server is not running")
            return

        self.server.close()
        await self.server.wait_closed()
        self.server = None
        self.logger.info("Visualization server stopped")

    def track_conversation(self, conversation: ConversationGraph) -> None:
        """Start tracking a conversation for visualization."""
        self.logger.info(f"Starting to track conversation {conversation.id}")
        self.logger.debug(f"Initial conversation state: {len(conversation.nodes)} nodes, {len(conversation.edges)} edges")

        # Calculate token usage
        token_usage = conversation.total_tokens
        self.logger.info(f"Initial token usage: {token_usage.total_tokens} tokens ({token_usage.prompt_tokens} prompt, {token_usage.completion_tokens} completion)")

        self.active_conversations[conversation.id] = conversation
        asyncio.create_task(self._broadcast_update(conversation.id))

    def update_conversation(self, conversation_id: str, conversation: ConversationGraph) -> None:
        """Update the state of a tracked conversation."""
        self.logger.info(f"Updating conversation {conversation_id}")
        self.logger.debug(f"Updated conversation state: {len(conversation.nodes)} nodes, {len(conversation.edges)} edges")

        # Calculate token usage
        token_usage = conversation.total_tokens
        self.logger.info(f"Current token usage: {token_usage.total_tokens} tokens ({token_usage.prompt_tokens} prompt, {token_usage.completion_tokens} completion)")

        self.active_conversations[conversation_id] = conversation
        asyncio.create_task(self._broadcast_update(conversation_id))

class VisualizedPDFTaxProcessor(PDFTaxProcessor):
    """PDF Tax Processor with live visualization support."""

    def __init__(self, input_dir: str, output_file: str, llm_registry: LLMRegistry,
                 model_name: str, visualizer: Optional[LiveTaxProcessorVisualizer] = None):
        super().__init__(input_dir, output_file, llm_registry, model_name)
        self.visualizer = visualizer
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize the processor components and connect visualization events."""
        await super().initialize()
        if self.visualizer and self.orchestrator:
            self.logger.info("Visualization enabled - connecting visualization callbacks")
            # Set up visualization callbacks
            self.orchestrator.conversation_created_callback = self._on_conversation_created
            self.orchestrator.conversation_updated_callback = self._on_conversation_updated
            self.orchestrator.node_added_callback = self._on_node_added
            self.orchestrator.node_updated_callback = self._on_node_updated

    async def _on_conversation_created(self, conversation: ConversationGraph) -> None:
        """Handle new conversation creation."""
        if self.visualizer:
            self.logger.info(f"New conversation created: {conversation.id}")
            self.visualizer.track_conversation(conversation)

    async def _on_conversation_updated(self, conversation: ConversationGraph) -> None:
        """Handle conversation updates."""
        if self.visualizer:
            self.logger.debug(f"Conversation updated: {conversation.id}")
            self.visualizer.update_conversation(conversation.id, conversation)

    async def _on_node_added(self, conversation_id: str, node_id: str) -> None:
        """Handle new node addition."""
        if self.visualizer and self.orchestrator:
            conversation = self.orchestrator._get_conversation(conversation_id)
            if conversation:
                self.visualizer.update_conversation(conversation_id, conversation)

    async def _on_node_updated(self, conversation_id: str, node_id: str) -> None:
        """Handle node updates."""
        if self.visualizer and self.orchestrator:
            conversation = self.orchestrator._get_conversation(conversation_id)
            if conversation:
                self.visualizer.update_conversation(conversation_id, conversation)

    async def _process_single_pdf(self, pdf_path: Path) -> Tuple[List[TaxableItem], Path]:
        """Process a single PDF file using the orchestrator for conversation management."""
        all_items: List[TaxableItem] = []

        # Create initial prompt
        initial_prompt = MemoryPrompt(
            name="tax_extraction_init",
            description="Initialize tax document processing",
            system_prompt="You are processing a multi-page tax document. Maintain context across pages.",
            user_prompt="Beginning tax document processing.",
            metadata=PromptMetadata(type="tax_extraction_init")
        )

        # Create conversation through orchestrator
        conversation = await self.orchestrator.create_conversation(
            name=f"tax_processing_{pdf_path.stem}",
            initial_prompt=initial_prompt,
            metadata={"source_file": str(pdf_path)}
        )

        try:
            # Convert PDF to images
            png_files = self._convert_pdf_to_png(pdf_path)

            # Process each page
            for i, png_path in enumerate(png_files, 1):
                image_attachment = self._create_image_attachment(png_path)

                # Create page-specific prompt
                prompt = self._create_pdf_prompt(
                    image_attachment,
                    self._create_response_format(),
                    page_number=i,
                    total_pages=len(png_files)
                )

                # Execute prompt through orchestrator
                response_id = await self.orchestrator.execute_prompt(
                    conversation=conversation,
                    prompt=prompt
                )

                # Get response and process items
                response_node = conversation.nodes[response_id]
                if isinstance(response_node.content, LLMResponse) and response_node.content.success:
                    try:
                        items_data = json.loads(response_node.content.content)
                        items = TaxableItems.model_validate(items_data).items
                        all_items.extend(items)
                        self.logger.info(f"Extracted {len(items)} items from page {i}")
                    except Exception as e:
                        self.logger.error(f"Error parsing items from page {i}: {e}")

                # Clean up PNG file
                png_path.unlink()

            # Save conversation dump
            dump_file = self._save_conversation_dump(
                conversation.id,
                pdf_path.stem,
                conversation
            )

            return all_items, dump_file

        except Exception as e:
            self.logger.error(f"Error processing {pdf_path.name}: {str(e)}")
            raise

async def main():
    """Example usage of visualized PDF tax processor."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run PDF Tax Processor with live visualization')
    parser.add_argument('--api-key', required=True, help='API key for the LLM service')
    parser.add_argument('--provider', default='openai', help='Provider name (e.g., openai, anthropic)')
    parser.add_argument('--input-dir', default='tax_receipts', help='Directory containing PDF files to process')
    parser.add_argument('--output-file', default='tax_items.csv', help='Output CSV file path')
    parser.add_argument('--model-name', default='gpt-4o-mini-2024-07-18', help='Name of the LLM model to use')
    parser.add_argument('--port', type=int, default=8765, help='Port for visualization server')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--allow-model-failover', action='store_true', help='Allow model failover')

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Initialize visualization server
    visualizer = LiveTaxProcessorVisualizer(port=args.port)
    await visualizer.start()

    # Print instructions for the user
    print("\n" + "="*80)
    print("Visualization server is running!")
    print("="*80)
    print(f"\nTo view the visualization:")
    print("1. Open examples/templates/tax_processor_viz.html in your web browser")
    print("2. You should see 'Connected' status in the top-left corner")
    print("\nThe visualization will show the conversation graph in real-time as PDFs are processed.")
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
        # Initialize LLM registry with credentials using the factory
        credentials = {args.provider: APIKey(key=args.api_key)}
        factory = LLMDefaultFactory(credentials=credentials)

        try:
            llm_registry = await factory.DefaultLLMRegistryFactory()
            logger.info(f"Successfully initialized LLM registry with provider: {args.provider}")

            # List available models
            available_models = llm_registry.get_registered_models()
            logger.info(f"Available models: {available_models}")

            # Model selection logic
            if args.model_name not in available_models:
                logger.warning(f"Requested model '{args.model_name}' not found in available models.")
                if getattr(args, 'allow_model_failover', False):
                    # Only try to find alternative model if failover is explicitly allowed
                    for model in available_models:
                        if not model.endswith('vision-preview'):
                            args.model_name = model
                            logger.info(f"Using '{model}' instead.")
                            break
                    else:
                        raise ValueError("No suitable non-vision models available in the registry")
                else:
                    raise ValueError(f"Requested model '{args.model_name}' not found in available models and failover is not enabled")

            # Create and run processor with visualization
            processor = VisualizedPDFTaxProcessor(
                input_dir=args.input_dir,
                output_file=args.output_file,
                llm_registry=llm_registry,
                model_name=args.model_name,
                visualizer=visualizer
            )

            print(f"\nStarting PDF processing...")
            print(f"Input directory: {args.input_dir}")
            print(f"Output file: {args.output_file}")
            print(f"Provider: {args.provider}")
            print(f"Requested model: {args.model_name}")

            # Initialize the processor
            await processor.initialize()

            # Verify the actual model being used
            if processor.llm and processor.llm.state:
                actual_model = processor.llm.state.profile.name
                if actual_model != args.model_name:
                    print(f"WARNING: Using model '{actual_model}' instead of requested '{args.model_name}'")
                else:
                    print(f"Using model: {args.model_name}")
            else:
                print(f"Using model: {args.model_name}")

            await processor.process_pdfs()

        except ImportError as e:
            logger.error(f"Missing required dependency: {e}")
            logger.error("Please install required packages: pip install pdf2image")
        except Exception as e:
            logger.error(f"Error processing PDFs: {e}")
            raise

    finally:
        # Clean up visualization server
        await visualizer.stop()

if __name__ == "__main__":
    asyncio.run(main())
