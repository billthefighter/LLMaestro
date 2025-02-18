import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Set

import websockets.legacy.server
from websockets.legacy.server import WebSocketServer, WebSocketServerProtocol, serve

from src.core.logging_config import configure_logging
from src.llm.chains import ChainStep
from src.visualization.chain_visualizer import ChainVisualizer

# Configure module logger
logger = configure_logging(module_name=__name__)


@dataclass
class StepMetadata:
    """Metadata for a chain step."""

    status: str = "pending"
    error: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


class LiveChainVisualizer:
    """Real-time chain visualization using WebSockets."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.chain_visualizer = ChainVisualizer()
        self.server: Optional[WebSocketServer] = None
        self.step_metadata: Dict[str, StepMetadata] = {}
        logger.info(f"LiveChainVisualizer initialized with host={host}, port={port}")

    async def register(self, websocket: WebSocketServerProtocol):
        """Register a new client connection."""
        self.clients.add(websocket)
        logger.info(f"New client connected. Total clients: {len(self.clients)}")
        # Send initial graph state
        await self.send_update(websocket)

    async def unregister(self, websocket: WebSocketServerProtocol):
        """Unregister a client connection."""
        self.clients.remove(websocket)
        logger.info(f"Client disconnected. Remaining clients: {len(self.clients)}")

    async def send_update(self, websocket: WebSocketServerProtocol):
        """Send current graph state to a client."""
        graph_data = self.chain_visualizer.get_visualization_data()
        message = {"type": "graph_update", "data": asdict(graph_data)}
        logger.debug(f"Sending graph update to client: {json.dumps(message)[:200]}...")
        await websocket.send(json.dumps(message))

    async def broadcast_update(self):
        """Send current graph state to all connected clients."""
        if not self.clients:
            logger.debug("No clients connected, skipping broadcast")
            return

        graph_data = self.chain_visualizer.get_visualization_data()
        message = json.dumps({"type": "graph_update", "data": asdict(graph_data)})

        logger.debug(f"Broadcasting update to {len(self.clients)} clients")
        # Broadcast to all connected clients
        await asyncio.gather(*[client.send(message) for client in self.clients], return_exceptions=True)

    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Handle individual client WebSocket connection."""
        await self.register(websocket)
        try:
            async for message in websocket:
                logger.debug(f"Received message from client: {message[:200]}...")
                # Handle any client messages if needed
                pass
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        finally:
            await self.unregister(websocket)

    async def start_server(self):
        """Start the WebSocket server."""
        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        self.server = await serve(self.handle_client, self.host, self.port)
        logger.info("WebSocket server started successfully")

    async def stop_server(self):
        """Stop the WebSocket server."""
        if self.server:
            logger.info("Stopping WebSocket server...")
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")

    # Chain event handlers
    async def on_step_start(self, step: ChainStep):
        """Handle step start event."""
        logger.info(f"Step started: {step.task_type}")
        metadata = StepMetadata(status="running")
        self.step_metadata[step.id] = metadata
        self.chain_visualizer._process_chain_step(step)
        await self.broadcast_update()

    async def on_step_complete(self, step: ChainStep):
        """Handle step completion event."""
        logger.info(f"Step completed: {step.task_type}")
        if step.id in self.step_metadata:
            self.step_metadata[step.id].status = "completed"
        await self.broadcast_update()

    async def on_step_error(self, step: ChainStep, error: Exception):
        """Handle step error event."""
        logger.error(f"Step error in {step.task_type}: {error}")
        if step.id in self.step_metadata:
            metadata = self.step_metadata[step.id]
            metadata.status = "error"
            metadata.error = str(error)
        await self.broadcast_update()
