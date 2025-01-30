# Visualization Module

This module provides tools for visualizing LLM Orchestrator's chain execution patterns using Cytoscape.js. It generates interactive visualizations that help understand how complex tasks are broken down and processed.

## Components

- `chain_visualizer.py`: Core visualization logic that converts chain structures to Cytoscape graphs
- `cytoscape_renderer.py`: Handles HTML rendering of Cytoscape visualizations
- `cytoscape_config.py`: Configuration for visualization styling and layout
- `visualize.py`: High-level interface for generating visualizations
- `examples.py`: Example chain patterns for demonstration

## Supported Chain Types

1. **Sequential Chain**: Linear task decomposition where steps are executed in order
2. **Parallel Chain**: Concurrent task execution for independent steps
3. **Chord Chain**: Complex dependencies between steps with cross-connections
4. **Group Chain**: Logical grouping of related steps
5. **Map Chain**: Map-reduce style processing patterns
6. **Reminder Chain**: Steps with temporal dependencies or checkpoints
7. **Recursive Chain**: Self-referential task decomposition
8. **Nested Chain**: Hierarchical organization of steps

## Usage

```python
from llm_orchestrator.visualization import ChainVisualizer
from llm_orchestrator.llm.chains import SequentialChain

# Create a chain
chain = SequentialChain(...)

# Initialize visualizer
visualizer = ChainVisualizer()

# Process the chain
visualizer.process_chain(chain)

# Get visualization data
graph_data = visualizer.get_visualization_data()

# Render to HTML
from llm_orchestrator.visualization import render_chain_visualization
render_chain_visualization(graph_data, "output.html")
```

## Visualization Features

- Interactive graph layout with draggable nodes
- Automatic layout using Dagre algorithm
- Color coding for different step types and states
- Edge labels showing relationships between steps
- Hover tooltips with detailed step information
- Zoom and pan controls
- Responsive design that works on different screen sizes

## Customization

The visualization appearance can be customized through `cytoscape_config.py`:
- Node styles (colors, shapes, sizes)
- Edge styles (width, color, arrow style)
- Layout parameters (spacing, orientation)
- Animation settings
- Interaction behaviors

## Example Visualizations

Example visualizations for each chain type can be found in `examples/visualizations/`:
- `sequential_chain.html`
- `parallel_chain.html`
- `chord_chain.html`
- `group_chain.html`
- `map_chain.html`
- `reminder_chain.html`
- `recursive_chain.html`
- `nested_chain.html`

These examples demonstrate different task decomposition patterns and can serve as references for understanding the visualization capabilities.

## Live Visualization

The visualization module can be extended to support real-time visualization of running chains. Here's how to implement it:

### Method 1: WebSocket-Based Live Updates

```python
from llm_orchestrator.visualization import LiveChainVisualizer
import asyncio

# Initialize live visualizer with WebSocket server
live_viz = LiveChainVisualizer(port=8765)

# Start visualization server
await live_viz.start_server()

# Create a chain with live visualization
chain = SequentialChain(
    on_step_start=live_viz.on_step_start,
    on_step_complete=live_viz.on_step_complete,
    on_step_error=live_viz.on_step_error
)

# Run chain (visualization updates automatically)
await chain.run()
```

Client-side HTML will automatically reconnect if connection is lost:
```html
<script>
    const ws = new WebSocket('ws://localhost:8765');
    ws.onmessage = (event) => {
        const update = JSON.parse(event.data);
        cy.json(update); // Update Cytoscape graph
    };
    // Auto-reconnect logic
    ws.onclose = () => {
        setTimeout(() => {
            new WebSocket('ws://localhost:8765');
        }, 1000);
    };
</script>
```

### Method 2: Server-Sent Events (SSE)

For simpler cases where bi-directional communication isn't needed:

```python
from llm_orchestrator.visualization import SSEChainVisualizer
from fastapi import FastAPI

app = FastAPI()
live_viz = SSEChainVisualizer(app)

# Chain with SSE updates
chain = SequentialChain(
    on_step_start=live_viz.push_update,
    on_step_complete=live_viz.push_update,
    on_step_error=live_viz.push_update
)
```

### Features

- Real-time node and edge updates
- Step status indicators (pending, running, completed, error)
- Progress animations
- Live metrics (execution time, resource usage)
- Event timeline
- Automatic layout adjustments as chain grows

### Implementation Notes

1. **State Management**:
   - Maintain graph state on server
   - Send delta updates to minimize network traffic
   - Support multiple concurrent viewers

2. **Performance Considerations**:
   - Batch updates for high-frequency changes
   - Throttle update rate if needed
   - Prune completed branches for long-running chains

3. **Error Handling**:
   - Graceful degradation if connection lost
   - Automatic state recovery on reconnect
   - Error state visualization

4. **Security**:
   - Optional authentication for visualization endpoint
   - Rate limiting
   - Configurable CORS policies
