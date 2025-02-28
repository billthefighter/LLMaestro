# LLMaestro Visualization

This package provides visualization components for LLMaestro, allowing real-time visualization of chain execution and conversation flows.

## Components

### Base Components

- `BaseVisualizer`: Base class for all visualizers, providing common functionality for graph visualization
- `CytoscapeNode`, `CytoscapeEdge`, `CytoscapeGraph`: Data structures for Cytoscape.js compatibility

### Specialized Visualizers

- `ChainVisualizer`: Visualizes chain execution graphs
- `ConversationVisualizer`: Visualizes conversation graphs
- `LiveVisualizer`: Real-time visualization server that can handle both chain and conversation updates

## Usage

### Basic Example

```python
from llm_orchestrator.visualization import LiveVisualizer

# Create a live visualizer
live_viz = LiveVisualizer(port=8765)

# Start the WebSocket server
await live_viz.start_server()

try:
    # Chain visualization
    await live_viz.on_chain_start(chain)
    await live_viz.on_step_start(step)
    await live_viz.on_step_complete(step)

    # Conversation visualization
    await live_viz.on_conversation_update(conversation)
finally:
    await live_viz.stop_server()
```

### HTML Visualization

1. Create an HTML file with Cytoscape.js:

```html
<!DOCTYPE html>
<html>
<head>
    <title>LLMaestro Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.21.1/cytoscape.min.js"></script>
    <style>
        #cy {
            width: 100%;
            height: 100vh;
            position: absolute;
            top: 0;
            left: 0;
            background-color: #f5f5f5;
        }
        .tooltip {
            position: absolute;
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            display: none;
        }
    </style>
</head>
<body>
    <div id="cy"></div>
    <div class="tooltip"></div>
    <script>
        // Initialize Cytoscape
        const cy = cytoscape({
            container: document.getElementById('cy'),
            style: [
                {
                    selector: 'node',
                    style: {
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'text-wrap': 'wrap',
                        'font-size': '14px',
                        'width': '200px',
                        'height': '75px',
                        'shape': 'roundrectangle',
                        'background-color': '#9E9E9E',
                        'color': 'white',
                        'border-width': '2px',
                        'border-color': '#616161',
                        'text-max-width': '180px'
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'label': 'data(label)',
                        'curve-style': 'bezier',
                        'target-arrow-shape': 'triangle',
                        'line-color': '#757575',
                        'target-arrow-color': '#757575',
                        'width': 2,
                        'font-size': '12px'
                    }
                }
            ],
            layout: {
                name: 'dagre',
                rankDir: 'LR',
                nodeSep: 50,
                rankSep: 100,
                padding: 30
            }
        });

        // Add tooltip functionality
        const tooltip = document.querySelector('.tooltip');
        cy.on('mouseover', 'node', function(e) {
            const node = e.target;
            const data = node.data();
            tooltip.innerHTML = `
                Type: ${data.type}<br>
                Created: ${data.created_at}<br>
                ${data.token_usage ? `Tokens: ${data.token_usage.total_tokens}` : ''}
            `;
            tooltip.style.display = 'block';
            tooltip.style.left = e.renderedPosition.x + 'px';
            tooltip.style.top = e.renderedPosition.y + 'px';
        });
        cy.on('mouseout', 'node', function() {
            tooltip.style.display = 'none';
        });

        // Connect to WebSocket server
        const ws = new WebSocket('ws://localhost:8765');

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'graph_update') {
                // Apply custom styles from the data
                const elements = data.data.nodes.map(node => ({
                    data: {
                        ...node.data,
                        id: node.id,
                        label: node.label
                    },
                    style: node.data.style || {}
                })).concat(data.data.edges.map(edge => ({
                    data: {
                        ...edge.data,
                        id: edge.id,
                        source: edge.source,
                        target: edge.target,
                        label: edge.label
                    },
                    style: edge.data.style || {}
                })));

                cy.elements().remove();
                cy.add(elements);
                cy.layout({
                    name: 'dagre',
                    rankDir: 'LR',
                    nodeSep: 50,
                    rankSep: 100,
                    padding: 30
                }).run();
            }
        };
    </script>
</body>
</html>
```

2. Open the HTML file in a browser while running your visualization server.

## Customization

### Node and Edge Styles

You can customize the appearance of nodes and edges by modifying the Cytoscape.js style definitions in your HTML file:

```javascript
const styles = [
    {
        selector: 'node[type = "step"]',
        style: {
            'background-color': '#6FB1FC',
            'shape': 'roundrectangle'
        }
    },
    {
        selector: 'edge[label = "next"]',
        style: {
            'line-color': '#6FB1FC',
            'target-arrow-color': '#6FB1FC'
        }
    }
];

cy.style(styles);
```

### Layout Options

You can adjust the graph layout by modifying the layout options:

```javascript
cy.layout({
    name: 'dagre',
    rankDir: 'TB',
    nodeSep: 50,
    rankSep: 100,
    padding: 30
}).run();
```

## Advanced Features

### Custom Node Processing

You can create custom visualizers by inheriting from `BaseVisualizer`:

```python
from llmaestro.visualization import BaseVisualizer

class CustomVisualizer(BaseVisualizer[CustomNode, CustomEdge]):
    def _process_node(self, node: CustomNode, parent_id: Optional[str] = None) -> None:
        self._add_node(
            node_id=node.id,
            label=node.label,
            type_=node.type,
            data=node.get_visualization_data()
        )

        if parent_id:
            self._add_edge(source=parent_id, target=node.id, label="custom", data={})
```

### Real-time Updates

The `LiveVisualizer` supports real-time updates for both chains and conversations:

```python
# Chain updates
await live_viz.on_chain_start(chain)
await live_viz.on_step_start(step)
await live_viz.on_step_complete(step)
await live_viz.on_step_error(step, error)

# Conversation updates
await live_viz.on_conversation_update(conversation)
```

## Error Handling

The visualization components include built-in error handling:

- WebSocket connection errors are logged and handled gracefully
- Graph cycles are detected and reported
- Invalid node/edge data is validated before visualization
- Server cleanup is managed through context managers and signal handlers
