import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CytoscapeStyle:
    """Represents a Cytoscape style configuration."""

    selector: str
    style: Dict[str, Any]


class CytoscapeRenderer:
    """Handles rendering configuration for chain visualizations."""

    def __init__(self):
        self.default_layout = {
            "name": "dagre",  # Directed graph layout
            "rankDir": "TB",  # Top to bottom direction
            "nodeSep": 50,  # Minimum distance between nodes
            "rankSep": 100,  # Minimum distance between layers
            "padding": 30,  # Padding around the graph
            "animate": True,  # Enable layout animation
            "animationDuration": 500,  # Animation duration in ms
        }

        # Define base styles for different node types
        self.base_styles = [
            # Default node style
            CytoscapeStyle(
                "node",
                {
                    "label": "data(label)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "text-wrap": "wrap",
                    "text-max-width": "100px",
                    "font-size": "12px",
                    "width": "120px",
                    "height": "40px",
                    "border-width": "1px",
                    "padding": "10px",
                    "shape": "roundrectangle",
                },
            ),
            # Chain nodes
            CytoscapeStyle(
                "node[type = 'sequential_chain']",
                {
                    "background-color": "#E8F5E9",
                    "border-color": "#66BB6A",
                    "border-width": "2px",
                },
            ),
            CytoscapeStyle(
                "node[type = 'parallel_chain']",
                {
                    "background-color": "#E3F2FD",
                    "border-color": "#42A5F5",
                    "border-width": "2px",
                },
            ),
            CytoscapeStyle(
                "node[type = 'chord_chain']",
                {
                    "background-color": "#F3E5F5",
                    "border-color": "#AB47BC",
                    "border-width": "2px",
                },
            ),
            CytoscapeStyle(
                "node[type = 'group_chain']",
                {
                    "background-color": "#FFF3E0",
                    "border-color": "#FB8C00",
                    "border-width": "2px",
                },
            ),
            CytoscapeStyle(
                "node[type = 'map_chain']",
                {
                    "background-color": "#E0F7FA",
                    "border-color": "#26C6DA",
                    "border-width": "2px",
                },
            ),
            CytoscapeStyle(
                "node[type = 'reminder_chain']",
                {
                    "background-color": "#F1F8E9",
                    "border-color": "#7CB342",
                    "border-width": "2px",
                },
            ),
            CytoscapeStyle(
                "node[type = 'recursive_chain']",
                {
                    "background-color": "#FBE9E7",
                    "border-color": "#FF7043",
                    "border-width": "2px",
                },
            ),
            # Step nodes
            CytoscapeStyle(
                "node[type = 'step']",
                {
                    "background-color": "#FFFFFF",
                    "border-color": "#9E9E9E",
                    "border-width": "1px",
                    "shape": "round-rectangle",
                    "width": "100px",
                    "height": "35px",
                },
            ),
            # Edge styles
            CytoscapeStyle(
                "edge",
                {
                    "width": "2px",
                    "curve-style": "bezier",
                    "target-arrow-shape": "triangle",
                    "line-color": "#9E9E9E",
                    "target-arrow-color": "#9E9E9E",
                    "label": "data(label)",
                    "font-size": "10px",
                    "text-rotation": "autorotate",
                    "text-margin-y": "-10px",
                },
            ),
            CytoscapeStyle(
                "edge[label = 'contains']",
                {
                    "line-style": "dashed",
                    "line-color": "#BDBDBD",
                    "target-arrow-color": "#BDBDBD",
                },
            ),
            CytoscapeStyle(
                "edge[label = 'next']",
                {
                    "line-color": "#2196F3",
                    "target-arrow-color": "#2196F3",
                },
            ),
            CytoscapeStyle(
                "edge[label = 'callback']",
                {
                    "line-color": "#9C27B0",
                    "target-arrow-color": "#9C27B0",
                },
            ),
            CytoscapeStyle(
                "edge[label = 'next_with_context']",
                {
                    "line-color": "#4CAF50",
                    "target-arrow-color": "#4CAF50",
                },
            ),
            CytoscapeStyle(
                "edge[label = 'may_recurse']",
                {
                    "line-color": "#FF5722",
                    "target-arrow-color": "#FF5722",
                    "line-style": "dotted",
                },
            ),
        ]

    def get_config(
        self,
        elements: Dict[str, List[Dict[str, Any]]],
        layout: Optional[Dict[str, Any]] = None,
        additional_styles: Optional[List[CytoscapeStyle]] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete Cytoscape.js configuration.

        Args:
            elements: Graph elements (nodes and edges) in Cytoscape.js format
            layout: Optional custom layout configuration
            additional_styles: Optional additional style rules

        Returns:
            Complete Cytoscape.js configuration object
        """
        # Combine base styles with any additional styles
        all_styles = self.base_styles
        if additional_styles:
            all_styles.extend(additional_styles)

        # Convert styles to Cytoscape.js format
        style_config = [{"selector": style.selector, "style": style.style} for style in all_styles]

        return {
            "elements": elements,
            "style": style_config,
            "layout": layout or self.default_layout,
            "wheelSensitivity": 0.2,  # Slower zoom for better control
            "minZoom": 0.2,
            "maxZoom": 3.0,
            "userZoomingEnabled": True,
            "userPanningEnabled": True,
            "boxSelectionEnabled": True,
        }

    def get_html_template(self, config: Dict[str, Any], container_id: str = "cy") -> str:
        """
        Generate HTML template with embedded Cytoscape.js visualization.

        Args:
            config: Cytoscape.js configuration
            container_id: ID for the container element

        Returns:
            Complete HTML document as a string
        """
        # Convert config to JSON, ensuring proper string formatting
        config_json = json.dumps(config)

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Chain Visualization</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        #cy {{
            width: 100%;
            height: 800px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .controls {{
            margin-bottom: 20px;
        }}
        button {{
            padding: 8px 16px;
            margin-right: 8px;
            border: none;
            border-radius: 4px;
            background-color: #2196F3;
            color: white;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{
            background-color: #1976D2;
        }}
    </style>
</head>
<body>
    <div class="controls">
        <button onclick="cy.fit()">Fit to View</button>
        <button onclick="cy.center()">Center</button>
        <button onclick="cy.layout(layout).run()">Reset Layout</button>
    </div>
    <div id="{container_id}"></div>
    <script>
        // Initialize Cytoscape
        const config = {config_json};
        const layout = config.layout;

        // Create Cytoscape instance
        const cy = cytoscape({{
            ...config,
            container: document.getElementById('{container_id}')
        }});

        // Add interactive features
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            console.log('Node tapped:', node.data());
        }});

        cy.on('mouseover', 'node', function(evt) {{
            const node = evt.target;
            node.style({{
                'border-width': '3px',
                'border-color': '#000000'
            }});
        }});

        cy.on('mouseout', 'node', function(evt) {{
            const node = evt.target;
            node.removeStyle();
        }});

        // Initial layout
        cy.layout(layout).run();
    </script>
</body>
</html>
"""
