import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import pandas as pd
import py4cytoscape as p4c

from ..llm.chains import AbstractChain
from .chain_visualizer import ChainVisualizer
from .cytoscape_config import CytoscapeConfig
from .cytoscape_renderer import CytoscapeRenderer


class ChainVisualizationManager:
    """Utility class for visualizing chains."""

    def __init__(self):
        self.visualizer = ChainVisualizer()
        self.renderer = CytoscapeRenderer()
        self.cytoscape = CytoscapeConfig()

    def visualize_chain(
        self,
        chain: AbstractChain,
        output_path: Optional[str] = None,
        layout: Optional[Dict[str, Any]] = None,
        method: Literal["html", "cytoscape"] = "html",
    ) -> str:
        """
        Generate a visualization of a chain.

        Args:
            chain: The chain to visualize
            output_path: Optional path to save the visualization
            layout: Optional custom layout configuration
            method: Visualization method ("html" for static HTML, "cytoscape" for Cytoscape desktop)

        Returns:
            str: Path to the generated visualization file
        """
        # Extract chain structure
        self.visualizer.process_chain(chain)
        graph_data = self.visualizer.get_visualization_data()

        if method == "html":
            # Generate HTML visualization
            config = self.renderer.get_config(graph_data.to_cytoscape_format(), layout=layout)
            html = self.renderer.get_html_template(config)

            # Save to file if path provided, otherwise use default
            if output_path is None:
                # Create visualization directory if it doesn't exist
                viz_dir = Path("chain_visualizations")
                viz_dir.mkdir(exist_ok=True)

                # Use chain ID for filename
                chain_id = chain.context.metadata.get("chain_id", "chain")
                output_path = str(viz_dir / f"{chain_id}.html")

            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Write HTML file
            with open(output_path, "w") as f:
                f.write(html)

            return output_path

        elif method == "cytoscape":
            # Ensure Cytoscape is running
            self.cytoscape.ensure_cytoscape_running()

            # Convert graph data to pandas DataFrames for py4cytoscape
            nodes_df = pd.DataFrame(
                [{"id": node.id, "label": node.label, "type": node.type, **node.data} for node in graph_data.nodes]
            )

            edges_df = pd.DataFrame(
                [
                    {"source": edge.source, "target": edge.target, "label": edge.label, **edge.data}
                    for edge in graph_data.edges
                ]
            )

            # Clear any existing networks
            self.cytoscape.clear_session()

            # Create network from DataFrames
            network_name = f"Chain {chain.context.metadata.get('chain_id', '')}"
            network_suid = p4c.create_network_from_data_frames(nodes=nodes_df, edges=edges_df, title=network_name)

            # Apply visual style
            self.cytoscape.create_visual_style()
            p4c.set_visual_style(self.cytoscape.default_style_name)

            # Apply layout
            self.cytoscape.apply_layout()

            # Export image if output path provided
            if output_path:
                return self.cytoscape.export_image(output_path)

            return f"Network created in Cytoscape with SUID: {network_suid}"

        else:
            raise ValueError(f"Unsupported visualization method: {method}")

    def visualize_chain_to_string(self, chain: AbstractChain, layout: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate HTML visualization as a string.

        Args:
            chain: The chain to visualize
            layout: Optional custom layout configuration

        Returns:
            HTML visualization as a string
        """
        # Extract chain structure
        self.visualizer.process_chain(chain)
        graph_data = self.visualizer.get_visualization_data().to_cytoscape_format()

        # Generate visualization config
        config = self.renderer.get_config(graph_data, layout=layout)

        # Return HTML string
        return self.renderer.get_html_template(config)
