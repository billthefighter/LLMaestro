import os

import py4cytoscape as p4c


class CytoscapeConfig:
    """Configuration and utilities for Cytoscape integration."""

    def __init__(self, port: int = 1234):
        self.port = port
        self.default_style_name = "LLM Chain Style"

    def ensure_cytoscape_running(self) -> bool:
        """
        Check if Cytoscape is running and accessible.

        Returns:
            bool: True if Cytoscape is running and accessible
        """
        try:
            return p4c.cytoscape_ping()
        except Exception as e:
            raise RuntimeError(
                "Cytoscape is not running. Please start Cytoscape desktop application "
                "before attempting to visualize chains."
            ) from e

    def create_visual_style(self) -> None:
        """Create or update the default visual style for chain visualization."""
        # Define node colors for different chain types
        chain_colors = {
            "sequential_chain": "#E8F5E9",
            "parallel_chain": "#E3F2FD",
            "chord_chain": "#F3E5F5",
            "group_chain": "#FFF3E0",
            "map_chain": "#E0F7FA",
            "reminder_chain": "#F1F8E9",
            "recursive_chain": "#FBE9E7",
            "step": "#FFFFFF",
        }

        # Create style
        style = {
            "title": self.default_style_name,
            "defaults": [
                {"visualProperty": "NODE_SHAPE", "value": "roundrectangle"},
                {"visualProperty": "NODE_WIDTH", "value": 120},
                {"visualProperty": "NODE_HEIGHT", "value": 40},
                {"visualProperty": "NODE_BORDER_WIDTH", "value": 2},
                {"visualProperty": "NODE_LABEL_FONT_SIZE", "value": 12},
                {"visualProperty": "EDGE_WIDTH", "value": 2},
                {"visualProperty": "EDGE_TARGET_ARROW_SHAPE", "value": "ARROW"},
                {"visualProperty": "EDGE_LABEL_FONT_SIZE", "value": 10},
            ],
            "mappings": [
                {
                    "visualProperty": "NODE_BACKGROUND_COLOR",
                    "mappingType": "discrete",
                    "mappingColumn": "type",
                    "mappingColumnType": "String",
                    "discreteMappings": [{"value": color, "key": key} for key, color in chain_colors.items()],
                },
                {
                    "visualProperty": "EDGE_LINE_STYLE",
                    "mappingType": "discrete",
                    "mappingColumn": "label",
                    "mappingColumnType": "String",
                    "discreteMappings": [
                        {"value": "SOLID", "key": "next"},
                        {"value": "EQUAL_DASH", "key": "contains"},
                        {"value": "SOLID", "key": "callback"},
                        {"value": "SOLID", "key": "next_with_context"},
                        {"value": "DOT", "key": "may_recurse"},
                    ],
                },
            ],
        }

        # Delete existing style if it exists
        try:
            p4c.delete_visual_style(self.default_style_name)
        except Exception:
            pass

        # Create new style
        p4c.create_visual_style(style, self.default_style_name)

    def apply_layout(self, layout_name: str = "hierarchical") -> None:
        """
        Apply a layout to the current network.

        Args:
            layout_name: Name of the layout to apply
        """
        layout_params = {"hierarchical": {"name": "hierarchical", "rankDir": "TB", "nodeSep": 50, "rankSep": 100}}

        params = layout_params.get(layout_name, {})
        p4c.layout_network(layout_name, **params)

    def export_image(self, output_path: str, format: str = "PNG", overwrite: bool = True) -> str:
        """
        Export the current network visualization as an image.

        Args:
            output_path: Path to save the image
            format: Image format (PNG, PDF, SVG)
            overwrite: Whether to overwrite existing file

        Returns:
            str: Path to the exported image
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Export image
        return p4c.export_image(output_path, format=format, overwrite=overwrite)

    def clear_session(self) -> None:
        """Clear the current Cytoscape session."""
        try:
            p4c.delete_all_networks()
        except Exception:
            pass  # Ignore errors if no networks exist
