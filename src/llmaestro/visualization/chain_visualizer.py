"""Chain visualization components."""

from typing import Optional

from llmaestro.chains.chains import ChainEdge, ChainNode
from llmaestro.visualization.base_visualizer import BaseVisualizer


class ChainVisualizer(BaseVisualizer[ChainNode, ChainEdge]):
    """Extracts and prepares chain structure for visualization."""

    def _process_node(self, node: ChainNode, parent_id: Optional[str] = None) -> None:
        """Process a single chain node."""
        # Add step node
        self._add_node(
            node_id=node.id,
            label=node.node_type.value,
            type_="step",
            data={
                "has_input_transform": bool(node.step.input_transform),
                "has_output_transform": bool(node.step.output_transform),
                "retry_strategy": node.step.retry_strategy.model_dump(),
                **node.metadata.model_dump(),
            },
        )

        # Connect to parent if exists
        if parent_id:
            self._add_edge(source=parent_id, target=node.id, label="contains", data={})
