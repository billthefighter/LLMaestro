from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from src.llm.chains import (
    AbstractChain,
    ChainStep,
    ChordChain,
    GroupChain,
    MapChain,
    ParallelChain,
    RecursiveChain,
    ReminderChain,
    SequentialChain,
)


@dataclass
class CytoscapeNode:
    """Represents a node in the Cytoscape graph."""

    id: str
    label: str
    type: str
    data: Dict[str, Any]


@dataclass
class CytoscapeEdge:
    """Represents an edge in the Cytoscape graph."""

    id: str
    source: str
    target: str
    label: str
    data: Dict[str, Any]


@dataclass
class CytoscapeGraph:
    """Complete graph representation for Cytoscape."""

    nodes: List[CytoscapeNode]
    edges: List[CytoscapeEdge]

    def to_cytoscape_format(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert to Cytoscape.js compatible format."""
        return {
            "nodes": [
                {"data": {"id": node.id, "label": node.label, "type": node.type, **node.data}} for node in self.nodes
            ],
            "edges": [
                {
                    "data": {
                        "id": edge.id,
                        "source": edge.source,
                        "target": edge.target,
                        "label": edge.label,
                        **edge.data,
                    }
                }
                for edge in self.edges
            ],
        }


class ChainVisualizer:
    """Extracts and prepares chain structure for visualization."""

    def __init__(self):
        self.nodes: List[CytoscapeNode] = []
        self.edges: List[CytoscapeEdge] = []
        self.processed_steps: Set[str] = set()
        self.edge_counter = 0

    def _add_node(self, node_id: str, label: str, type_: str, data: Dict[str, Any]) -> None:
        """Add a node if it doesn't already exist."""
        if not any(n.id == node_id for n in self.nodes):
            self.nodes.append(CytoscapeNode(id=node_id, label=label, type=type_, data=data))

    def _add_edge(self, source: str, target: str, label: str, data: Dict[str, Any]) -> None:
        """Add an edge between nodes."""
        self.edge_counter += 1
        self.edges.append(
            CytoscapeEdge(id=f"e{self.edge_counter}", source=source, target=target, label=label, data=data)
        )

    def _process_chain_step(self, step: ChainStep, parent_id: Optional[str] = None) -> None:
        """Process a single chain step."""
        if step.id in self.processed_steps:
            return

        self.processed_steps.add(step.id)

        # Add step node
        self._add_node(
            node_id=step.id,
            label=step.task_type,
            type_="step",
            data={
                "has_input_transform": bool(step.input_transform),
                "has_output_transform": bool(step.output_transform),
                "retry_strategy": step.retry_strategy,
            },
        )

        # Connect to parent if exists
        if parent_id:
            self._add_edge(source=parent_id, target=step.id, label="contains", data={})

    def _process_sequential_chain(self, chain: SequentialChain, parent_id: Optional[str] = None) -> None:
        """Process a sequential chain."""
        chain_id = str(chain.context.metadata.get("chain_id", ""))
        self._add_node(
            node_id=chain_id,
            label="Sequential Chain",
            type_="sequential_chain",
            data={"artifacts": list(chain.context.artifacts.keys())},
        )

        if parent_id:
            self._add_edge(source=parent_id, target=chain_id, label="contains", data={})

        # Process steps and their connections
        prev_step = None
        for step in chain.steps:
            self._process_chain_step(step, chain_id)
            if prev_step:
                self._add_edge(source=prev_step.id, target=step.id, label="next", data={"chain_id": chain_id})
            prev_step = step

    def _process_parallel_chain(self, chain: ParallelChain, parent_id: Optional[str] = None) -> None:
        """Process a parallel chain."""
        chain_id = str(chain.context.metadata.get("chain_id", ""))
        self._add_node(
            node_id=chain_id,
            label="Parallel Chain",
            type_="parallel_chain",
            data={
                "artifacts": list(chain.context.artifacts.keys()),
                "max_concurrent": chain.max_concurrent,
            },
        )

        if parent_id:
            self._add_edge(source=parent_id, target=chain_id, label="contains", data={})

        # Process parallel steps
        for step in chain.steps:
            self._process_chain_step(step, chain_id)

    def _process_chord_chain(self, chain: ChordChain, parent_id: Optional[str] = None) -> None:
        """Process a chord chain."""
        chain_id = str(chain.context.metadata.get("chain_id", ""))
        self._add_node(
            node_id=chain_id,
            label="Chord Chain",
            type_="chord_chain",
            data={"artifacts": list(chain.context.artifacts.keys())},
        )

        if parent_id:
            self._add_edge(source=parent_id, target=chain_id, label="contains", data={})

        # Process parallel chain
        self._process_parallel_chain(chain.parallel_chain, chain_id)

        # Process callback step
        self._process_chain_step(chain.callback_step, chain_id)

        # Add edges from parallel chain to callback
        for step in chain.parallel_chain.steps:
            self._add_edge(
                source=step.id,
                target=chain.callback_step.id,
                label="callback",
                data={"chain_id": chain_id},
            )

    def _process_group_chain(self, chain: GroupChain, parent_id: Optional[str] = None) -> None:
        """Process a group chain."""
        chain_id = str(chain.context.metadata.get("chain_id", ""))
        self._add_node(
            node_id=chain_id,
            label="Group Chain",
            type_="group_chain",
            data={"artifacts": list(chain.context.artifacts.keys())},
        )

        if parent_id:
            self._add_edge(source=parent_id, target=chain_id, label="contains", data={})

        # Process each chain in the group
        for sub_chain in chain.chains:
            self.process_chain(sub_chain, chain_id)

    def _process_map_chain(self, chain: MapChain, parent_id: Optional[str] = None) -> None:
        """Process a map chain."""
        chain_id = str(chain.context.metadata.get("chain_id", ""))
        self._add_node(
            node_id=chain_id,
            label="Map Chain",
            type_="map_chain",
            data={"artifacts": list(chain.context.artifacts.keys())},
        )

        if parent_id:
            self._add_edge(source=parent_id, target=chain_id, label="contains", data={})

        # Process template chain
        self.process_chain(chain.chain_template, chain_id)

    def _process_reminder_chain(self, chain: ReminderChain, parent_id: Optional[str] = None) -> None:
        """Process a reminder chain."""
        chain_id = str(chain.context.metadata.get("chain_id", ""))
        self._add_node(
            node_id=chain_id,
            label="Reminder Chain",
            type_="reminder_chain",
            data={
                "artifacts": list(chain.context.artifacts.keys()),
                "prune_completed": chain.prune_completed,
                "max_context_items": chain.max_context_items,
            },
        )

        if parent_id:
            self._add_edge(source=parent_id, target=chain_id, label="contains", data={})

        # Process steps with reminder connections
        prev_step = None
        for step in chain.steps:
            self._process_chain_step(step, chain_id)
            if prev_step:
                self._add_edge(
                    source=prev_step.id,
                    target=step.id,
                    label="next_with_context",
                    data={"chain_id": chain_id},
                )
            prev_step = step

    def _process_recursive_chain(self, chain: RecursiveChain, parent_id: Optional[str] = None) -> None:
        """Process a recursive chain."""
        chain_id = str(chain.context.metadata.get("chain_id", ""))
        self._add_node(
            node_id=chain_id,
            label="Recursive Chain",
            type_="recursive_chain",
            data={
                "artifacts": list(chain.context.artifacts.keys()),
                "max_recursion_depth": chain.context.max_recursion_depth,
            },
        )

        if parent_id:
            self._add_edge(source=parent_id, target=chain_id, label="contains", data={})

        # Process initial step
        self._process_chain_step(chain.initial_step, chain_id)

        # Add self-referential edge to represent recursion
        self._add_edge(
            source=chain.initial_step.id,
            target=chain_id,
            label="may_recurse",
            data={"max_depth": chain.context.max_recursion_depth},
        )

    def process_chain(self, chain: AbstractChain, parent_id: Optional[str] = None) -> None:
        """Process any type of chain and extract its structure."""
        if isinstance(chain, SequentialChain):
            self._process_sequential_chain(chain, parent_id)
        elif isinstance(chain, ParallelChain):
            self._process_parallel_chain(chain, parent_id)
        elif isinstance(chain, ChordChain):
            self._process_chord_chain(chain, parent_id)
        elif isinstance(chain, GroupChain):
            self._process_group_chain(chain, parent_id)
        elif isinstance(chain, MapChain):
            self._process_map_chain(chain, parent_id)
        elif isinstance(chain, ReminderChain):
            self._process_reminder_chain(chain, parent_id)
        elif isinstance(chain, RecursiveChain):
            self._process_recursive_chain(chain, parent_id)

    def get_visualization_data(self) -> CytoscapeGraph:
        """Get the complete graph data in Cytoscape format."""
        return CytoscapeGraph(nodes=self.nodes, edges=self.edges)
