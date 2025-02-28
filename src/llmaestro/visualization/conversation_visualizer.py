"""Conversation visualization components."""

from typing import Any, Dict, Optional

from llmaestro.core.conversations import ConversationEdge, ConversationNode
from llmaestro.visualization.base_visualizer import BaseVisualizer
from llmaestro.prompts.base import BasePrompt
from llmaestro.core.models import LLMResponse


class ConversationVisualizer(BaseVisualizer[ConversationNode, ConversationEdge]):
    """Extracts and prepares conversation structure for visualization."""

    def _get_content_snippet(self, content: Any) -> str:
        """Get a displayable snippet of content."""
        try:
            if isinstance(content, BasePrompt):
                # For prompts, show system role and first line of user prompt
                system_hint = content.system_prompt.split("\n")[0][:30]
                user_text = content.user_prompt.split("\n")[0][:50]
                return f"{system_hint}...\n{user_text}..."
            elif isinstance(content, LLMResponse):
                # For responses, show first meaningful line of content
                lines = content.content.split("\n")
                content_line = next((line for line in lines if line.strip()), lines[0] if lines else "")
                return content_line[:80] + ("..." if len(content_line) > 80 else "")
            return str(content)[:80] + ("..." if len(str(content)) > 80 else "")
        except Exception:
            return "Error extracting content"

    def _get_node_style(self, node: ConversationNode) -> Dict[str, Any]:
        """Get style information for a node based on its type and state."""
        base_style = {
            "shape": "roundrectangle",
            "width": 200,
            "height": 75,
            "font-size": 14,
            "text-wrap": "wrap",
            "text-max-width": 180,
            "border-width": 2,
            "text-valign": "center",
            "text-halign": "center",
        }

        if node.node_type == "prompt":
            base_style.update(
                {
                    "background-color": "#E3F2FD",  # Light blue for prompts
                    "border-color": "#1976D2",
                    "color": "#1976D2",
                }
            )
        elif node.node_type == "response":
            if isinstance(node.content, LLMResponse):
                if node.content.success:
                    base_style.update(
                        {
                            "background-color": "#E8F5E9",  # Light green for successful responses
                            "border-color": "#43A047",
                            "color": "#2E7D32",
                        }
                    )
                else:
                    base_style.update(
                        {
                            "background-color": "#FFEBEE",  # Light red for failed responses
                            "border-color": "#D32F2F",
                            "color": "#C62828",
                        }
                    )
            else:
                base_style.update(
                    {
                        "background-color": "#FFF3E0",  # Light orange for processing
                        "border-color": "#F57C00",
                        "color": "#E65100",
                    }
                )

        return base_style

    def _process_node(self, node: ConversationNode, parent_id: Optional[str] = None) -> None:
        """Process a single conversation node."""
        # Create node label with content snippet
        content_snippet = self._get_content_snippet(node.content)
        label = f"{node.node_type.title()}\n{content_snippet}"

        # Add node with style information
        self._add_node(
            node_id=node.id,
            label=label,
            type_=node.node_type,
            data={
                "token_usage": node.token_usage.model_dump() if node.token_usage else None,
                "style": self._get_node_style(node),
                "created_at": node.created_at.strftime("%H:%M:%S"),
                "content_type": type(node.content).__name__,
                **node.metadata,
            },
        )

        # Connect to parent if exists
        if parent_id:
            self._add_edge(
                source=parent_id,
                target=node.id,
                label="next",
                data={
                    "style": {
                        "width": 2,
                        "line-color": "#757575",
                        "target-arrow-color": "#757575",
                        "curve-style": "bezier",
                    }
                },
            )
