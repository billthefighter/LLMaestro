#!/usr/bin/env python3
import asyncio
from pathlib import Path

from src.visualization.examples import create_example_chains
from src.visualization.visualize import ChainVisualizationManager


async def main():
    # Create output directory
    output_dir = Path("examples/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create example chains
    chains = create_example_chains()

    # Create visualization manager
    viz_manager = ChainVisualizationManager()

    # Generate visualizations for each chain type
    for chain_type, chain in chains.items():
        print(f"Generating visualization for {chain_type} chain...")

        # Generate HTML visualization
        html_path = viz_manager.visualize_chain(
            chain, output_path=str(output_dir / f"{chain_type}_chain.html"), method="html"
        )
        print(f"  HTML visualization saved to: {html_path}")

        try:
            # Try to generate Cytoscape visualization if available
            cytoscape_path = viz_manager.visualize_chain(
                chain, output_path=str(output_dir / f"{chain_type}_chain.png"), method="cytoscape"
            )
            print(f"  Cytoscape visualization saved to: {cytoscape_path}")
        except RuntimeError as e:
            print(f"  Cytoscape visualization skipped: {e}")

        print()


if __name__ == "__main__":
    asyncio.run(main())
