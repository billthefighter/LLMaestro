#!/usr/bin/env python3
"""Script to generate model status badges from test results."""
import json
from pathlib import Path


def generate_badge_json(model_name: str, is_connected: bool) -> dict:
    """Generate badge JSON for shields.io endpoint."""
    return {
        "schemaVersion": 1,
        "label": model_name,
        "message": "Connected" if is_connected else "Disconnected",
        "color": "success" if is_connected else "critical",
        "style": "flat",
    }


def main():
    """Generate badge JSON files from test results."""
    results_dir = Path("test-results")
    badges_dir = results_dir / "badges"
    badges_dir.mkdir(parents=True, exist_ok=True)

    # Read test results
    with open(results_dir / "model_connectivity.json") as f:
        test_results = json.load(f)

    # Generate badge JSON files
    for model_name, is_connected in test_results.items():
        badge_data = generate_badge_json(model_name, is_connected)
        badge_file = badges_dir / f"{model_name}.json"

        with open(badge_file, "w") as f:
            json.dump(badge_data, f, indent=2)


if __name__ == "__main__":
    main()
