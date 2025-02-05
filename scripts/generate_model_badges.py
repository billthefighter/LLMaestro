#!/usr/bin/env python3
"""Script to generate model status badges from test results."""
import json
import shutil
from pathlib import Path
from typing import Literal

TestResult = Literal["success", "failure", "skip"]


def generate_badge_json(model_name: str, status: TestResult) -> dict:
    """Generate badge JSON for shields.io endpoint."""
    status_config = {
        "success": ("Connected", "success"),
        "failure": ("Disconnected", "critical"),
        "skip": ("Skipped", "yellow"),
    }

    message, color = status_config[status]
    return {
        "schemaVersion": 1,
        "label": model_name,
        "message": message,
        "color": color,
        "style": "flat",
    }


def main():
    """Generate badge JSON files from test results."""
    results_dir = Path("test-results")
    test_badges_dir = results_dir / "badges"
    docs_badges_dir = Path("docs/badges")

    # Create directories
    test_badges_dir.mkdir(parents=True, exist_ok=True)
    docs_badges_dir.mkdir(parents=True, exist_ok=True)

    # Read test results
    with open(results_dir / "model_connectivity.json") as f:
        test_results = json.load(f)

    # Generate badge JSON files
    for model_name, status in test_results.items():
        badge_data = generate_badge_json(model_name, status)

        # Save to test-results/badges
        test_badge_file = test_badges_dir / f"{model_name}.json"
        with open(test_badge_file, "w") as f:
            json.dump(badge_data, f, indent=2)

        # Copy to docs/badges
        docs_badge_file = docs_badges_dir / f"{model_name}.json"
        shutil.copy2(test_badge_file, docs_badge_file)


if __name__ == "__main__":
    main()
