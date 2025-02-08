#!/usr/bin/env python3
"""Script to generate model status badges from test results."""
import json
import shutil
from pathlib import Path
from typing import Literal, Optional

from src.llm.models import ModelFamily

TestResult = Literal["success", "failure", "skip"]


def generate_badge_json(
    model_name: str, status: TestResult, family: Optional[str] = None, description: Optional[str] = None
) -> dict:
    """Generate badge JSON for shields.io endpoint."""
    status_config = {
        "success": ("Connected", "success"),
        "failure": ("Disconnected", "critical"),
        "skip": ("Skipped", "yellow"),
    }

    message, color = status_config[status]

    # Include model family and description in badge tooltip
    title = f"{model_name} ({family})" if family else model_name
    if description:
        title += f"\n{description}"

    return {
        "schemaVersion": 1,
        "label": model_name,
        "message": message,
        "color": color,
        "style": "flat",
        "labelColor": get_family_color(family) if family else "gray",
        "title": title,
    }


def get_family_color(family: Optional[str]) -> str:
    """Get color for model family label."""
    if not family:
        return "gray"

    return {
        ModelFamily.CLAUDE.value: "5436DE",  # Anthropic blue
        ModelFamily.GPT.value: "10A37F",  # OpenAI green
        ModelFamily.GEMINI.value: "4285F4",  # Google blue
    }.get(family, "gray")


def main():
    """Generate badge JSON files from test results."""
    results_dir = Path("test-results")
    test_badges_dir = results_dir / "badges"
    docs_badges_dir = Path("docs/badges")

    # Create directories
    test_badges_dir.mkdir(parents=True, exist_ok=True)
    docs_badges_dir.mkdir(parents=True, exist_ok=True)

    # Read test results with metadata
    try:
        with open(results_dir / "model_connectivity.json") as f:
            test_results = json.load(f)
    except FileNotFoundError:
        print("❌ No test results found. Run validate_model_connectivity.py first.")
        return

    # Generate badge JSON files
    for model_name, data in test_results.items():
        status = data["result"]
        family = data.get("family")
        description = data.get("description")

        badge_data = generate_badge_json(model_name, status, family, description)

        # Save to test-results/badges
        test_badge_file = test_badges_dir / f"{model_name}.json"
        with open(test_badge_file, "w") as f:
            json.dump(badge_data, f, indent=2)

        # Copy to docs/badges
        docs_badge_file = docs_badges_dir / f"{model_name}.json"
        shutil.copy2(test_badge_file, docs_badge_file)

    print(f"✅ Generated badges for {len(test_results)} models")


if __name__ == "__main__":
    main()
