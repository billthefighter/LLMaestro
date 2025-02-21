#!/usr/bin/env python3
"""Script to generate model status badges from test results."""
import json
import shutil
from pathlib import Path
from typing import Dict, Literal, Optional

from llmaestro.llm.models import LLMRegistry, ModelFamily

TestResult = Literal["success", "failure", "skip"]


def generate_badge_json(
    model_name: str, status: TestResult, family: Optional[str] = None, description: Optional[str] = None
) -> dict:
    """Generate badge JSON for shields.io endpoint."""
    status_config = {
        "success": ("operational", "brightgreen"),
        "failure": ("error", "red"),
        "skip": ("untested", "yellow"),
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


def load_test_results() -> Dict[str, TestResult]:
    """Load test results from file, returning empty dict if file doesn't exist."""
    results_file = Path("test-results/model_connectivity.json")
    if not results_file.exists():
        return {}

    try:
        with open(results_file) as f:
            data = json.load(f)
            return {model: info["result"] for model, info in data.items()}
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        print("⚠️  Warning: Invalid or missing test results file")
        return {}


def main():
    """Generate badge JSON files from test results and model registry."""
    # Initialize directories
    results_dir = Path("test-results")
    test_badges_dir = results_dir / "badges"
    docs_badges_dir = Path("docs/badges")

    test_badges_dir.mkdir(parents=True, exist_ok=True)
    docs_badges_dir.mkdir(parents=True, exist_ok=True)

    # Load test results
    test_results = load_test_results()

    # Initialize model registry and load all models
    registry = LLMRegistry()
    models_dir = Path("src/llm/models")

    if not models_dir.exists():
        print("❌ Models directory not found")
        return

    # Load all model files
    for yaml_path in models_dir.glob("*.yaml"):
        try:
            loaded_registry = LLMRegistry.from_yaml(yaml_path)
            for model in loaded_registry._models.values():
                registry.register(model)
        except Exception as e:
            print(f"❌ Error loading {yaml_path}: {str(e)}")

    # Generate badges for all models in registry
    badge_count = 0
    for model_name, model in registry._models.items():
        # Get test result if available, otherwise mark as skipped
        status = test_results.get(model_name, "skip")

        badge_data = generate_badge_json(
            model_name=model_name,
            status=status,
            family=model.family,
            description=model.description if hasattr(model, "description") else None,
        )

        # Save to test-results/badges
        test_badge_file = test_badges_dir / f"{model_name}.json"
        with open(test_badge_file, "w") as f:
            json.dump(badge_data, f, indent=2)

        # Copy to docs/badges
        docs_badge_file = docs_badges_dir / f"{model_name}.json"
        shutil.copy2(test_badge_file, docs_badge_file)
        badge_count += 1

    print(f"✅ Generated badges for {badge_count} models from registry")
    if len(test_results) > 0:
        print(f"  - {len(test_results)} models had test results")
        print(f"  - {badge_count - len(test_results)} models marked as untested")


if __name__ == "__main__":
    main()
