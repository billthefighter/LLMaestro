"""Script to update README.md with current model status badges."""
import json
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple


class ModelFamily(str, Enum):
    """Model family enumeration."""

    CLAUDE = "claude"
    GPT = "gpt"
    GEMINI = "gemini"


TestResult = Literal["success", "failure", "skip"]


def load_test_results() -> Dict[str, Dict[str, str]]:
    """Load the test results from the JSON file."""
    results_path = Path("test-results/model_connectivity.json")
    if not results_path.exists():
        return {}

    with open(results_path) as f:
        return json.load(f)


def group_models_by_provider(results: Dict[str, Dict[str, str]]) -> Dict[str, List[Tuple[str, str, Optional[str]]]]:
    """Group models by their provider (Anthropic, OpenAI, etc).

    Returns:
        Dict mapping provider name to list of (model_name, status, description) tuples
    """
    providers = {}

    for model_name, data in results.items():
        family = data.get("family", "unknown")
        status = data.get("result", "skip")
        description = data.get("description")

        if family not in providers:
            providers[family] = []

        providers[family].append((model_name, status, description))

    # Sort models within each provider
    for provider in providers.values():
        provider.sort(key=lambda x: x[0])

    return providers


def get_provider_display_name(family: str) -> str:
    """Get display name for a model family."""
    try:
        model_family = ModelFamily(family)
        return {
            ModelFamily.CLAUDE: "Anthropic",
            ModelFamily.GPT: "OpenAI",
            ModelFamily.GEMINI: "Google",
        }.get(model_family, family.title())
    except ValueError:
        return family.title()


def generate_badge_section(models: List[Tuple[str, str, Optional[str]]]) -> str:
    """Generate markdown for a group of model badges."""
    lines = []
    for model_name, _, description in models:
        # Use the correct GitHub repository URL
        badge_url = f"https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/{model_name}.json"
        badge_line = f"![{model_name}]({badge_url})"
        if description:
            badge_line += f" - {description}"
        lines.append(badge_line)
    return "\n".join(lines)


def update_readme(results: Dict[str, Dict[str, str]]) -> None:
    """Update the README.md with current model status badges."""
    readme_path = Path("README.md")
    if not readme_path.exists():
        return

    with open(readme_path, "r") as f:
        content = f.read()

    # Create new Model Status section content
    model_status_section = "## Model Status\n\n"
    providers = group_models_by_provider(results)

    for family, models in providers.items():
        if not models:
            continue

        provider_name = get_provider_display_name(family)
        model_status_section += f"### {provider_name} Models\n{generate_badge_section(models)}\n\n"

    # Try to find and replace existing Model Status section including provider subsections
    pattern = (
        r"## Model Status\n\n"  # Main section header
        r"(?:### [^\n]+\n"  # Provider section headers
        r"(?:!\[.*?\]\(.*?\)(?:\s*-\s*[^\n]+)?\n)*\n?)*"  # Badge lines with optional descriptions
    )

    if re.search(pattern, content, re.DOTALL | re.IGNORECASE):
        # Replace existing section
        updated_content = re.sub(pattern, model_status_section, content, flags=re.DOTALL | re.IGNORECASE)
    else:
        # Add new section after first heading
        first_heading_end = content.find("#")
        next_heading = content.find("##", first_heading_end + 1)
        if next_heading != -1:
            # Insert before next heading
            updated_content = content[:next_heading].rstrip() + "\n\n" + model_status_section + content[next_heading:]
        else:
            # Append to end
            updated_content = content.rstrip() + "\n\n" + model_status_section

    # Write updated content
    with open(readme_path, "w") as f:
        f.write(updated_content)


def main() -> None:
    """Main entry point."""
    # Ensure docs/badges directory exists
    badges_dir = Path("docs/badges")
    badges_dir.mkdir(parents=True, exist_ok=True)

    # Load and process results
    results = load_test_results()
    if results:
        update_readme(results)
        print("✅ Updated README.md with model status badges")
    else:
        print("❌ No test results found. Run validate_model_connectivity.py first")


if __name__ == "__main__":
    main()
