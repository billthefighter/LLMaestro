"""Script to update README.md with current model status badges."""
import json
import re
from pathlib import Path
from typing import Dict, List, Literal, Tuple

TestResult = Literal["success", "failure", "skip"]


def load_test_results() -> Dict[str, TestResult]:
    """Load the test results from the JSON file."""
    results_path = Path("test-results/model_connectivity.json")
    if not results_path.exists():
        return {}

    with open(results_path) as f:
        return json.load(f)


def group_models_by_provider(results: Dict[str, TestResult]) -> Dict[str, List[Tuple[str, TestResult]]]:
    """Group models by their provider (Anthropic, OpenAI, etc)."""
    providers = {
        "anthropic": [],
        "openai": [],
    }

    for model, status in results.items():
        if "claude" in model.lower():
            providers["anthropic"].append((model, status))
        elif "gpt" in model.lower():
            providers["openai"].append((model, status))

    # Sort models within each provider
    for provider in providers.values():
        provider.sort(key=lambda x: x[0])

    return providers


def generate_badge_section(models: List[Tuple[str, TestResult]]) -> str:
    """Generate markdown for a group of model badges."""
    lines = []
    for model, _ in models:  # Use _ to indicate intentionally unused variable
        # Use the correct GitHub repository URL
        badge_url = f"https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/billthefighter/llm_orchestrator/main/docs/badges/{model}.json"
        lines.append(f"![{model}]({badge_url})")
    return "\n".join(lines)


def update_readme(results: Dict[str, TestResult]) -> None:
    """Update the README.md with current model status badges."""
    readme_path = Path("README.md")
    if not readme_path.exists():
        return

    with open(readme_path, "r") as f:
        content = f.read()

    # Create new Model Status section content
    model_status_section = "## Model Status\n\n"
    providers = group_models_by_provider(results)

    for provider, models in providers.items():
        if not models:
            continue

        provider_title = "OpenAI" if provider.lower() == "openai" else provider.title()
        model_status_section += f"### {provider_title} Models\n{generate_badge_section(models)}\n\n"

    # Try to find and replace existing Model Status section including provider subsections
    pattern = (
        r"## Model Status\n\n"  # Main section header
        r"(?:### (?:Anthropic|OpenAI|Openai) Models\n"  # Provider section headers
        r"(?:!\[.*?\]\(.*?\)\n)*\n?)*"  # Badge lines
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


if __name__ == "__main__":
    main()
