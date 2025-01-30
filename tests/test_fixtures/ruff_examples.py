import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import requests
from bs4 import BeautifulSoup


class RuffExampleBuilder:
    """Utility class for building Ruff rule examples."""

    def __init__(self):
        self.base_url = "https://docs.astral.sh/ruff/rules"
        self.cache_dir = Path.home() / ".llm_orchestrator" / "ruff_examples"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_rule_page(self, rule_code: str) -> str:
        """Fetch the documentation page for a specific rule."""
        # Convert rule code to URL path (e.g., E501 -> /rules/e501)
        rule_path = rule_code.lower()
        url = f"{self.base_url}/{rule_path}"

        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def _extract_examples(self, html_content: str) -> Dict[str, str]:
        """Extract bad and good code examples from the rule documentation."""
        soup = BeautifulSoup(html_content, "html.parser")
        examples = {}

        # Find code blocks marked as "Bad" and "Good"
        for block in soup.find_all("div", class_="highlight"):
            prev_text = block.find_previous_sibling("p")
            if prev_text and prev_text.text.strip().lower() in ["bad:", "good:"]:
                code = block.get_text(strip=True)
                key = "bad" if prev_text.text.strip().lower() == "bad:" else "good"
                examples[key] = code

        return examples

    def _write_example_file(self, tmp_path: Path, rule_code: str, code: str) -> Path:
        """Write example code to a temporary file."""
        file_path = tmp_path / f"{rule_code.lower()}_example.py"
        file_path.write_text(code)
        return file_path

    def _run_ruff(self, file_path: Path, rule_code: str) -> Tuple[bool, List[Dict]]:
        """Run Ruff on a file and check for specific rule violations.

        Returns:
            Tuple[bool, List[Dict]]: (has_error, violations)
            - has_error: True if the specific rule was violated
            - violations: List of violation dictionaries with details
        """
        try:
            # Run ruff with JSON output format
            result = subprocess.run(
                ["ruff", "check", "--select", rule_code, "--format", "json", str(file_path)],
                capture_output=True,
                text=True,
            )

            # Parse JSON output
            violations = json.loads(result.stdout) if result.stdout else []

            # Check if any violations match our rule code
            rule_violations = [v for v in violations if v.get("code") == rule_code]
            return bool(rule_violations), violations

        except (subprocess.SubprocessError, json.JSONDecodeError) as e:
            print(f"Error running Ruff: {e}")
            return False, []

    def validate_examples(self, examples: Dict[str, Path], rule_code: str) -> Dict[str, bool]:
        """Validate examples against Ruff.

        Returns:
            Dict with keys 'bad' and 'good', values are True if the example
            behaves as expected (bad has error, good has no error).
        """
        results = {}

        # Bad example should trigger the rule
        if "bad" in examples:
            has_error, _ = self._run_ruff(examples["bad"], rule_code)
            results["bad"] = has_error

        # Good example should not trigger the rule
        if "good" in examples:
            has_error, _ = self._run_ruff(examples["good"], rule_code)
            results["good"] = not has_error

        return results

    def build_example(self, tmp_path: Path, rule_code: str) -> Dict[str, Path]:
        """Build example files for a given rule code."""
        # Try to get from cache first
        cache_file = self.cache_dir / f"{rule_code.lower()}.html"
        if cache_file.exists():
            html_content = cache_file.read_text()
        else:
            html_content = self._fetch_rule_page(rule_code)
            cache_file.write_text(html_content)

        examples = self._extract_examples(html_content)
        result = {}

        for key, code in examples.items():
            if code:
                file_path = self._write_example_file(tmp_path, f"{rule_code}_{key}", code)
                result[key] = file_path

        return result


@pytest.fixture
def ruff_example_builder():
    """Fixture that provides a RuffExampleBuilder instance."""
    return RuffExampleBuilder()


@pytest.fixture
def ruff_examples(tmp_path, ruff_example_builder, request):
    """
    Fixture that generates example code for specified Ruff rules.

    Usage:
    @pytest.mark.parametrize('rule_codes', [['E501'], ['F401', 'F403']])
    def test_something(ruff_examples):
        examples = ruff_examples
        # Access examples['E501']['bad'] for the bad example
        # Access examples['E501']['good'] for the good example
    """
    rule_codes = request.param if hasattr(request, "param") else []
    examples = {}

    for rule_code in rule_codes:
        examples[rule_code] = ruff_example_builder.build_example(tmp_path, rule_code)

    return examples


# Example test using the fixture
@pytest.mark.parametrize(
    "rule_codes",
    [
        ["E501"],  # Line too long
        ["F401"],  # Unused import
        ["E711"],  # Comparison with None
    ],
)
def test_ruff_examples(ruff_examples):
    """Test that examples are properly generated for different rules."""
    for rule_code in ruff_examples:
        assert "bad" in ruff_examples[rule_code], f"No 'bad' example for {rule_code}"
        assert "good" in ruff_examples[rule_code], f"No 'good' example for {rule_code}"

        bad_path = ruff_examples[rule_code]["bad"]
        good_path = ruff_examples[rule_code]["good"]

        assert bad_path.exists(), f"Bad example file missing for {rule_code}"
        assert good_path.exists(), f"Good example file missing for {rule_code}"
        assert bad_path.read_text(), f"Bad example is empty for {rule_code}"
        assert good_path.read_text(), f"Good example is empty for {rule_code}"
