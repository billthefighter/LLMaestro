#!/usr/bin/env python
import subprocess
from pathlib import Path

import tomli
import tomli_w


def get_latest_version_tag():
    try:
        # Get all tags sorted by version
        result = subprocess.run(["git", "tag", "--sort=-v:refname"], capture_output=True, text=True, check=True)
        tags = result.stdout.strip().split("\n")

        # Filter for version tags (v*)
        version_tags = [tag for tag in tags if tag.startswith("v")]
        if not version_tags:
            return None

        # Return latest version tag without the 'v' prefix
        return version_tags[0][1:]  # Remove 'v' prefix
    except subprocess.CalledProcessError:
        return None


def main():
    # Get the latest version tag
    version = get_latest_version_tag()
    if not version:
        print("No version tag found")
        return 1

    # Read pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("pyproject.toml not found")
        return 1

    # Update version in pyproject.toml
    with open(pyproject_path, "rb") as f:
        config = tomli.load(f)

    current_version = config["tool"]["poetry"]["version"]

    if current_version != version:
        config["tool"]["poetry"]["version"] = version
        with open(pyproject_path, "wb") as f:
            tomli_w.dump(config, f)
        print(f"Updated version from {current_version} to {version}")
        return 1  # Return 1 to indicate file was modified

    return 0


if __name__ == "__main__":
    exit(main())
