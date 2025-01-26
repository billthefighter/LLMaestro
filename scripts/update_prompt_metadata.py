#!/usr/bin/env python3
"""Pre-commit hook to update Git metadata and version in prompt templates."""

import subprocess
import sys
from pathlib import Path
import yaml
from typing import Optional, Dict, Any
import json
from jsonschema import validate

def get_git_info(file_path: str) -> tuple[str, str]:
    """Get the Git commit hash and author for a file."""
    try:
        # Get the current commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True
        ).strip()
        
        # Get the commit author
        author = subprocess.check_output(
            ["git", "config", "user.name"],
            text=True
        ).strip()
        
        return commit_hash, author
    except subprocess.CalledProcessError as e:
        print(f"Error getting Git info: {e}", file=sys.stderr)
        sys.exit(1)

def get_previous_version(file_path: str) -> Optional[Dict[str, Any]]:
    """Get the previous version of the file from Git."""
    try:
        # Get the previous version of the file
        previous_content = subprocess.check_output(
            ["git", "show", "HEAD:" + file_path],
            text=True,
            stderr=subprocess.DEVNULL
        )
        return yaml.safe_load(previous_content)
    except subprocess.CalledProcessError:
        return None

def determine_version_bump(current: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> str:
    """Determine how to bump the version based on changes."""
    if not previous:
        return current.get("version", "0.1.0")  # New file
        
    current_version = current.get("version", "0.0.0")
    major, minor, patch = map(int, current_version.split("."))
    
    # If major version changed manually, use that
    if current_version != previous.get("version", "0.0.0"):
        return current_version
        
    # Check for prompt changes (minor version bump)
    if (current.get("system_prompt") != previous.get("system_prompt") or
        current.get("user_prompt") != previous.get("user_prompt")):
        minor += 1
        patch = 0
    else:
        # Any other changes (patch version bump)
        patch += 1
        
    return f"{major}.{minor}.{patch}"

def is_new_file(file_path: str) -> bool:
    """Check if this is a new file in Git."""
    try:
        subprocess.check_output(["git", "ls-files", file_path])
        return False
    except subprocess.CalledProcessError:
        return True

def load_schema():
    """Load the JSON schema for prompt validation."""
    schema_path = Path(__file__).parent.parent / "llm_orchestrator" / "prompts" / "schema.json"
    with open(schema_path) as f:
        return json.load(f)

def validate_prompt(content):
    """Validate the prompt content against the JSON schema."""
    schema = load_schema()
    validate(instance=content, schema=schema)

def update_prompt_metadata(file_path: str) -> None:
    """Update the Git metadata and version in a prompt template."""
    try:
        # Read current content
        with open(file_path) as f:
            current_data = yaml.safe_load(f)
            
        # Validate the prompt against schema before proceeding
        try:
            validate_prompt(current_data)
        except Exception as e:
            print(f"Error: {file_path} failed schema validation:")
            print(str(e))
            return
        
        # Get previous version from Git
        previous_data = get_previous_version(file_path)
        
        # Update version
        current_data["version"] = determine_version_bump(current_data, previous_data)
            
        # Update Git metadata
        commit_hash, author = get_git_info(file_path)
        
        # Initialize git_metadata if it doesn't exist
        if "git_metadata" not in current_data:
            current_data["git_metadata"] = {
                "created": {
                    "commit": commit_hash,
                    "author": author
                },
                "last_modified": {
                    "commit": commit_hash,
                    "author": author
                }
            }
        else:
            # Update only the last_modified field
            current_data["git_metadata"]["last_modified"] = {
                "commit": commit_hash,
                "author": author
            }
        
        # Write back to the file
        with open(file_path, "w") as f:
            yaml.dump(current_data, f, sort_keys=False, indent=2)
            
        # Stage the updated file
        subprocess.run(["git", "add", file_path], check=True)
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main entry point."""
    # Get the list of files to process from command line arguments
    files = sys.argv[1:]
    
    for file_path in files:
        if not file_path.endswith(".yaml"):
            continue
            
        if not Path(file_path).exists():
            continue
            
        update_prompt_metadata(file_path)

if __name__ == "__main__":
    main() 