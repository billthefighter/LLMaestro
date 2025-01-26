#!/usr/bin/env python3
"""Pre-commit hook to update Git metadata in prompt templates."""

import subprocess
import sys
from pathlib import Path
import yaml

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

def is_new_file(file_path: str) -> bool:
    """Check if this is a new file in Git."""
    try:
        subprocess.check_output(["git", "ls-files", file_path])
        return False
    except subprocess.CalledProcessError:
        return True

def update_prompt_metadata(file_path: str) -> None:
    """Update the Git metadata in a prompt template."""
    try:
        with open(file_path) as f:
            data = yaml.safe_load(f)
            
        commit_hash, author = get_git_info(file_path)
        
        # Initialize git_metadata if it doesn't exist
        if "git_metadata" not in data:
            data["git_metadata"] = {
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
            data["git_metadata"]["last_modified"] = {
                "commit": commit_hash,
                "author": author
            }
        
        # Write back to the file
        with open(file_path, "w") as f:
            yaml.dump(data, f, sort_keys=False, indent=2)
            
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