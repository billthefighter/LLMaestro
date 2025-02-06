#!/usr/bin/env python
"""Script to update changelist.md based on git changes."""

import asyncio
import subprocess
from datetime import datetime
from typing import List, Tuple

from src.applications.changelistmanager.app import ChangelistEntry, ChangelistManager


def get_git_diff() -> Tuple[str, List[str]]:
    """Get the git diff content and list of changed files.

    Returns:
        Tuple of (diff_content, changed_files)
    """
    # Get diff content
    diff_cmd = ["git", "diff", "--cached", "origin/main"]
    diff_content = subprocess.check_output(diff_cmd, text=True)

    # Get changed files
    files_cmd = ["git", "diff", "--cached", "--name-only", "origin/main"]
    changed_files = subprocess.check_output(files_cmd, text=True).splitlines()

    return diff_content, changed_files


async def main():
    """Main function to update changelist."""
    try:
        # Get git diff
        diff_content, changed_files = get_git_diff()
        if not diff_content:
            print("No changes to process")
            return

        # Initialize changelist manager
        manager = ChangelistManager()

        # Process changes
        response = await manager.process_changes(diff_content)

        # Create changelist entry
        entry = ChangelistEntry(
            summary=response.summary,
            files_changed=changed_files,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Update changelist file
        await manager.update_changelist_file(entry)

        # Check if READMEs need updates
        if response.needs_readme_updates:
            print("\nWarning: The following README files may need updates:")
            for readme in response.affected_readmes:
                print(f"- {readme}")
            if response.suggested_updates:
                print("\nSuggested updates:")
                for readme, suggestion in response.suggested_updates.items():
                    print(f"\n{readme}:")
                    print(suggestion)

        print("\nChangelist updated successfully!")

    except Exception as e:
        print(f"Error updating changelist: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
