#!/usr/bin/env python
"""Script to find all classes that inherit from PersistentModel in the src directory."""

import ast
import os
from pathlib import Path
from typing import List, Set, Tuple


def is_persistent_model_subclass(node: ast.ClassDef, imported_names: Set[str]) -> bool:
    """Check if a class inherits from PersistentModel."""
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in imported_names:
            return True
        elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
            # Handle cases like llmaestro.core.persistence.PersistentModel
            return base.attr == "PersistentModel"
    return False


def find_persistent_model_imports(node: ast.Module) -> Set[str]:
    """Find all names that could refer to PersistentModel in the current file."""
    persistent_names = {"PersistentModel"}

    for stmt in node.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module and "persistence" in stmt.module:
            for name in stmt.names:
                if name.name == "PersistentModel":
                    if name.asname:
                        persistent_names.add(name.asname)
                    else:
                        persistent_names.add(name.name)

    return persistent_names


def analyze_file(file_path: Path) -> List[Tuple[str, str]]:
    """Analyze a single Python file for PersistentModel subclasses."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            print(f"Error parsing {file_path}")
            return []

    persistent_names = find_persistent_model_imports(tree)
    results = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and is_persistent_model_subclass(node, persistent_names):
            results.append((node.name, str(file_path)))

    return results


def find_persistent_models(src_dir: Path) -> List[Tuple[str, str]]:
    """Recursively find all classes that inherit from PersistentModel."""
    results = []

    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                results.extend(analyze_file(file_path))

    return results


def main():
    """Main entry point."""
    src_dir = Path(__file__).parent.parent / "src"

    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}")
        return

    results = find_persistent_models(src_dir)

    if not results:
        print("No classes inheriting from PersistentModel found.")
        return

    print("\nClasses inheriting from PersistentModel:")
    print("-" * 80)

    # Sort by file path for better organization
    results.sort(key=lambda x: (x[1], x[0]))

    current_file = None
    for class_name, file_path in results:
        if file_path != current_file:
            print(f"\nFile: {file_path}")
            current_file = file_path
        print(f"  - {class_name}")


if __name__ == "__main__":
    main()
