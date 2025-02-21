#!/usr/bin/env python3
"""Script to generate JSON schemas for all Pydantic models in the codebase."""

import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Dict, List, Type

from pydantic import BaseModel


def find_pydantic_models(module_path: str) -> Dict[str, Type[BaseModel]]:
    """Find all Pydantic models in a module and its submodules."""
    models = {}

    # Convert path to module name
    module_name = module_path.replace("/", ".").rstrip(".py")
    if module_name.startswith("src."):
        module_name = module_name[4:]  # Remove src. prefix

    try:
        module = importlib.import_module(module_name)

        # Find all Pydantic models in the module
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseModel)
                and obj != BaseModel
                and obj.__module__ == module.__name__
            ):
                models[f"{module_name}.{name}"] = obj

    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not process module {module_name}: {e}")

    return models


def find_python_files(start_path: str) -> List[str]:
    """Recursively find all Python files in the given directory."""
    python_files = []
    for root, _, files in os.walk(start_path):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                python_files.append(path)
    return python_files


def generate_schemas(models: Dict[str, Type[BaseModel]], output_dir: Path) -> None:
    """Generate JSON schemas for the given models and save them to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_class in models.items():
        # Generate schema
        schema = model_class.model_json_schema()

        # Create a safe filename from the model name
        safe_name = model_name.replace(".", "_").lower()
        schema_file = output_dir / f"{safe_name}.json"

        # Save schema to file
        with open(schema_file, "w") as f:
            json.dump(schema, f, indent=2)
        print(f"Generated schema for {model_name}")


def main():
    """Main function to generate schemas for all Pydantic models."""
    # Setup paths
    src_dir = Path("src")
    schema_dir = Path("schemas")

    # Find all Python files
    python_files = find_python_files(str(src_dir))

    # Find all Pydantic models
    all_models = {}
    for file_path in python_files:
        models = find_pydantic_models(file_path)
        all_models.update(models)

    # Generate schemas
    generate_schemas(all_models, schema_dir)
    print(f"\nGenerated schemas for {len(all_models)} models in {schema_dir}/")


if __name__ == "__main__":
    main()
