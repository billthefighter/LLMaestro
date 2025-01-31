"""Prompt loading and management utilities."""
import json
from pathlib import Path
from typing import Dict, Optional, TypeVar

from src.prompts.base import BasePrompt

T = TypeVar("T", bound=BasePrompt)


class FilePrompt(BasePrompt):
    """Prompt implementation that loads from and saves to files."""

    file_path: Optional[Path] = None

    async def save(self) -> bool:
        """Save the prompt to a file."""
        if not self.file_path:
            return False

        try:
            with open(self.file_path, "w") as f:
                f.write(self.to_json(pretty=True))
            return True
        except Exception as e:
            print(f"Error saving prompt to {self.file_path}: {e}")
            return False

    @classmethod
    async def load(cls, identifier: str) -> Optional["FilePrompt"]:
        """Load a prompt from a file."""
        path = Path(identifier)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
                return cls(**data, file_path=path)
        except Exception as e:
            print(f"Error loading prompt from {path}: {e}")
            return None


class PromptLoader:
    """Loads and manages prompts from storage."""

    def __init__(self, prompt_types: Optional[Dict[str, type[BasePrompt]]] = None):
        self.prompt_types = prompt_types or {"file": FilePrompt}
        self.prompts: Dict[str, BasePrompt] = {}

    async def load_prompt(self, prompt_type: str, identifier: str) -> Optional[BasePrompt]:
        """Load a prompt using the appropriate implementation."""
        prompt_class = self.prompt_types.get(prompt_type)
        if not prompt_class:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        prompt = await prompt_class.load(identifier)
        if prompt:
            self.prompts[prompt.metadata.type] = prompt
        return prompt

    def get_prompt(self, task_type: str) -> Optional[BasePrompt]:
        """Get a loaded prompt by task type."""
        return self.prompts.get(task_type)

    async def save_all(self) -> Dict[str, bool]:
        """Save all loaded prompts."""
        results = {}
        for task_type, prompt in self.prompts.items():
            results[task_type] = await prompt.save()
        return results


# Optional implementations that require additional dependencies
try:
    import boto3

    class S3Prompt(BasePrompt):
        """Implementation for AWS S3 storage."""

        bucket: str
        key: str

        async def save(self) -> bool:
            """Save the prompt to S3."""
            try:
                s3 = boto3.client("s3")
                data = self.to_json()
                s3.put_object(Bucket=self.bucket, Key=self.key, Body=data)
                return True
            except Exception as e:
                print(f"Error saving prompt to S3: {e}")
                return False

        @classmethod
        async def load(cls, identifier: str) -> Optional["S3Prompt"]:
            """Load a prompt from S3 (format: bucket/key)."""
            try:
                s3 = boto3.client("s3")
                bucket, key = identifier.split("/", 1)
                response = s3.get_object(Bucket=bucket, Key=key)
                data = json.loads(response["Body"].read())
                return cls(**data, bucket=bucket, key=key)
            except Exception as e:
                print(f"Error loading prompt from S3: {e}")
                return None

except ImportError:
    pass  # S3Prompt will not be available

try:
    import git

    class GitRepoPrompt(BasePrompt):
        """Implementation for prompts stored in a git repository."""

        repo_path: Path
        file_path: Path
        branch: str = "main"

        async def save(self) -> bool:
            """Save the prompt and commit changes."""
            try:
                # Save the file
                full_path = self.repo_path / self.file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(self.to_json(pretty=True))

                # Commit changes
                repo = git.Repo(self.repo_path)
                repo.index.add([str(self.file_path)])
                repo.index.commit(f"Update prompt: {self.name} to version {self.version}")

                return True
            except Exception as e:
                print(f"Error saving prompt to git: {e}")
                return False

        @classmethod
        async def load(cls, identifier: str) -> Optional["GitRepoPrompt"]:
            """Load from git (format: repo_path:file_path[@branch])."""
            try:
                # Parse identifier
                parts = identifier.split(":")
                if len(parts) != 2:
                    raise ValueError("Invalid identifier format")

                repo_path = Path(parts[0])
                file_info = parts[1].split("@")
                file_path = Path(file_info[0])
                branch = file_info[1] if len(file_info) > 1 else "main"

                # Load from git
                repo = git.Repo(repo_path)
                repo.git.checkout(branch)

                with open(repo_path / file_path) as f:
                    data = json.load(f)

                return cls(**data, repo_path=repo_path, file_path=file_path, branch=branch)
            except Exception as e:
                print(f"Error loading prompt from git: {e}")
                return None

except ImportError:
    pass  # GitRepoPrompt will not be available
