"""Changelist Manager Application.

This application manages automatic changelist generation and README validation
by analyzing git diffs and using LLM to generate summaries and validate documentation.
"""
from pathlib import Path
from typing import Dict, List, Optional, Set, Type, cast

from pydantic import BaseModel

from src.agents.agent_pool import AgentPool
from src.core.config import get_config
from src.core.models import AgentConfig
from src.llm.chains import ChainStep, OutputTransform, SequentialChain
from src.llm.interfaces.factory import create_llm_interface
from src.prompts.loader import PromptLoader


class ChangelistEntry(BaseModel):
    """Model for a single changelist entry."""

    summary: str
    files_changed: List[str]
    timestamp: str


class ChangelistResponse(BaseModel):
    """Model for the LLM response when generating changelists."""

    summary: str
    affected_readmes: List[str]
    needs_readme_updates: bool
    suggested_updates: Optional[Dict[str, str]] = None


class ChangelistManager:
    """Application for managing changelists and README validation."""

    def __init__(
        self,
        output_model: Type[BaseModel] = ChangelistResponse,
        api_key: Optional[str] = None,
        config_path: Optional[Path] = None,
    ):
        """Initialize the Changelist Manager.

        Args:
            output_model: Pydantic model for LLM response validation
            api_key: Optional API key for LLM provider
            config_path: Optional path to config file
        """
        self.config = get_config()
        self.output_model = output_model
        self.prompt_loader = PromptLoader()
        self.agent_pool = AgentPool()

        # Initialize LLM interface
        agent_config = AgentConfig(
            provider=self.config.llm.provider,
            model_name=self.config.llm.model,
            api_key=api_key or self.config.llm.api_key,
            max_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature,
        )
        self.llm = create_llm_interface(agent_config)

    def _create_output_transform(self) -> OutputTransform:
        """Create an output transform for JSON responses."""
        return cast(OutputTransform, lambda response: self.output_model.model_validate_json(response.content))

    async def process_changes(self, diff_content: str) -> ChangelistResponse:
        """Process git diff and generate changelist entry.

        Args:
            diff_content: Git diff content as string

        Returns:
            ChangelistResponse containing summary and README validation info
        """
        # Create chain for processing diff
        chain = SequentialChain(
            steps=[ChainStep(task_type="generate_changelist", output_transform=self._create_output_transform())],
            llm=self.llm,
            prompt_loader=self.prompt_loader,
        )

        # Execute chain with diff content
        result = await chain.execute(diff_content=diff_content)
        return result

    async def validate_readmes(self, readmes: List[str], changes: str) -> Dict[str, bool]:
        """Validate if READMEs need updates based on changes.

        Args:
            readmes: List of README file paths
            changes: Description of changes made

        Returns:
            Dict mapping README paths to whether they need updates
        """
        # Create chain for README validation
        chain = SequentialChain(
            steps=[ChainStep(task_type="validate_readmes", output_transform=self._create_output_transform())],
            llm=self.llm,
            prompt_loader=self.prompt_loader,
        )

        # Execute chain with readmes and changes
        result = await chain.execute(readmes=readmes, changes=changes)
        return result

    async def update_changelist_file(self, new_entry: ChangelistEntry) -> None:
        """Update the changelist.md file with a new entry.

        Args:
            new_entry: New changelist entry to append
        """
        changelist_path = Path("changelist.md")

        # Create file if it doesn't exist
        if not changelist_path.exists():
            changelist_path.write_text("# Changelist\n\n")

        # Append new entry
        with changelist_path.open("a") as f:
            f.write(f"\n## {new_entry.timestamp}\n")
            f.write(f"\n{new_entry.summary}\n")
            f.write("\nFiles changed:\n")
            for file in new_entry.files_changed:
                f.write(f"- {file}\n")
            f.write("\n---\n")

    @staticmethod
    def get_readme_files(changed_files: List[str]) -> Set[str]:
        """Get all README files from a list of changed files.

        Args:
            changed_files: List of changed file paths

        Returns:
            Set of README file paths
        """
        return {f for f in changed_files if f.lower().endswith("readme.md")}
