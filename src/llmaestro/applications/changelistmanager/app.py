"""Changelist Manager Application.

This application manages automatic changelist generation and README validation
by analyzing git diffs and using LLM to generate summaries and validate documentation.
"""
from pathlib import Path
from typing import Dict, List, Optional, Set, Type, cast

from llmaestro.agents.agent_pool import AgentPool
from llmaestro.chains.chains import ChainGraph, ChainNode, NodeType, OutputTransform
from llmaestro.config.agent import AgentTypeConfig
from llmaestro.config.manager import ConfigurationManager
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.llm.interfaces.factory import create_llm_interface
from llmaestro.prompts.loader import PromptLoader
from pydantic import BaseModel


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
        llm_interface: Optional[BaseLLMInterface] = None,
    ):
        """Initialize the Changelist Manager.

        Args:
            output_model: Pydantic model for LLM response validation
            api_key: Optional API key for LLM provider
            config_path: Optional path to config file
            llm_interface: Optional LLM interface for testing
        """
        self.config = ConfigurationManager()
        self.output_model = output_model
        self.prompt_loader = PromptLoader()
        self.agent_pool = AgentPool()

        # Use provided LLM interface or create a new one
        if llm_interface:
            self.llm = llm_interface
        else:
            # Initialize LLM interface
            agent_config = AgentTypeConfig(
                provider=self.config.llm.provider,
                model_name=self.config.llm.model_name,
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
        # Create a chain graph for processing diff
        chain = ChainGraph(id="process_changes", llm=self.llm, prompt_loader=self.prompt_loader)

        # Create and add the changelist generation node
        changelist_node = await ChainNode.create(
            task_type="generate_changelist",
            prompt_loader=self.prompt_loader,
            node_type=NodeType.SEQUENTIAL,
            output_transform=self._create_output_transform(),
        )
        chain.add_node(changelist_node)

        # Execute chain with diff content
        result = await chain.execute(diff_content=diff_content)
        return next(iter(result.values()))  # Get the first (and only) result

    async def validate_readmes(self, readmes: List[str], changes: str) -> Dict[str, bool]:
        """Validate if READMEs need updates based on changes.

        Args:
            readmes: List of README file paths
            changes: Description of changes made

        Returns:
            Dict mapping README paths to whether they need updates
        """
        # Create chain graph for README validation
        chain = ChainGraph(id="validate_readmes", llm=self.llm, prompt_loader=self.prompt_loader)

        # Create and add the README validation node
        validation_node = await ChainNode.create(
            task_type="validate_readmes",
            prompt_loader=self.prompt_loader,
            node_type=NodeType.SEQUENTIAL,
            output_transform=self._create_output_transform(),
        )
        chain.add_node(validation_node)

        # Execute chain with readmes and changes
        result = await chain.execute(readmes=readmes, changes=changes)
        return next(iter(result.values()))  # Get the first (and only) result

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
