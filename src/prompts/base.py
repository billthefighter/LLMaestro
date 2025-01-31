"""Base classes for prompts."""
import json
import re
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from src.llm.models import ModelFamily
from src.llm.token_utils import TokenCounter
from src.prompts.types import PromptMetadata, VersionInfo


class BasePrompt(BaseModel):
    """Base class for all prompts."""

    name: str
    description: str
    system_prompt: str
    user_prompt: str
    metadata: PromptMetadata
    examples: Optional[List[Dict[str, Any]]] = None

    # Version control
    current_version: VersionInfo
    version_history: List[VersionInfo] = Field(default_factory=list)

    # Token counting
    _token_counter: Optional[TokenCounter] = None

    def model_dump_json(self, **kwargs) -> str:
        data = self.model_dump(**kwargs)
        if "current_version" in data and "timestamp" in data["current_version"]:
            data["current_version"]["timestamp"] = data["current_version"]["timestamp"].isoformat()
        for version in data.get("version_history", []):
            if "timestamp" in version:
                version["timestamp"] = version["timestamp"].isoformat()
        return json.dumps(data, **kwargs)

    def render(self, **kwargs) -> Tuple[str, str]:
        """Render the prompt template with the given variables."""
        self._validate_template()  # Validate before rendering
        try:
            formatted_system_prompt = self.system_prompt.format(**kwargs)
            formatted_user_prompt = self.user_prompt.format(**kwargs)
            return formatted_system_prompt, formatted_user_prompt
        except KeyError as e:
            required_vars = self._extract_template_vars()
            missing_vars = [var for var in required_vars if var not in kwargs]
            raise ValueError(f"Missing required variables: {missing_vars}. Error: {e}") from e

    def _validate_template(self) -> None:
        """Validate the template format and variables."""
        # Check for balanced braces
        for template in [self.system_prompt, self.user_prompt]:
            open_count = template.count("{")
            close_count = template.count("}")
            if open_count != close_count:
                raise ValueError(f"Unbalanced braces in template: {open_count} open, {close_count} close")

        # Check for valid variable names
        pattern = r"\{([^}]+)\}"
        for match in re.finditer(pattern, self.system_prompt + self.user_prompt):
            var_name = match.group(1)
            if not var_name.isidentifier() and not any(c in var_name for c in ":.[]"):
                raise ValueError(f"Invalid variable name in template: {var_name}")

    def _extract_template_vars(self) -> Set[str]:
        """Extract required variables from the prompt template."""
        pattern = r"\{([^}]+)\}"
        system_vars = set(re.findall(pattern, self.system_prompt))
        user_vars = set(re.findall(pattern, self.user_prompt))
        return system_vars.union(user_vars)

    @property
    def version(self) -> str:
        """Get current version number."""
        return self.current_version.number

    @property
    def author(self) -> str:
        """Get original author."""
        return self.version_history[0].author if self.version_history else self.current_version.author

    @property
    def created_at(self) -> datetime:
        """Get creation timestamp."""
        return self.version_history[0].timestamp if self.version_history else self.current_version.timestamp

    @property
    def updated_at(self) -> datetime:
        """Get last update timestamp."""
        return self.current_version.timestamp

    @property
    def age(self) -> timedelta:
        """Get the age of this prompt."""
        return datetime.now() - self.created_at

    def to_dict(self, include_history: bool = True) -> Dict[str, Any]:
        """Convert prompt to dictionary format."""
        data = self.model_dump(mode="json")
        if not include_history:
            data.pop("version_history", None)
        return data

    def to_json(self, include_history: bool = True, pretty: bool = False) -> str:
        """Convert prompt to JSON format."""
        data = self.to_dict(include_history=include_history)
        return json.dumps(data, indent=2 if pretty else None, ensure_ascii=False)

    @abstractmethod
    async def save(self) -> bool:
        """Save the prompt to its storage backend."""
        pass

    @classmethod
    @abstractmethod
    async def load(cls, identifier: str) -> Optional["BasePrompt"]:
        """Load a prompt from its storage backend."""
        pass

    @property
    def token_counter(self) -> TokenCounter:
        """Get or create token counter."""
        if self._token_counter is None:
            self._token_counter = TokenCounter()
        return self._token_counter

    def estimate_tokens(
        self, model_family: ModelFamily, model_name: str, variables: Optional[Dict] = None
    ) -> Dict[str, int]:
        """Estimate tokens for this prompt.

        Args:
            model_family: The model family (e.g. GPT, CLAUDE)
            model_name: Specific model name
            variables: Optional variable values for rendering

        Returns:
            Dict containing token counts and metadata
        """
        try:
            # Try to render with provided variables
            if variables:
                system, user = self.render(**variables)
            else:
                # Use template analysis for estimation
                system = self.system_prompt
                user = self.user_prompt

                # Add placeholder estimates for variables
                for var in self._extract_template_vars():
                    placeholder = "X" * 10  # Assume 10 chars per var
                    if "{" + var + "}" in system:
                        system = system.replace("{" + var + "}", placeholder)
                    if "{" + var + "}" in user:
                        user = user.replace("{" + var + "}", placeholder)

        except Exception:
            # Fallback to raw templates if rendering fails
            system = self.system_prompt
            user = self.user_prompt

        # Format as messages for token counter
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        # Get token estimates
        counts = self.token_counter.estimate_messages(messages, model_family, model_name)

        # Create result with both counts and metadata
        result = {
            **counts,
            "has_variables": bool(self._extract_template_vars()),
            "is_estimate": variables is None,
            "model_family": model_family.name,
            "model_name": model_name,
        }

        return result

    def validate_context(
        self, model_family: ModelFamily, model_name: str, max_completion_tokens: int, variables: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """Check if prompt fits in model's context window.

        Args:
            model_family: The model family (e.g. GPT, CLAUDE)
            model_name: Specific model name
            max_completion_tokens: Maximum tokens needed for completion
            variables: Optional variable values

        Returns:
            Tuple of (is_valid, error_message)
        """
        counts = self.estimate_tokens(model_family, model_name, variables)

        is_valid, error = self.token_counter.validate_context(
            counts["total_tokens"], max_completion_tokens, model_family, model_name
        )

        return is_valid, error or ""
