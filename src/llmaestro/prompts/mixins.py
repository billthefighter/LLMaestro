import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.models import ModelFamily
from llmaestro.llm.token_utils import TokenCounter
from llmaestro.prompts.types import VersionInfo
from pydantic import BaseModel, Field, PrivateAttr


class TokenCountingMixin(BaseModel, ABC):
    """Mixin for token counting functionality in prompts.

    This mixin provides token counting capabilities for prompts, including:
    - Token estimation for prompts with variables
    - Context window validation
    - LLM registry management for token counting

    To use this mixin, your class must:
    1. Have system_prompt and user_prompt attributes
    2. Implement _extract_template_vars() method
    3. Implement render() method if you want accurate token counting with variables
    4. Initialize with an LLMRegistry instance
    """

    _token_counter: TokenCounter = PrivateAttr()
    _llm_registry: LLMRegistry = PrivateAttr()

    def __init__(self, llm_registry: LLMRegistry, **data):
        """Initialize the mixin with required dependencies.

        Args:
            llm_registry: LLM registry for token counting
            **data: Additional data for parent class initialization
        """
        super().__init__(**data)
        self._llm_registry = llm_registry
        self._token_counter = TokenCounter(llm_registry=llm_registry)

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Get the system prompt template."""
        pass

    @property
    @abstractmethod
    def user_prompt(self) -> str:
        """Get the user prompt template."""
        pass

    @abstractmethod
    def _extract_template_vars(self) -> Set[str]:
        """Extract required variables from the prompt template."""
        pass

    @abstractmethod
    def render(self, **kwargs) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Render the prompt template with variables."""
        pass

    @property
    def token_counter(self) -> TokenCounter:
        """Get the token counter instance."""
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
                system, user, _ = self.render(**variables)
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


class VersionMixin(BaseModel):
    """Mixin for version control functionality in prompts."""

    current_version: VersionInfo
    version_history: List[VersionInfo] = Field(default_factory=list)

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

    def model_dump_json(self, **kwargs) -> str:
        """Override to handle datetime serialization in versions."""
        data = self.model_dump(**kwargs)
        if "current_version" in data and "timestamp" in data["current_version"]:
            data["current_version"]["timestamp"] = data["current_version"]["timestamp"].isoformat()
        for version in data.get("version_history", []):
            if "timestamp" in version:
                version["timestamp"] = version["timestamp"].isoformat()
        return json.dumps(data, **kwargs)
