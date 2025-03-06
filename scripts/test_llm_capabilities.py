import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import httpx
from pydantic import BaseModel

from llmaestro.llm.interfaces.base import BaseLLMInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseCapabilityTester(ABC):
    """Base class for testing LLM capabilities across different providers."""

    # Common test messages that can be overridden by provider-specific implementations
    BASIC_TEST_MESSAGE = [{"role": "user", "content": "Hello, how are you?"}]
    SYSTEM_PROMPT_TEST_MESSAGE = [
        {"role": "system", "content": "You are a helpful assistant that speaks like a pirate."},
        {"role": "user", "content": "Introduce yourself briefly."}
    ]
    STOP_SEQUENCE_TEST_MESSAGE = [{"role": "user", "content": "Count from 1 to 10."}]

    def __init__(self, api_key: str):
        """Initialize the base capability tester.

        Args:
            api_key: API key for the LLM provider
        """
        self.api_key = api_key
        self.capability_results: Dict[str, Dict[str, Any]] = {}
        self.pricing_data: Dict[str, Dict[str, float]] = {}
        self.include_deprecated: bool = False
        self.include_preview: bool = False
        self.include_non_provider_owned: bool = False


    @abstractmethod
    async def get_models(self) -> List[Any]:
        """Get available models from the provider.

        Returns:
            List of model objects
        """
        pass

    @abstractmethod
    async def fetch_pricing_data(self) -> Dict[str, Dict[str, float]]:
        """Fetch pricing data for models.

        Returns:
            Dictionary mapping model IDs to pricing information
        """
        pass

    @abstractmethod
    def _get_fallback_pricing(self) -> Dict[str, Dict[str, float]]:
        """Get fallback pricing data for common models.

        Returns:
            Dictionary mapping model IDs to pricing information
        """
        pass

    @abstractmethod
    async def test_model_capabilities(self, model_id: str) -> Dict[str, Any]:
        """Test a model for its capabilities.

        Args:
            model_id: ID of the model to test

        Returns:
            Dictionary of capability test results
        """
        pass

    @abstractmethod
    async def _test_basic_chat(self, model_id: str) -> bool:
        """Test if a model supports basic chat functionality.

        Args:
            model_id: ID of the model to test

        Returns:
            True if the model supports basic chat, False otherwise
        """
        pass

    @abstractmethod
    async def _test_context_window(self, model_id: str) -> int:
        """Test the context window size of a model.

        Args:
            model_id: ID of the model to test

        Returns:
            Estimated context window size in tokens
        """
        pass

    @abstractmethod
    async def _test_streaming(self, model_id: str) -> bool:
        """Test if a model supports streaming.

        Args:
            model_id: ID of the model to test

        Returns:
            True if the model supports streaming, False otherwise
        """
        pass

    @abstractmethod
    async def _test_system_prompt(self, model_id: str, token_usage: Dict[str, float]) -> bool:
        """Test if a model supports system prompts.

        Args:
            model_id: ID of the model to test
            token_usage: Dictionary to track token usage

        Returns:
            True if the model supports system prompts, False otherwise
        """
        pass

    @abstractmethod
    async def _test_stop_sequences(self, model_id: str, token_usage: Dict[str, float]) -> bool:
        """Test if a model supports stop sequences.

        Args:
            model_id: ID of the model to test
            token_usage: Dictionary to track token usage

        Returns:
            True if the model supports stop sequences, False otherwise
        """
        pass

    def is_model_deprecated(self, model_id: str) -> bool:
        """Check if a model is deprecated.

        Args:
            model_id: ID of the model to check

        Returns:
            True if the model is deprecated, False otherwise
        """
        return False  # Default implementation, should be overridden by provider-specific classes

    def is_model_preview(self, model_id: str) -> bool:
        """Check if a model is a preview model.

        Args:
            model_id: ID of the model to check

        Returns:
            True if the model is a preview model, False otherwise
        """
        return False  # Default implementation, should be overridden by provider-specific classes

    def _find_best_pricing_match(self, model_id: str) -> Tuple[float, float]:
        """Find the best pricing match for a model.

        Args:
            model_id: ID of the model to find pricing for

        Returns:
            Tuple of (input_cost_per_1k_tokens, output_cost_per_1k_tokens)
        """
        # Default implementation that can be overridden
        if model_id in self.pricing_data:
            pricing = self.pricing_data[model_id]
            return pricing.get("input_cost_per_1k_tokens", 0.0), pricing.get("output_cost_per_1k_tokens", 0.0)

        # Try to find a match by prefix
        for pricing_model_id, pricing in self.pricing_data.items():
            if model_id.startswith(pricing_model_id):
                return pricing.get("input_cost_per_1k_tokens", 0.0), pricing.get("output_cost_per_1k_tokens", 0.0)

        # Return default pricing if no match found
        logger.warning(f"No pricing data found for model {model_id}, using default pricing")
        return 0.0, 0.0

    def _add_fallback_pricing(self, pricing_data: Dict[str, Dict[str, float]]) -> None:
        """Add fallback pricing data to the pricing data dictionary.

        Args:
            pricing_data: Dictionary of pricing data to update
        """
        fallback_pricing = self._get_fallback_pricing()
        for model_id, pricing in fallback_pricing.items():
            if model_id not in pricing_data:
                pricing_data[model_id] = pricing

    @abstractmethod
    def generate_model_code(self, model_id: str) -> str:
        """Generate code for a model.

        Args:
            model_id: ID of the model to generate code for

        Returns:
            Generated code as a string
        """
        pass

    @abstractmethod
    def update_models_file(self, models_to_add: List[str]) -> None:
        """Update the models file with new models.

        Args:
            models_to_add: List of model IDs to add to the models file
        """
        pass

    @abstractmethod
    async def update_pricing_in_file(self, pricing_data: Dict[str, Dict[str, float]]) -> None:
        """Update pricing data in the models file.

        Args:
            pricing_data: Dictionary of pricing data to update
        """
        pass


async def run_capability_tests(
    tester: BaseCapabilityTester,
    models_to_test: Optional[List[str]] = None,
    update_file: bool = False,
    include_deprecated: bool = False,
    include_preview: bool = False,
    include_non_provider_owned: bool = False
) -> None:
    """Run capability tests for models.

    Args:
        tester: Capability tester instance
        models_to_test: List of model IDs to test, or None to test all available models
        update_file: Whether to update the models file with test results
        include_deprecated: Whether to include deprecated models
        include_preview: Whether to include preview models
        include_non_provider_owned: Whether to include models not owned by the provider
    """
    tester.include_deprecated = include_deprecated
    tester.include_preview = include_preview
    tester.include_non_provider_owned = include_non_provider_owned

    # Fetch pricing data
    pricing_data = await tester.fetch_pricing_data()
    tester.pricing_data = pricing_data

    # Get available models if not specified
    if not models_to_test:
        models = await tester.get_models()
        models_to_test = [model.id for model in models]

    # Filter models based on inclusion flags
    filtered_models = []
    for model_id in models_to_test:
        if tester.is_model_deprecated(model_id) and not include_deprecated:
            logger.info(f"Skipping deprecated model: {model_id}")
            continue
        if tester.is_model_preview(model_id) and not include_preview:
            logger.info(f"Skipping preview model: {model_id}")
            continue
        filtered_models.append(model_id)

    # Test each model
    for model_id in filtered_models:
        try:
            capabilities = await tester.test_model_capabilities(model_id)
            tester.capability_results[model_id] = capabilities
            logger.info(f"Capabilities for {model_id}: {capabilities}")
        except Exception as e:
            logger.error(f"Error testing model {model_id}: {e}")

    # Update models file if requested
    if update_file:
        tester.update_models_file(filtered_models)
        await tester.update_pricing_in_file(pricing_data)


async def update_pricing_only(
    tester: BaseCapabilityTester,
    update_file: bool = False,
    include_deprecated: bool = False,
    include_preview: bool = False,
    include_non_provider_owned: bool = False
) -> None:
    """Update pricing data only.

    Args:
        tester: Capability tester instance
        update_file: Whether to update the models file with pricing data
        include_deprecated: Whether to include deprecated models
        include_preview: Whether to include preview models
        include_non_provider_owned: Whether to include models not owned by the provider
    """
    tester.include_deprecated = include_deprecated
    tester.include_preview = include_preview
    tester.include_non_provider_owned = include_non_provider_owned

    # Fetch pricing data
    pricing_data = await tester.fetch_pricing_data()

    # Update models file if requested
    if update_file:
        await tester.update_pricing_in_file(pricing_data)
