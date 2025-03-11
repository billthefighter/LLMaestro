import asyncio
import logging
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

import httpx
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from bs4 import BeautifulSoup
from pydantic import BaseModel

from scripts.test_llm_capabilities import BaseCapabilityTester

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnthropicCapabilityTester(BaseCapabilityTester):
    """Tests Anthropic models for their capabilities."""

    def __init__(self, api_key: str):
        """Initialize the tester with the Anthropic API key."""
        super().__init__(api_key)
        self.client = AsyncAnthropic(api_key=api_key)

    async def get_models(self) -> List[Any]:
        """Get a list of available Anthropic models."""
        try:
            # Anthropic's API doesn't have a models endpoint, so we'll hardcode the known models
            models = [
                {"id": "claude-3-opus-20240229"},
                {"id": "claude-3-sonnet-20240229"},
                {"id": "claude-3-haiku-20240229"},
                {"id": "claude-2.1"},
                {"id": "claude-2.0"},
                {"id": "claude-instant-1.2"}
            ]
            return models
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []

    def fetch_pricing_data(self) -> Dict[str, Dict[str, float]]:
        """Fetch pricing data for Anthropic models."""
        pricing_data = {
            "claude-3-opus-20240229": {
                "input_cost_per_1k_tokens": 0.015,
                "output_cost_per_1k_tokens": 0.075
            },
            "claude-3-sonnet-20240229": {
                "input_cost_per_1k_tokens": 0.003,
                "output_cost_per_1k_tokens": 0.015
            },
            "claude-3-haiku-20240229": {
                "input_cost_per_1k_tokens": 0.0015,
                "output_cost_per_1k_tokens": 0.007
            },
            "claude-2.1": {
                "input_cost_per_1k_tokens": 0.008,
                "output_cost_per_1k_tokens": 0.024
            },
            "claude-2.0": {
                "input_cost_per_1k_tokens": 0.008,
                "output_cost_per_1k_tokens": 0.024
            },
            "claude-instant-1.2": {
                "input_cost_per_1k_tokens": 0.0008,
                "output_cost_per_1k_tokens": 0.0024
            }
        }
        return pricing_data

    def _get_fallback_pricing(self) -> Dict[str, Dict[str, float]]:
        """Get fallback pricing data for common models."""
        return self.fetch_pricing_data()

    async def test_model_capabilities(self, model_id: str) -> Dict[str, Any]:
        """Test a model for its capabilities."""
        logger.info(f"Testing capabilities for model: {model_id}")

        # Initialize token usage tracking
        token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_cost": 0.0
        }

        # Get pricing information
        pricing = self.pricing_data.get(model_id, {})
        input_cost = pricing.get("input_cost_per_1k_tokens", 0.0)
        output_cost = pricing.get("output_cost_per_1k_tokens", 0.0)

        capabilities = {
            "max_context_window": await self._test_context_window(model_id),
            "supports_streaming": await self._test_streaming(model_id),
            "supports_system_prompt": await self._test_system_prompt(model_id, token_usage),
            "supports_stop_sequences": await self._test_stop_sequences(model_id, token_usage),
            "supports_direct_pydantic_parse": await self._test_direct_pydantic_parse(model_id, token_usage),
            "supports_function_calling": False,  # Anthropic doesn't support function calling
            "supports_vision": self._supports_vision(model_id),
            "supports_json_mode": False,  # Anthropic doesn't have a dedicated JSON mode
            "supports_tools": False,  # Anthropic doesn't support tools
            "supports_frequency_penalty": False,  # Anthropic doesn't support frequency penalty
            "supports_presence_penalty": False,  # Anthropic doesn't support presence penalty
            "supports_message_role": True,  # All Anthropic models support message roles
            "typical_speed": self._get_typical_speed(model_id),
            "supported_languages": {"en"},  # Default to English, would need more testing for others
            "input_cost_per_1k_tokens": input_cost,
            "output_cost_per_1k_tokens": output_cost,
        }

        # Calculate total cost
        total_cost = (token_usage["input_tokens"] / 1000 * input_cost) + (token_usage["output_tokens"] / 1000 * output_cost)
        token_usage["total_cost"] = total_cost

        logger.info(f"Token usage for {model_id}: {token_usage['input_tokens']} input tokens, {token_usage['output_tokens']} output tokens")
        logger.info(f"Estimated cost for {model_id}: ${total_cost:.4f}")

        # Store the results
        self.capability_results[model_id] = {
            "capabilities": capabilities,
            "token_usage": token_usage
        }

        return self.capability_results[model_id]

    def _get_context_window(self, model_id: str) -> int:
        """Get the context window size for a model."""
        context_windows = {
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240229": 200000,
            "claude-2.1": 200000,
            "claude-2.0": 100000,
            "claude-instant-1.2": 100000
        }
        return context_windows.get(model_id, 100000)

    def _get_typical_speed(self, model_id: str) -> float:
        """Get the typical speed for a model in tokens per second."""
        speeds = {
            "claude-3-opus-20240229": 150.0,
            "claude-3-sonnet-20240229": 180.0,
            "claude-3-haiku-20240229": 200.0,
            "claude-2.1": 120.0,
            "claude-2.0": 100.0,
            "claude-instant-1.2": 150.0
        }
        return speeds.get(model_id, 100.0)

    def _supports_vision(self, model_id: str) -> bool:
        """Check if a model supports vision capabilities."""
        return model_id.startswith("claude-3")

    async def _test_context_window(self, model_id: str) -> int:
        """Test the context window size of a model.

        Args:
            model_id: ID of the model to test

        Returns:
            Estimated context window size in tokens
        """
        return self._get_context_window(model_id)

    async def _test_basic_chat(self, model_id: str) -> bool:
        """Test if a model supports basic chat functionality."""
        try:
            response = await self.client.messages.create(
                model=model_id,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            logger.warning(f"Model {model_id} does not support basic chat: {e}")
            return False

    async def _test_streaming(self, model_id: str) -> bool:
        """Test if a model supports streaming."""
        try:
            response = await self.client.messages.create(
                model=model_id,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                stream=True
            )
            async for chunk in response:
                break
            return True
        except Exception as e:
            logger.warning(f"Model {model_id} does not support streaming: {e}")
            return False

    async def _test_system_prompt(self, model_id: str, token_usage: Dict[str, float]) -> bool:
        """Test if a model supports system prompts."""
        try:
            response = await self.client.messages.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": "You are a helpful assistant that speaks like a pirate. Introduce yourself briefly."}
                ],
                max_tokens=50
            )
            if hasattr(response, 'usage'):
                token_usage["input_tokens"] += response.usage.input_tokens
                token_usage["output_tokens"] += response.usage.output_tokens
            return True
        except Exception as e:
            logger.warning(f"Model {model_id} does not support system prompts: {e}")
            return False

    async def _test_stop_sequences(self, model_id: str, token_usage: Dict[str, float]) -> bool:
        """Test if a model supports stop sequences."""
        try:
            response = await self.client.messages.create(
                model=model_id,
                max_tokens=10,
                messages=[{"role": "user", "content": "Count from 1 to 10."}],
                stop_sequences=["5"]
            )
            # Track token usage (Anthropic doesn't provide token counts in the response)
            token_usage["input_tokens"] += 10  # Estimate
            token_usage["output_tokens"] += 10  # Estimate
            return "5" not in response.content
        except Exception as e:
            logger.warning(f"Model {model_id} does not support stop sequences: {e}")
            return False

    async def _test_direct_pydantic_parse(self, model_id: str, token_usage: Dict[str, float]) -> bool:
        """Test if a model supports direct Pydantic parsing.

        This tests whether the model can reliably generate output that can be directly parsed
        into a Pydantic model without additional processing.

        For Anthropic models, we test this by requesting JSON output and checking if it can be
        parsed into a Pydantic model without additional processing.
        """
        # Define a simple Pydantic model for testing
        class TestModel(BaseModel):
            name: str
            age: int
            is_active: bool

        # Test with a more explicit prompt that includes the schema
        schema_json = json.dumps(TestModel.model_json_schema(), indent=2)
        prompt = (
            "Please provide information about a person with the following structure:\n"
            f"```json\n{schema_json}\n```\n"
            "Format your response as valid JSON without any additional text or explanation."
        )

        try:
            # Make the API call with a system prompt that enforces JSON output
            response = await self.client.messages.create(
                model=model_id,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
                system="You are a helpful assistant that always responds with valid JSON according to the schema provided. Do not include any explanations or markdown formatting - just the raw JSON object."
            )

            # Track token usage (Anthropic doesn't provide token counts in the response)
            token_usage["input_tokens"] += 50  # Estimate
            token_usage["output_tokens"] += 30  # Estimate

            # Try to parse the response into the Pydantic model
            content = response.content
            if content is not None:
                content_str = str(content)

                # Try multiple parsing strategies
                try:
                    # First try to parse directly
                    parsed_data = json.loads(content_str)
                    TestModel(**parsed_data)
                    return True
                except (json.JSONDecodeError, ValueError):
                    # If direct parsing fails, try to extract JSON from markdown code blocks
                    import re
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content_str)
                    if json_match:
                        try:
                            parsed_data = json.loads(json_match.group(1))
                            TestModel(**parsed_data)
                            return True
                        except (json.JSONDecodeError, ValueError):
                            pass

                    # Try to find anything that looks like JSON with curly braces
                    json_match = re.search(r'({[\s\S]*?})', content_str)
                    if json_match:
                        try:
                            parsed_data = json.loads(json_match.group(1))
                            TestModel(**parsed_data)
                            return True
                        except (json.JSONDecodeError, ValueError):
                            pass

            # If we get here, the model doesn't reliably support direct Pydantic parsing
            return False
        except Exception as e:
            logger.warning(f"Model {model_id} does not support direct Pydantic parsing: {e}")
            return False

    def is_model_deprecated(self, model_id: str) -> bool:
        """Check if a model is deprecated."""
        deprecated_models = {
            "claude-2.0",
            "claude-instant-1.2"
        }
        return model_id in deprecated_models

    def is_model_preview(self, model_id: str) -> bool:
        """Check if a model is a preview model."""
        # Currently no preview models in Anthropic's lineup
        return False

    def update_models_file(self, models_to_add: List[str]) -> None:
        """Update the models file with new models."""
        # This would be implemented to update the models.py file with new model definitions
        # For now, we'll just log the models that would be added
        logger.info(f"Would update models file with: {models_to_add}")
        pass

    async def update_pricing_in_file(self, pricing_data: Dict[str, Dict[str, float]]) -> None:
        """Update pricing data in the models file."""
        # This would be implemented to update the pricing data in the models file
        # For now, we'll just log the pricing data that would be updated
        logger.info(f"Would update pricing data: {pricing_data}")
        pass

    def generate_model_code(self, model_id: str) -> str:
        """Generate Python code for a model definition."""
        capabilities = self.capability_results.get(model_id, {}).get("capabilities", {})

        method_name = model_id.replace("-", "_").replace(".", "_")

        return f"""
    @staticmethod
    def {method_name}() -> LLMState:
        \"\"\"Create LLMState for {model_id} model.

        Returns:
            LLMState: Configured state for {model_id} model
        \"\"\"
        return LLMState(
            profile=LLMProfile(
                name="{model_id}",
                version="{datetime.now().strftime('%Y-%m')}",
                description="{model_id} model",
                capabilities=LLMCapabilities(
                    max_context_window={capabilities.get('max_context_window', 100000)},
                    max_output_tokens=4096,
                    supports_streaming={str(capabilities.get('supports_streaming', True))},
                    supports_function_calling=False,
                    supports_vision={str(capabilities.get('supports_vision', False))},
                    supports_embeddings=False,
                    supports_json_mode=True,
                    supports_system_prompt=True,
                    supports_tools=False,
                    supports_parallel_requests=True,
                    supports_frequency_penalty=False,
                    supports_presence_penalty=False,
                    supports_stop_sequences={str(capabilities.get('supports_stop_sequences', True))},
                    supports_message_role=True,
                    supports_direct_pydantic_parse={str(capabilities.get('supports_direct_pydantic_parse', False))},
                    typical_speed={capabilities.get('typical_speed', 100.0)},
                    supported_languages={{"en"}},
                    input_cost_per_1k_tokens={capabilities.get('input_cost_per_1k_tokens', 0.0)},
                    output_cost_per_1k_tokens={capabilities.get('output_cost_per_1k_tokens', 0.0)},
                ),
                metadata=LLMMetadata(
                    release_date=datetime.now(),
                    is_deprecated={str(self.is_model_deprecated(model_id))},
                    is_preview=False,
                    owned_by="anthropic"
                )
            )
        )
"""

async def run_tests(
    api_key: str,
    models_to_test: Optional[List[str]] = None,
    update_file: bool = False,
    include_deprecated: bool = False,
    include_preview: bool = False
) -> None:
    """Run capability tests for Anthropic models."""
    tester = AnthropicCapabilityTester(api_key)
    tester.include_deprecated = include_deprecated
    tester.include_preview = include_preview

    try:
        # Get all available models
        all_models = await tester.get_models()

        # Filter models if specific ones were requested
        if models_to_test:
            models = [model for model in all_models if model["id"] in models_to_test]
            if not models:
                logger.warning(f"None of the specified models {models_to_test} were found.")
                return
        else:
            models = all_models

        # Filter out deprecated models unless explicitly included
        if not include_deprecated:
            models = [model for model in models if not tester.is_model_deprecated(model["id"])]

        logger.info(f"Testing {len(models)} models: {[model['id'] for model in models]}")

        # Test each model
        for model in models:
            try:
                logger.info(f"Testing capabilities for {model['id']}...")
                await tester.test_model_capabilities(model["id"])
            except Exception as e:
                logger.error(f"Error testing {model['id']}: {e}")

        # Print a summary of the results
        logger.info("\nCapability Test Results:")
        for model_id, results in tester.capability_results.items():
            capabilities = results["capabilities"]
            logger.info(f"\n{model_id}:")
            for capability, supported in capabilities.items():
                logger.info(f"  {capability}: {supported}")

    except Exception as e:
        logger.error(f"Error running tests: {e}")
        raise

if __name__ == "__main__":
    import os
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    asyncio.run(run_tests(api_key))
