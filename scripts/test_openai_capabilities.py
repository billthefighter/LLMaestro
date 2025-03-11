#!/usr/bin/env python
"""
Script to test OpenAI models for their capabilities and update the models.py file.

This script:
1. Gets a list of OpenAI models
2. For each model, attempts to make a call that would test LLMCapabilities or VisionCapabilities
3. Updates/adds the results to the models.py file
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, cast, Tuple
import logging
import re
from bs4 import BeautifulSoup, Tag
import httpx

import openai
from openai import AsyncOpenAI
from openai.types.model import Model
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartParam
from openai.types.chat.chat_completion_content_part_image_param import ChatCompletionContentPartImageParam
from openai.types.chat.chat_completion_content_part_text_param import ChatCompletionContentPartTextParam
from pydantic import BaseModel

# Add the src directory to the path so we can import the llmaestro modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from llmaestro.llm.capabilities import LLMCapabilities, VisionCapabilities
from llmaestro.llm.models import LLMState, LLMProfile, LLMMetadata, LLMRuntimeConfig
from llmaestro.default_library.defined_providers.openai.provider import PROVIDER

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Path to the models.py file
MODELS_PY_PATH = "src/llmaestro/default_library/defined_providers/openai/models.py"

# Test messages for different capabilities
BASIC_TEST_MESSAGE: List[ChatCompletionMessageParam] = [
    ChatCompletionUserMessageParam(
        role="user",
        content="Hello, can you respond with a short greeting?"
    )
]

FUNCTION_CALLING_TEST_MESSAGE: List[ChatCompletionMessageParam] = [
    ChatCompletionUserMessageParam(
        role="user",
        content="What's the weather like in San Francisco?"
    )
]

VISION_TEST_MESSAGE: List[ChatCompletionMessageParam] = [
    ChatCompletionUserMessageParam(
        role="user",
        content=[
            ChatCompletionContentPartTextParam(
                type="text",
                text="What's in this image?"
            ),
            ChatCompletionContentPartImageParam(
                type="image_url",
                image_url={
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
            )
        ]
    )
]

JSON_MODE_TEST_MESSAGE: List[ChatCompletionMessageParam] = [
    ChatCompletionUserMessageParam(
        role="user",
        content="Return a JSON object with your name and version"
    )
]

SYSTEM_PROMPT_TEST_MESSAGE: List[ChatCompletionMessageParam] = [
    ChatCompletionSystemMessageParam(
        role="system",
        content="You are a helpful assistant."
    ),
    ChatCompletionUserMessageParam(
        role="user",
        content="Hello, who are you?"
    )
]

TOOLS_TEST_MESSAGE: List[ChatCompletionMessageParam] = [
    ChatCompletionUserMessageParam(
        role="user",
        content="What's the weather like in San Francisco?"
    )
]

TOOLS_CONFIG: List[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

def is_model_deprecated_by_name(model_id: str) -> bool:
    """Check if a model is likely deprecated based on its name."""
    # Check if this is an older dated version by comparing with current date
    model_date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', model_id)
    if model_date_match:
        model_date = datetime(int(model_date_match.group(1)), int(model_date_match.group(2)), int(model_date_match.group(3)))
        six_months_ago = datetime.now() - timedelta(days=180)
        if model_date < six_months_ago:
            return True
    else:
        # Check for YYMM format (e.g., 0613)
        model_date_match = re.search(r'(\d{2})(\d{2})', model_id)
        if model_date_match:
            # Assume 20xx for the year
            month = int(model_date_match.group(2))
            year = 2000 + int(model_date_match.group(1))
            if month > 0 and month <= 12:
                model_date = datetime(year, month, 1)
                six_months_ago = datetime.now() - timedelta(days=180)
                if model_date < six_months_ago:
                    return True

    # Known deprecated models
    deprecated_models = [
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
        "text-davinci-003",
        "text-davinci-002",
        "text-davinci-001",
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001",
        "davinci",
        "curie",
        "babbage",
        "ada"
    ]

    return model_id in deprecated_models

def is_model_preview_by_name(model_id: str) -> bool:
    """Check if a model is a preview model based on its name."""
    return "preview" in model_id.lower()

class ModelCapabilityTester:
    """Tests OpenAI models for their capabilities."""

    def __init__(self, api_key: str):
        """Initialize the tester with the OpenAI API key."""
        self.client = AsyncOpenAI(api_key=api_key)
        self.models_cache: Dict[str, Any] = {}
        self.capability_results: Dict[str, Dict[str, Any]] = {}
        self.pricing_data: Dict[str, Dict[str, float]] = {}
        self.include_deprecated: bool = False
        self.include_preview: bool = False
        self.include_non_openai_owned: bool = False
        # Models that require max_completion_tokens instead of max_tokens
        self.models_requiring_completion_tokens = [
            "o1", "o1-mini", "o1-preview",
            "o3", "o3-mini", "o3-preview"
        ]

    def is_model_deprecated(self, model_id: str) -> bool:
        """Check if a model is deprecated."""
        return is_model_deprecated_by_name(model_id)

    def is_model_preview(self, model_id: str) -> bool:
        """Check if a model is a preview model."""
        return is_model_preview_by_name(model_id)

    def requires_completion_tokens(self, model_id: str) -> bool:
        """Check if a model requires max_completion_tokens instead of max_tokens."""
        return any(model_id.startswith(prefix) for prefix in self.models_requiring_completion_tokens)

    async def get_models(self) -> List[Model]:
        """Get a list of available OpenAI models."""
        try:
            response = await self.client.models.list()
            # Filter out non-chat models
            chat_models = [model for model in response.data if self._is_chat_model(model.id)]
            return chat_models
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []

    def _is_chat_model(self, model_id: str) -> bool:
        """Check if a model is a chat model based on its ID."""
        # Filter for models that are likely to support chat completions
        chat_prefixes = [
            "gpt-", "text-davinci-", "claude-", "o1-", "o3-", "chatgpt-"
        ]
        # Exclude embedding models
        exclude_patterns = [
            "embedding", "search", "similarity", "moderation", "whisper", "tts", "dall-e"
        ]

        is_chat = any(model_id.startswith(prefix) for prefix in chat_prefixes)
        is_excluded = any(pattern in model_id.lower() for pattern in exclude_patterns)

        return is_chat and not is_excluded

    async def fetch_pricing_data(self) -> Dict[str, Dict[str, float]]:
        """Fetch pricing data from the OpenAI pricing page.

        Returns:
            Dict mapping model names to their pricing information
        """
        logger.info("Fetching pricing data from OpenAI pricing page...")
        pricing_url = "https://platform.openai.com/docs/pricing"

        try:
            # Add headers to mimic a browser request
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Cache-Control": "max-age=0"
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(pricing_url, headers=headers, follow_redirects=True, timeout=10.0)
                response.raise_for_status()

                # Parse the HTML
                soup = BeautifulSoup(response.text, 'html.parser')

                # Initialize pricing data dictionary
                pricing_data: Dict[str, Dict[str, float]] = {}

                # Find pricing tables
                pricing_tables = soup.find_all('table')

                for table in pricing_tables:
                    if not isinstance(table, Tag):
                        continue

                    # Check if this is a model pricing table
                    headers = table.find_all('th')
                    if not headers or len(headers) < 2:
                        continue

                    # Process each row in the table
                    rows = table.find_all('tr')[1:]  # Skip header row
                    for row in rows:
                        if not isinstance(row, Tag):
                            continue

                        cells = row.find_all('td')
                        if len(cells) < 3:
                            continue

                        # Extract model name and pricing
                        model_cell = cells[0]
                        if not isinstance(model_cell, Tag):
                            continue

                        model_name = model_cell.get_text().strip().lower()

                        # Clean up model name to match API model names
                        model_name = self._normalize_model_name(model_name)
                        if not model_name:
                            continue

                        # Extract input and output pricing
                        try:
                            # Different tables have different formats
                            # Try to find cells with pricing information
                            pricing_cells = [c for c in cells[1:] if isinstance(c, Tag) and '$' in c.get_text()]

                            if len(pricing_cells) >= 2:
                                # Likely input and output pricing
                                input_price_text = pricing_cells[0].get_text().strip()
                                output_price_text = pricing_cells[1].get_text().strip()
                            elif len(pricing_cells) == 1:
                                # Might be combined pricing or just one price
                                input_price_text = pricing_cells[0].get_text().strip()
                                output_price_text = input_price_text
                            else:
                                continue

                            # Extract numeric values using regex
                            input_price = self._extract_price(input_price_text)
                            output_price = self._extract_price(output_price_text)

                            if input_price is not None and output_price is not None:
                                pricing_data[model_name] = {
                                    "input_cost_per_1k_tokens": input_price,
                                    "output_cost_per_1k_tokens": output_price
                                }
                                logger.info(f"Found pricing for {model_name}: input=${input_price}, output=${output_price} per 1K tokens")
                        except Exception as e:
                            logger.warning(f"Error extracting pricing for {model_name}: {e}")

                # Add fallback pricing for common models if not found
                self._add_fallback_pricing(pricing_data)

                self.pricing_data = pricing_data
                return pricing_data

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.warning("Received 403 Forbidden when accessing OpenAI pricing page. Using fallback pricing data.")
            else:
                logger.error(f"Error fetching pricing data: {e}")

            # Return comprehensive fallback pricing data
            fallback_pricing = self._get_comprehensive_fallback_pricing()
            self.pricing_data = fallback_pricing
            return fallback_pricing
        except Exception as e:
            logger.error(f"Error fetching pricing data: {e}")
            # Return fallback pricing data
            fallback_pricing = self._get_comprehensive_fallback_pricing()
            self.pricing_data = fallback_pricing
            return fallback_pricing

    def _get_comprehensive_fallback_pricing(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive fallback pricing data for all known OpenAI models."""
        logger.info("Using comprehensive fallback pricing data")
        return {
            # GPT-4 models
            "gpt-4": {"input_cost_per_1k_tokens": 0.03, "output_cost_per_1k_tokens": 0.06},
            "gpt-4-0613": {"input_cost_per_1k_tokens": 0.03, "output_cost_per_1k_tokens": 0.06},
            "gpt-4-32k": {"input_cost_per_1k_tokens": 0.06, "output_cost_per_1k_tokens": 0.12},
            "gpt-4-32k-0613": {"input_cost_per_1k_tokens": 0.06, "output_cost_per_1k_tokens": 0.12},
            "gpt-4-turbo": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4-turbo-preview": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4-0125-preview": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4-1106-preview": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4-turbo-2024-04-09": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4-vision-preview": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},

            # GPT-4o models
            "gpt-4o": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4o-2024-05-13": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4o-2024-08-06": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4o-2024-11-20": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "chatgpt-4o-latest": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4o-mini": {"input_cost_per_1k_tokens": 0.005, "output_cost_per_1k_tokens": 0.015},
            "gpt-4o-mini-2024-07-18": {"input_cost_per_1k_tokens": 0.005, "output_cost_per_1k_tokens": 0.015},

            # GPT-4.5 models
            "gpt-4.5-preview": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4.5-preview-2025-02-27": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},

            # GPT-3.5 models
            "gpt-3.5-turbo": {"input_cost_per_1k_tokens": 0.0015, "output_cost_per_1k_tokens": 0.002},
            "gpt-3.5-turbo-0125": {"input_cost_per_1k_tokens": 0.0015, "output_cost_per_1k_tokens": 0.002},
            "gpt-3.5-turbo-1106": {"input_cost_per_1k_tokens": 0.0015, "output_cost_per_1k_tokens": 0.002},
            "gpt-3.5-turbo-16k": {"input_cost_per_1k_tokens": 0.003, "output_cost_per_1k_tokens": 0.004},
            "gpt-3.5-turbo-16k-0613": {"input_cost_per_1k_tokens": 0.003, "output_cost_per_1k_tokens": 0.004},
            "gpt-3.5-turbo-instruct": {"input_cost_per_1k_tokens": 0.0015, "output_cost_per_1k_tokens": 0.002},
            "gpt-3.5-turbo-instruct-0914": {"input_cost_per_1k_tokens": 0.0015, "output_cost_per_1k_tokens": 0.002},

            # O1 models
            "o1": {"input_cost_per_1k_tokens": 0.015, "output_cost_per_1k_tokens": 0.075},
            "o1-preview": {"input_cost_per_1k_tokens": 0.015, "output_cost_per_1k_tokens": 0.075},
            "o1-mini": {"input_cost_per_1k_tokens": 0.005, "output_cost_per_1k_tokens": 0.025},
            "o1-2024-12-17": {"input_cost_per_1k_tokens": 0.015, "output_cost_per_1k_tokens": 0.075},
            "o1-preview-2024-09-12": {"input_cost_per_1k_tokens": 0.015, "output_cost_per_1k_tokens": 0.075},
            "o1-mini-2024-09-12": {"input_cost_per_1k_tokens": 0.005, "output_cost_per_1k_tokens": 0.025},

            # O3 models
            "o3-mini": {"input_cost_per_1k_tokens": 0.003, "output_cost_per_1k_tokens": 0.015},
            "o3-mini-2025-01-31": {"input_cost_per_1k_tokens": 0.003, "output_cost_per_1k_tokens": 0.015},

            # Audio models
            "gpt-4o-audio-preview": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4o-audio-preview-2024-10-01": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4o-audio-preview-2024-12-17": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4o-mini-audio-preview": {"input_cost_per_1k_tokens": 0.005, "output_cost_per_1k_tokens": 0.015},
            "gpt-4o-mini-audio-preview-2024-12-17": {"input_cost_per_1k_tokens": 0.005, "output_cost_per_1k_tokens": 0.015},

            # Realtime models
            "gpt-4o-realtime-preview": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4o-realtime-preview-2024-10-01": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4o-realtime-preview-2024-12-17": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4o-mini-realtime-preview": {"input_cost_per_1k_tokens": 0.005, "output_cost_per_1k_tokens": 0.015},
            "gpt-4o-mini-realtime-preview-2024-12-17": {"input_cost_per_1k_tokens": 0.005, "output_cost_per_1k_tokens": 0.015},
        }

    def _get_fallback_pricing(self) -> Dict[str, Dict[str, float]]:
        """Get fallback pricing data for common models."""
        logger.info("Using basic fallback pricing data")
        return {
            "gpt-4": {"input_cost_per_1k_tokens": 0.03, "output_cost_per_1k_tokens": 0.06},
            "gpt-4-turbo": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4o": {"input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03},
            "gpt-4o-mini": {"input_cost_per_1k_tokens": 0.005, "output_cost_per_1k_tokens": 0.015},
            "gpt-3.5-turbo": {"input_cost_per_1k_tokens": 0.0015, "output_cost_per_1k_tokens": 0.002},
            "gpt-3.5-turbo-instruct": {"input_cost_per_1k_tokens": 0.0015, "output_cost_per_1k_tokens": 0.002},
        }

    def _normalize_model_name(self, display_name: str) -> str:
        """Normalize display names from the pricing page to match API model names."""
        # Common mappings between display names and API names
        mappings = {
            "gpt-4": "gpt-4",
            "gpt-4 turbo": "gpt-4-turbo",
            "gpt-4o": "gpt-4o",
            "gpt-4o mini": "gpt-4o-mini",
            "gpt-3.5 turbo": "gpt-3.5-turbo",
            "gpt-3.5 turbo instruct": "gpt-3.5-turbo-instruct",
            "claude instant": "claude-instant",
            "claude 3 opus": "claude-3-opus",
            "claude 3 sonnet": "claude-3-sonnet",
            "claude 3 haiku": "claude-3-haiku",
        }

        # Try direct mapping first
        clean_name = display_name.lower().strip()
        if clean_name in mappings:
            return mappings[clean_name]

        # Try partial matching
        for display, api_name in mappings.items():
            if display in clean_name:
                return api_name

        # Try to normalize the name by replacing spaces with dashes
        normalized = clean_name.replace(" ", "-")

        # Remove any text in parentheses
        normalized = re.sub(r'\([^)]*\)', '', normalized).strip()

        return normalized

    def _extract_price(self, price_text: str) -> Optional[float]:
        """Extract price value from a text string."""
        # Look for patterns like $0.01, $0.001, etc.
        match = re.search(r'\$(\d+\.\d+)', price_text)
        if match:
            return float(match.group(1))
        return None

    def _add_fallback_pricing(self, pricing_data: Dict[str, Dict[str, float]]) -> None:
        """Add fallback pricing for common models if not found in the scraped data."""
        fallback_pricing = self._get_fallback_pricing()

        for model, prices in fallback_pricing.items():
            if model not in pricing_data:
                pricing_data[model] = prices

    def _find_best_pricing_match(self, model_id: str) -> Tuple[float, float]:
        """Find the best pricing match for a model ID."""
        # Try exact match first
        if model_id in self.pricing_data:
            pricing = self.pricing_data[model_id]
            return (pricing["input_cost_per_1k_tokens"], pricing["output_cost_per_1k_tokens"])

        # Try prefix matching
        for pricing_model, pricing in self.pricing_data.items():
            if model_id.startswith(pricing_model):
                return (pricing["input_cost_per_1k_tokens"], pricing["output_cost_per_1k_tokens"])

        # Use fallback pricing based on model family
        if "gpt-4" in model_id:
            if "mini" in model_id:
                return (0.005, 0.015)  # gpt-4o-mini pricing
            elif "o" in model_id:
                return (0.01, 0.03)    # gpt-4o pricing
            else:
                return (0.03, 0.06)    # gpt-4 pricing
        elif "gpt-3.5" in model_id:
            return (0.0015, 0.002)     # gpt-3.5-turbo pricing

        # Default fallback
        logger.warning(f"No pricing match found for {model_id}, using default pricing")
        return (0.01, 0.03)  # Default to gpt-4o pricing

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
        input_cost, output_cost = self._find_best_pricing_match(model_id)

        capabilities = {
            "max_context_window": await self._test_context_window(model_id),
            "supports_streaming": await self._test_streaming(model_id),
            "supports_function_calling": await self._test_function_calling(model_id, token_usage),
            "supports_vision": await self._test_vision(model_id, token_usage),
            "supports_json_mode": await self._test_json_mode(model_id, token_usage),
            "supports_system_prompt": await self._test_system_prompt(model_id, token_usage),
            "supports_tools": await self._test_tools(model_id, token_usage),
            "supports_frequency_penalty": await self._test_frequency_penalty(model_id, token_usage),
            "supports_presence_penalty": await self._test_presence_penalty(model_id, token_usage),
            "supports_stop_sequences": await self._test_stop_sequences(model_id, token_usage),
            "supports_direct_pydantic_parse": await self._test_direct_pydantic_parse(model_id, token_usage),
            "supports_temperature": await self._test_temperature(model_id, token_usage),
            "supports_message_role": True,  # All OpenAI chat models support message roles
            "typical_speed": None,  # This would require more extensive testing
            "supported_languages": {"en"},  # Default to English, would need more testing for others
            "input_cost_per_1k_tokens": input_cost,
            "output_cost_per_1k_tokens": output_cost,
        }

        # Test vision capabilities if the model supports vision
        vision_capabilities = None
        if capabilities["supports_vision"]:
            vision_capabilities = await self._test_vision_capabilities(model_id, token_usage)

        # Calculate total cost
        total_cost = (token_usage["input_tokens"] / 1000 * input_cost) + (token_usage["output_tokens"] / 1000 * output_cost)
        token_usage["total_cost"] = total_cost

        logger.info(f"Token usage for {model_id}: {token_usage['input_tokens']} input tokens, {token_usage['output_tokens']} output tokens")
        logger.info(f"Estimated cost for {model_id}: ${total_cost:.4f}")

        # Store the results
        self.capability_results[model_id] = {
            "capabilities": capabilities,
            "vision_capabilities": vision_capabilities,
            "created": None,  # This would require more information to populate
            "owned_by": None,  # This would require more information to populate
            "token_usage": token_usage
        }

        return self.capability_results[model_id]

    async def _test_basic_chat(self, model_id: str) -> bool:
        """Test if a model supports basic chat completion."""
        try:
            # Create base kwargs with proper type annotation
            kwargs: Dict[str, Any] = {
                "messages": BASIC_TEST_MESSAGE,
            }

            # Add the appropriate tokens parameter based on the model
            if self.requires_completion_tokens(model_id):
                kwargs["max_completion_tokens"] = 10
            else:
                kwargs["max_tokens"] = 10

            # Make the API call
            response = await self.client.chat.completions.create(model=model_id, **kwargs)
            return True
        except Exception as e:
            logger.warning(f"Model {model_id} does not support basic chat: {e}")
            return False

    async def _test_context_window(self, model_id: str) -> int:
        """Estimate the context window size for a model."""
        # This is a simplified approach - in reality, you'd need to check the model specs
        # or test with increasingly large contexts
        context_windows = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }

        # Check for exact matches
        if model_id in context_windows:
            return context_windows[model_id]

        # Check for prefix matches
        for prefix, size in context_windows.items():
            if model_id.startswith(prefix):
                return size

        # Default value if unknown
        return 4096

    async def _test_streaming(self, model_id: str) -> bool:
        """Test if a model supports streaming."""
        try:
            # Create base kwargs with proper type annotation
            kwargs: Dict[str, Any] = {
                "messages": BASIC_TEST_MESSAGE,
                "stream": True
            }

            # Add the appropriate tokens parameter based on the model
            if self.requires_completion_tokens(model_id):
                kwargs["max_completion_tokens"] = 10
            else:
                kwargs["max_tokens"] = 10

            # Make the API call
            response = await self.client.chat.completions.create(model=model_id, **kwargs)
            # Just check if we can get the first chunk
            async for chunk in response:
                break
            return True
        except Exception as e:
            logger.warning(f"Model {model_id} does not support streaming: {e}")
            return False

    async def _test_function_calling(self, model_id: str, token_usage: Dict[str, float]) -> bool:
        """Test if a model supports function calling."""
        try:
            # Create base kwargs with proper type annotation
            kwargs: Dict[str, Any] = {
                "messages": FUNCTION_CALLING_TEST_MESSAGE,
                "functions": [{
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }],
                "function_call": "auto"
            }

            # Add the appropriate tokens parameter based on the model
            if self.requires_completion_tokens(model_id):
                kwargs["max_completion_tokens"] = 50
            else:
                kwargs["max_tokens"] = 50

            # Make the API call
            response = await self.client.chat.completions.create(model=model_id, **kwargs)
            # Track token usage from the API response
            if hasattr(response, 'usage') and response.usage:
                token_usage["input_tokens"] += response.usage.prompt_tokens
                token_usage["output_tokens"] += response.usage.completion_tokens
            return response.choices[0].message.function_call is not None
        except Exception as e:
            logger.warning(f"Model {model_id} does not support function calling: {e}")
            return False

    async def _test_vision(self, model_id: str, token_usage: Dict[str, float]) -> bool:
        """Test if a model supports vision."""
        try:
            # Create base kwargs with proper type annotation
            kwargs: Dict[str, Any] = {
                "messages": VISION_TEST_MESSAGE,
            }

            # Add the appropriate tokens parameter based on the model
            if self.requires_completion_tokens(model_id):
                kwargs["max_completion_tokens"] = 50
            else:
                kwargs["max_tokens"] = 50

            # Make the API call
            response = await self.client.chat.completions.create(model=model_id, **kwargs)
            # Track token usage from the API response
            if hasattr(response, 'usage') and response.usage:
                token_usage["input_tokens"] += response.usage.prompt_tokens
                token_usage["output_tokens"] += response.usage.completion_tokens
            return True
        except Exception as e:
            logger.warning(f"Model {model_id} does not support vision: {e}")
            return False

    async def _test_json_mode(self, model_id: str, token_usage: Dict[str, float]) -> bool:
        """Test if a model supports JSON mode."""
        try:
            # Create base kwargs with proper type annotation
            kwargs: Dict[str, Any] = {
                "messages": JSON_MODE_TEST_MESSAGE,
                "response_format": {"type": "json_object"}
            }

            # Add the appropriate tokens parameter based on the model
            if self.requires_completion_tokens(model_id):
                kwargs["max_completion_tokens"] = 50
            else:
                kwargs["max_tokens"] = 50

            # Make the API call
            response = await self.client.chat.completions.create(model=model_id, **kwargs)
            # Track token usage from the API response
            if hasattr(response, 'usage') and response.usage:
                token_usage["input_tokens"] += response.usage.prompt_tokens
                token_usage["output_tokens"] += response.usage.completion_tokens
            content = response.choices[0].message.content
            if content is not None:
                json.loads(content)
            return True
        except Exception as e:
            logger.warning(f"Model {model_id} does not support JSON mode: {e}")
            return False

    async def _test_system_prompt(self, model_id: str, token_usage: Dict[str, float]) -> bool:
        """Test if a model supports system prompts."""
        try:
            # Create base kwargs with proper type annotation
            kwargs: Dict[str, Any] = {
                "messages": SYSTEM_PROMPT_TEST_MESSAGE,
            }

            # Add the appropriate tokens parameter based on the model
            if self.requires_completion_tokens(model_id):
                kwargs["max_completion_tokens"] = 50
            else:
                kwargs["max_tokens"] = 50

            # Make the API call
            response = await self.client.chat.completions.create(model=model_id, **kwargs)
            # Track token usage from the API response
            if hasattr(response, 'usage') and response.usage:
                token_usage["input_tokens"] += response.usage.prompt_tokens
                token_usage["output_tokens"] += response.usage.completion_tokens
            return True
        except Exception as e:
            logger.warning(f"Model {model_id} does not support system prompts: {e}")
            return False

    async def _test_tools(self, model_id: str, token_usage: Dict[str, float]) -> bool:
        """Test if a model supports tools."""
        try:
            # Create base kwargs with proper type annotation
            kwargs: Dict[str, Any] = {
                "messages": TOOLS_TEST_MESSAGE,
                "tools": TOOLS_CONFIG,
                "tool_choice": "auto"
            }

            # Add the appropriate tokens parameter based on the model
            if self.requires_completion_tokens(model_id):
                kwargs["max_completion_tokens"] = 50
            else:
                kwargs["max_tokens"] = 50

            # Make the API call
            response = await self.client.chat.completions.create(model=model_id, **kwargs)
            # Track token usage from the API response
            if hasattr(response, 'usage') and response.usage:
                token_usage["input_tokens"] += response.usage.prompt_tokens
                token_usage["output_tokens"] += response.usage.completion_tokens
            return response.choices[0].message.tool_calls is not None
        except Exception as e:
            logger.warning(f"Model {model_id} does not support tools: {e}")
            return False

    async def _test_frequency_penalty(self, model_id: str, token_usage: Dict[str, float]) -> bool:
        """Test if a model supports frequency penalty."""
        try:
            # Create base kwargs with proper type annotation
            kwargs: Dict[str, Any] = {
                "messages": BASIC_TEST_MESSAGE,
                "frequency_penalty": 0.5
            }

            # Add the appropriate tokens parameter based on the model
            if self.requires_completion_tokens(model_id):
                kwargs["max_completion_tokens"] = 50
            else:
                kwargs["max_tokens"] = 50

            # Make the API call
            response = await self.client.chat.completions.create(model=model_id, **kwargs)
            # Track token usage from the API response
            if hasattr(response, 'usage') and response.usage:
                token_usage["input_tokens"] += response.usage.prompt_tokens
                token_usage["output_tokens"] += response.usage.completion_tokens
            return True
        except Exception as e:
            logger.warning(f"Model {model_id} does not support frequency penalty: {e}")
            return False

    async def _test_presence_penalty(self, model_id: str, token_usage: Dict[str, float]) -> bool:
        """Test if a model supports presence penalty."""
        try:
            # Create base kwargs with proper type annotation
            kwargs: Dict[str, Any] = {
                "messages": BASIC_TEST_MESSAGE,
                "presence_penalty": 0.5
            }

            # Add the appropriate tokens parameter based on the model
            if self.requires_completion_tokens(model_id):
                kwargs["max_completion_tokens"] = 50
            else:
                kwargs["max_tokens"] = 50

            # Make the API call
            response = await self.client.chat.completions.create(model=model_id, **kwargs)
            # Track token usage from the API response
            if hasattr(response, 'usage') and response.usage:
                token_usage["input_tokens"] += response.usage.prompt_tokens
                token_usage["output_tokens"] += response.usage.completion_tokens
            return True
        except Exception as e:
            logger.warning(f"Model {model_id} does not support presence penalty: {e}")
            return False

    async def _test_stop_sequences(self, model_id: str, token_usage: Dict[str, float]) -> bool:
        """Test if a model supports stop sequences."""
        try:
            # Create base kwargs with proper type annotation
            kwargs: Dict[str, Any] = {
                "messages": BASIC_TEST_MESSAGE,
                "stop": [".", "!"]
            }

            # Add the appropriate tokens parameter based on the model
            if self.requires_completion_tokens(model_id):
                kwargs["max_completion_tokens"] = 50
            else:
                kwargs["max_tokens"] = 50

            # Make the API call
            response = await self.client.chat.completions.create(model=model_id, **kwargs)
            # Track token usage from the API response
            if hasattr(response, 'usage') and response.usage:
                token_usage["input_tokens"] += response.usage.prompt_tokens
                token_usage["output_tokens"] += response.usage.completion_tokens
            return True
        except Exception as e:
            logger.warning(f"Model {model_id} does not support stop sequences: {e}")
            return False

    async def _test_direct_pydantic_parse(self, model_id: str, token_usage: Dict[str, float]) -> bool:
        """Test if a model supports direct Pydantic parsing.

        This tests whether the model can reliably generate output that can be directly parsed
        into a Pydantic model without additional processing.
        """
        # Define a simple Pydantic model for testing
        class TestModel(BaseModel):
            name: str
            age: int
            is_active: bool

        # Test message that asks for structured data
        PYDANTIC_TEST_MESSAGE = [
            {"role": "user", "content": "Please provide information about a person with the following information: name (string), age (integer), and is_active (boolean)."}
        ]

        try:
            # Create base kwargs with proper type annotation
            kwargs: Dict[str, Any] = {
                "messages": PYDANTIC_TEST_MESSAGE,
            }

            # Add the appropriate tokens parameter based on the model
            if self.requires_completion_tokens(model_id):
                kwargs["max_completion_tokens"] = 50
            else:
                kwargs["max_tokens"] = 50

            # Try to use the beta.chat.completions.parse endpoint with the Pydantic model
            try:
                # Make the API call with direct Pydantic parsing
                response = await self.client.beta.chat.completions.parse(
                    model=model_id,
                    response_format=TestModel,
                    **kwargs
                )

                # If we get here, the model supports direct Pydantic parsing
                # Track token usage from the API response
                if hasattr(response, 'usage') and response.usage:
                    token_usage["input_tokens"] += response.usage.prompt_tokens
                    token_usage["output_tokens"] += response.usage.completion_tokens

                return True
            except (AttributeError, NotImplementedError):
                # The beta.chat.completions.parse endpoint might not be available
                # Fall back to the standard JSON object approach
                logger.warning(f"Direct Pydantic parsing not available for model {model_id}, falling back to JSON mode")
                return False
            except Exception as e:
                # Any other exception indicates the model doesn't support direct Pydantic parsing
                logger.warning(f"Model {model_id} does not support direct Pydantic parsing: {e}")
                return False

        except Exception as e:
            logger.warning(f"Error testing direct Pydantic parsing for model {model_id}: {e}")
            return False

    async def _test_vision_capabilities(self, model_id: str, token_usage: Dict[str, float]) -> Dict[str, Any]:
        """Test vision-specific capabilities for a model."""
        try:
            # Test with different image formats
            supported_formats = {"jpeg", "png", "webp"}

            # Use more reliable image URLs
            # Test with a high-resolution image to check limits
            # Create base kwargs for high-res image test with proper type annotation
            high_res_kwargs: Dict[str, Any] = {
                "messages": [
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=[
                            ChatCompletionContentPartTextParam(
                                type="text",
                                text="Describe this high-resolution image in detail."
                            ),
                            ChatCompletionContentPartImageParam(
                                type="image_url",
                                image_url={
                                    "url": "https://images.unsplash.com/photo-1575936123452-b67c3203c357?q=80&w=2070&auto=format&fit=crop"
                                }
                            )
                        ]
                    )
                ],
            }

            # Add the appropriate tokens parameter based on the model
            if self.requires_completion_tokens(model_id):
                high_res_kwargs["max_completion_tokens"] = 100
            else:
                high_res_kwargs["max_tokens"] = 100

            # Make the API call for high-res image
            high_res_image_response = await self.client.chat.completions.create(model=model_id, **high_res_kwargs)

            # Track token usage from the API response
            if hasattr(high_res_image_response, 'usage') and high_res_image_response.usage:
                token_usage["input_tokens"] += high_res_image_response.usage.prompt_tokens
                token_usage["output_tokens"] += high_res_image_response.usage.completion_tokens

            # Test with multiple images
            # Create base kwargs for multi-image test with proper type annotation
            multi_image_kwargs: Dict[str, Any] = {
                "messages": [
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=[
                            ChatCompletionContentPartTextParam(
                                type="text",
                                text="Compare these two images."
                            ),
                            ChatCompletionContentPartImageParam(
                                type="image_url",
                                image_url={
                                    "url": "https://images.unsplash.com/photo-1579762715118-a6f1d4b934f1?q=80&w=2071&auto=format&fit=crop"
                                }
                            ),
                            ChatCompletionContentPartImageParam(
                                type="image_url",
                                image_url={
                                    "url": "https://images.unsplash.com/photo-1561948955-570b270e7c36?q=80&w=1000&auto=format&fit=crop"
                                }
                            )
                        ]
                    )
                ],
            }

            # Add the appropriate tokens parameter based on the model
            if self.requires_completion_tokens(model_id):
                multi_image_kwargs["max_completion_tokens"] = 100
            else:
                multi_image_kwargs["max_tokens"] = 100

            # Make the API call for multi-image test
            multi_image_response = await self.client.chat.completions.create(model=model_id, **multi_image_kwargs)

            # Track token usage from the API response
            if hasattr(multi_image_response, 'usage') and multi_image_response.usage:
                token_usage["input_tokens"] += multi_image_response.usage.prompt_tokens
                token_usage["output_tokens"] += multi_image_response.usage.completion_tokens

            # Determine max images per request based on successful test
            max_images_per_request = 2  # Default based on our test

            return {
                "max_images_per_request": max_images_per_request,
                "supported_formats": list(supported_formats),
                "max_image_size_mb": 20,  # Changed from 20.0 to 20 to match int type
                "max_image_resolution": 2048,  # Default for most models
                "supports_image_annotations": False,  # Would need more specific testing
                "supports_image_analysis": True,  # If we got here, it supports basic analysis
                "supports_image_generation": False,  # OpenAI separates generation (DALL-E) from vision
                "cost_per_image": 0.002,  # Default estimate, would need more precise testing
            }
        except Exception as e:
            logger.warning(f"Error testing vision capabilities for {model_id}: {e}")
            # Return default values if testing fails
            return {
                "max_images_per_request": 1,
                "supported_formats": ["jpeg", "png"],
                "max_image_size_mb": 20,  # Changed from 20.0 to 20 to match int type
                "max_image_resolution": 2048,
                "supports_image_annotations": False,
                "supports_image_analysis": True,
                "supports_image_generation": False,
                "cost_per_image": 0.002,
            }

    async def _test_temperature(self, model_id: str, token_usage: Dict[str, float]) -> bool:
        """Test if a model supports temperature parameter.

        Args:
            model_id: ID of the model to test
            token_usage: Dictionary to track token usage

        Returns:
            True if the model supports temperature, False otherwise
        """
        try:
            # Create base kwargs with proper type annotation
            kwargs: Dict[str, Any] = {
                "messages": BASIC_TEST_MESSAGE,
                "temperature": 0.5  # Use a non-default temperature value
            }

            # Add the appropriate tokens parameter based on the model
            if self.requires_completion_tokens(model_id):
                kwargs["max_completion_tokens"] = 50
            else:
                kwargs["max_tokens"] = 50

            # Make the API call
            response = await self.client.chat.completions.create(model=model_id, **kwargs)
            # Track token usage from the API response
            if hasattr(response, 'usage') and response.usage:
                token_usage["input_tokens"] += response.usage.prompt_tokens
                token_usage["output_tokens"] += response.usage.completion_tokens
            return True
        except Exception as e:
            logger.warning(f"Model {model_id} does not support temperature: {e}")
            return False

    def generate_model_code(self, model_id: str) -> str:
        """Generate Python code for a model definition."""
        if model_id not in self.capability_results:
            raise ValueError(f"No test results for model {model_id}")

        results = self.capability_results[model_id]
        capabilities = results["capabilities"]
        vision_capabilities = results["vision_capabilities"]

        # Format the model name for the method name
        method_name = model_id.replace("-", "_").replace(".", "_")

        # Check if the model is a preview model
        is_preview = "preview" in model_id.lower()

        # Check if the model might be deprecated (older dated versions)
        # This is a simple heuristic - a more comprehensive approach would check against a list of known deprecated models
        is_deprecated = False
        if re.search(r'-\d{4}-\d{2}-\d{2}', model_id) or re.search(r'-\d{4}', model_id):
            # Check if this is an older dated version by comparing with current date
            model_date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', model_id)
            if model_date_match:
                model_date = datetime(int(model_date_match.group(1)), int(model_date_match.group(2)), int(model_date_match.group(3)))
                six_months_ago = datetime.now() - timedelta(days=180)
                if model_date < six_months_ago:
                    is_deprecated = True
            else:
                # Check for YYMM format (e.g., 0613)
                model_date_match = re.search(r'(\d{2})(\d{2})', model_id)
                if model_date_match:
                    # Assume 20xx for the year
                    month = int(model_date_match.group(2))
                    year = 2000 + int(model_date_match.group(1))
                    if month > 0 and month <= 12:
                        model_date = datetime(year, month, 1)
                        six_months_ago = datetime.now() - timedelta(days=180)
                        if model_date < six_months_ago:
                            is_deprecated = True

        # Generate a timestamp for the release date
        # For simplicity, we'll use the current timestamp
        release_timestamp = int(datetime.now().timestamp())

        # Configure runtime config based on capabilities
        # Only include temperature if the model supports it
        supports_temperature = capabilities.get('supports_temperature', False)
        if supports_temperature:
            runtime_config = f"max_tokens=1024, temperature=0.7, max_context_tokens={capabilities['max_context_window']}, stream=True"
        else:
            runtime_config = f"max_tokens=1024, max_context_tokens={capabilities['max_context_window']}, stream=True"
            logger.warning(f"Model {model_id} does not support temperature, excluding from runtime config")

        # Generate the method code
        method_code = f"""
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
                    max_context_window={capabilities['max_context_window']},
                    max_output_tokens={capabilities.get('max_output_tokens', 4096)},
                    supports_streaming={str(capabilities['supports_streaming'])},
                    supports_function_calling={str(capabilities['supports_function_calling'])},
                    supports_vision={str(capabilities['supports_vision'])},
                    supports_embeddings=False,
                    supports_json_mode={str(capabilities['supports_json_mode'])},
                    supports_system_prompt={str(capabilities['supports_system_prompt'])},
                    supports_tools={str(capabilities['supports_tools'])},
                    supports_parallel_requests=True,
                    supports_frequency_penalty={str(capabilities['supports_frequency_penalty'])},
                    supports_presence_penalty={str(capabilities['supports_presence_penalty'])},
                    supports_stop_sequences={str(capabilities['supports_stop_sequences'])},
                    supports_direct_pydantic_parse={str(capabilities['supports_direct_pydantic_parse'])},
                    supports_temperature={str(supports_temperature)},
                    supports_message_role=True,
                    typical_speed=None,
                    supported_languages={{"en"}},
                    input_cost_per_1k_tokens={capabilities['input_cost_per_1k_tokens']},
                    output_cost_per_1k_tokens={capabilities['output_cost_per_1k_tokens']},
                ),
                metadata=LLMMetadata(
                    release_date=datetime.fromtimestamp({release_timestamp}),
                    is_preview={str(is_preview)},
                    is_deprecated={str(is_deprecated)},
                    min_api_version="2023-05-15",
                ),"""

        # Add vision capabilities if the model supports vision
        if capabilities['supports_vision']:
            method_code += f"""
                vision_capabilities=VisionCapabilities(
                    max_images_per_request={vision_capabilities['max_images_per_request']},
                    supported_formats={vision_capabilities['supported_formats']},
                    max_image_size_mb={vision_capabilities['max_image_size_mb']},
                    max_image_resolution={vision_capabilities['max_image_resolution']},
                    supports_image_annotations={str(vision_capabilities['supports_image_annotations'])},
                    supports_image_analysis={str(vision_capabilities['supports_image_analysis'])},
                    supports_image_generation={str(vision_capabilities['supports_image_generation'])},
                    cost_per_image={vision_capabilities['cost_per_image']},
                ),"""

        # Close the method code
        method_code += f"""
            ),
            provider=PROVIDER,
            runtime_config=LLMRuntimeConfig({runtime_config}),
        )"""

        return method_code

    def update_models_file(self, models_to_add: List[str]) -> None:
        """Update the models.py file with new model definitions."""
        try:
            # Read the current file
            models_file_path = "src/llmaestro/default_library/defined_providers/openai/models.py"
            with open(models_file_path, "r") as f:
                content = f.read()

            # Generate code for each model
            new_methods = []

            for model_id in models_to_add:
                # Skip deprecated models unless explicitly included
                if not self.include_deprecated and self.is_model_deprecated(model_id):
                    logger.info(f"Skipping deprecated model: {model_id}")
                    continue

                # Skip preview models unless explicitly included
                if not self.include_preview and self.is_model_preview(model_id):
                    logger.info(f"Skipping preview model: {model_id}")
                    continue

                # Generate the method code
                method_code = self.generate_model_code(model_id)
                new_methods.append(method_code)

            # Find the end of the class definition to append new methods
            class_pattern = r"class\s+OpenAIModels\s*:"
            class_match = re.search(class_pattern, content)

            if not class_match:
                logger.error("Could not find OpenAIModels class in the file")
                return

            # Find the get_model method to insert new methods before it
            get_model_pattern = r"@classmethod\s+def\s+get_model\s*\("
            get_model_match = re.search(get_model_pattern, content)

            updated_content = content

            if get_model_match:
                # Insert new methods before the get_model method
                insert_position = get_model_match.start()

                # Add each new method
                for method in new_methods:
                    # Check if the method already exists
                    method_name_match = re.search(r"def\s+(\w+)\(\)", method)
                    if method_name_match:
                        method_name = method_name_match.group(1)
                        method_pattern = rf"def\s+{method_name}\(\)"

                        if not re.search(method_pattern, content):
                            # Insert the new method
                            updated_content = updated_content[:insert_position] + method + "\n\n" + updated_content[insert_position:]
            else:
                # If get_model method not found, append at the end of the file
                updated_content += "\n\n"
                for method in new_methods:
                    method_name_match = re.search(r"def\s+(\w+)\(\)", method)
                    if method_name_match:
                        method_name = method_name_match.group(1)
                        method_pattern = rf"def\s+{method_name}\(\)"

                        if not re.search(method_pattern, content):
                            updated_content += method + "\n\n"

            # Write the updated content back to the file
            with open(models_file_path, "w") as f:
                f.write(updated_content)

            logger.info(f"Successfully updated models.py with {len(models_to_add)} new models")
        except Exception as e:
            logger.error(f"Error updating models.py: {e}")

    async def update_pricing_in_file(self, pricing_data: Dict[str, Dict[str, float]]) -> None:
        """Update pricing information in the models.py file."""
        try:
            # Read the current file
            models_file_path = "src/llmaestro/default_library/defined_providers/openai/models.py"
            with open(models_file_path, "r") as f:
                content = f.read()

            # For each model in the pricing data
            for model_id, prices in pricing_data.items():
                # Skip updating deprecated or preview models if not included
                if not self.include_deprecated and self.is_model_deprecated(model_id):
                    logger.info(f"Skipping update for deprecated model: {model_id}")
                    continue

                if not self.include_preview and self.is_model_preview(model_id):
                    logger.info(f"Skipping update for preview model: {model_id}")
                    continue

                # Format the model name for the method name
                method_name = model_id.replace("-", "_").replace(".", "_")

                # Search for the input_cost_per_1k_tokens parameter
                input_cost_pattern = re.compile(rf"def\s+{method_name}\(\).*?input_cost_per_1k_tokens=([0-9.]+)", re.DOTALL)
                if input_cost_match := input_cost_pattern.search(content):
                    old_input_cost = input_cost_match.group(1)
                    new_input_cost = str(prices.get('input_cost_per_1k_tokens', old_input_cost))
                    content = content.replace(f"input_cost_per_1k_tokens={old_input_cost}", f"input_cost_per_1k_tokens={new_input_cost}")
                    logger.info(f"Updated input cost for {model_id}: {old_input_cost} -> {new_input_cost}")

                # Search for the output_cost_per_1k_tokens parameter
                output_cost_pattern = re.compile(rf"def\s+{method_name}\(\).*?output_cost_per_1k_tokens=([0-9.]+)", re.DOTALL)
                if output_cost_match := output_cost_pattern.search(content):
                    old_output_cost = output_cost_match.group(1)
                    new_output_cost = str(prices.get('output_cost_per_1k_tokens', old_output_cost))
                    content = content.replace(f"output_cost_per_1k_tokens={old_output_cost}", f"output_cost_per_1k_tokens={new_output_cost}")
                    logger.info(f"Updated output cost for {model_id}: {old_output_cost} -> {new_output_cost}")

            # Write the updated content back to the file
            with open(models_file_path, "w") as f:
                f.write(content)

            logger.info(f"Successfully updated pricing in models.py")
        except Exception as e:
            logger.error(f"Error updating pricing in models.py: {e}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test OpenAI model capabilities and update models.py")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--models", nargs="+", help="Specific models to test (default: all chat models)")
    parser.add_argument("--update-file", action="store_true", help="Update the models.py file with test results")
    parser.add_argument("--update-pricing-only", action="store_true", help="Only update pricing information without testing capabilities")
    parser.add_argument("--include-deprecated", action="store_true", help="Include deprecated models (default: skip deprecated)")
    parser.add_argument("--include-preview", action="store_true", help="Include preview models (default: skip preview)")
    parser.add_argument("--include-non-openai-owned", action="store_true", help="Include models not owned by OpenAI or system (default: only OpenAI and system models)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run the tests asynchronously
    if args.update_pricing_only:
        asyncio.run(update_pricing_only(
            api_key=args.api_key,
            update_file=args.update_file,
            include_deprecated=args.include_deprecated,
            include_preview=args.include_preview,
            include_non_openai_owned=args.include_non_openai_owned
        ))
    else:
        asyncio.run(run_tests(
            api_key=args.api_key,
            models_to_test=args.models,
            update_file=args.update_file,
            include_deprecated=args.include_deprecated,
            include_preview=args.include_preview,
            include_non_openai_owned=args.include_non_openai_owned
        ))

async def run_tests(
    api_key: str,
    models_to_test: Optional[List[str]] = None,
    update_file: bool = False,
    include_deprecated: bool = False,
    include_preview: bool = False,
    include_non_openai_owned: bool = False
) -> None:
    """Run capability tests for OpenAI models."""
    tester = ModelCapabilityTester(api_key)
    tester.include_deprecated = include_deprecated
    tester.include_preview = include_preview
    tester.include_non_openai_owned = include_non_openai_owned

    try:
        # Get all available models
        all_models = await tester.get_models()

        # Filter models if specific ones were requested
        if models_to_test:
            models = [model for model in all_models if model.id in models_to_test]
            if not models:
                logger.warning(f"None of the specified models {models_to_test} were found.")
                return
        else:
            # Only test chat models by default
            models = [model for model in all_models if model.id.startswith(("gpt-", "o1", "o3"))]

        # Filter out deprecated models unless explicitly included
        if not include_deprecated:
            models = [model for model in models if not tester.is_model_deprecated(model.id)]

        # Filter out preview models unless explicitly included
        if not include_preview:
            models = [model for model in models if not tester.is_model_preview(model.id)]

        # Filter out non-OpenAI/system-owned models unless explicitly included
        if not include_non_openai_owned:
            models = [model for model in models if getattr(model, 'owned_by', '') in ['openai', 'system']]

        logger.info(f"Testing {len(models)} models: {[model.id for model in models]}")

        # Test each model
        for model in models:
            try:
                logger.info(f"Testing capabilities for {model.id}...")
                await tester.test_model_capabilities(model.id)
            except Exception as e:
                logger.error(f"Error testing {model.id}: {e}")

        # Fetch pricing data
        logger.info("Fetching pricing data...")
        pricing_data = await tester.fetch_pricing_data()

        # Update the models.py file if requested
        if update_file:
            logger.info("Updating models.py file...")
            tester.update_models_file([model.id for model in models])

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

async def update_pricing_only(
    api_key: str,
    update_file: bool = False,
    include_deprecated: bool = False,
    include_preview: bool = False,
    include_non_openai_owned: bool = False
) -> None:
    """Update only the pricing information in the models.py file."""
    tester = ModelCapabilityTester(api_key)
    tester.include_deprecated = include_deprecated
    tester.include_preview = include_preview
    tester.include_non_openai_owned = include_non_openai_owned

    try:
        # Fetch pricing data
        logger.info("Fetching pricing data...")
        pricing_data = await tester.fetch_pricing_data()

        # Update the pricing in the models.py file if requested
        if update_file:
            logger.info("Updating pricing in models.py file...")
            await tester.update_pricing_in_file(pricing_data)

        # Print a summary of the pricing data
        logger.info("\nPricing Data:")
        for model_id, prices in pricing_data.items():
            logger.info(f"{model_id}: ${prices.get('input_cost', 'N/A')} input, ${prices.get('output_cost', 'N/A')} output per 1K tokens")

    except Exception as e:
        logger.error(f"Error updating pricing: {e}")
        raise

if __name__ == "__main__":
    main()
