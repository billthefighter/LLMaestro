"""OpenAI-specific model implementations."""
from datetime import datetime
from typing import Dict, Callable

from llmaestro.llm.capabilities import LLMCapabilities, VisionCapabilities
from llmaestro.llm.models import LLMState, LLMProfile, LLMMetadata, LLMRuntimeConfig
from llmaestro.default_library.defined_providers.openai.provider import OPENAI_PROVIDER


class OpenAIModels:
    """Interface for accessing OpenAI model configurations."""

    @staticmethod
    def chatgpt_4o_latest() -> LLMState:
        """Create LLMState for ChatGPT-4o-latest model.

        Returns:
            LLMState: Configured state for ChatGPT-4o-latest model
        """
        return LLMState(
            profile=LLMProfile(
                name="chatgpt-4o-latest",
                version="2024-07",
                description="Latest GPT-4o model used in ChatGPT",
                capabilities=LLMCapabilities(
                    max_context_window=128000,
                    max_output_tokens=16384,
                    supports_streaming=True,
                    supports_function_calling=True,
                    supports_vision=False,
                    supports_embeddings=False,
                    supports_json_mode=True,
                    supports_system_prompt=True,
                    supports_tools=True,
                    supports_parallel_requests=True,
                    supports_frequency_penalty=True,
                    supports_presence_penalty=True,
                    supports_stop_sequences=True,
                    supports_message_role=True,
                    supports_direct_pydantic_parse=False,
                    typical_speed=200.0,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.01,
                    output_cost_per_1k_tokens=0.03,
                ),
                metadata=LLMMetadata(
                    release_date=datetime(2024, 7, 1),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2024-07-01",
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=16384, temperature=0.7, max_context_tokens=128000, stream=True),
        )

    @staticmethod
    def gpt4o_mini() -> LLMState:
        """Create LLMState for GPT-4o-mini model.

        Returns:
            LLMState: Configured state for GPT-4o-mini model
        """
        return LLMState(
            profile=LLMProfile(
                name="gpt-4o-mini-2024-07-18",
                version="2024-07",
                description="Efficient GPT-4o mini model",
                capabilities=LLMCapabilities(
                    max_context_window=128000,
                    max_output_tokens=16384,
                    supports_streaming=True,
                    supports_function_calling=True,
                    supports_vision=True,
                    supports_embeddings=False,
                    supports_json_mode=True,
                    supports_system_prompt=True,
                    supports_tools=True,
                    supports_parallel_requests=True,
                    supports_frequency_penalty=True,
                    supports_presence_penalty=True,
                    supports_stop_sequences=True,
                    supports_message_role=True,
                    supports_direct_pydantic_parse=True,
                    typical_speed=220.0,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.008,
                    output_cost_per_1k_tokens=0.024,
                ),
                metadata=LLMMetadata(
                    release_date=datetime(2024, 7, 18),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2024-07-18",
                ),
                vision_capabilities=VisionCapabilities(
                    max_images_per_request=1,
                    supported_formats=["png", "jpeg", "gif", "webp"],
                    max_image_size_mb=20,
                    max_image_resolution=2048,
                    supports_image_annotations=False,
                    supports_image_analysis=True,
                    supports_image_generation=False,
                    cost_per_image=0.01,
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=16384, temperature=0.7, max_context_tokens=128000, stream=True),
        )

    @staticmethod
    def o1() -> LLMState:
        """Create LLMState for O1 model.

        Returns:
            LLMState: Configured state for O1 model
        """
        return LLMState(
            profile=LLMProfile(
                name="o1-2024-12-17",
                version="2024-12",
                description="Advanced O1 model with extended context window",
                capabilities=LLMCapabilities(
                    max_context_window=200000,
                    max_output_tokens=100000,
                    supports_streaming=True,
                    supports_function_calling=True,
                    supports_vision=False,
                    supports_embeddings=False,
                    supports_json_mode=True,
                    supports_system_prompt=True,
                    supports_tools=True,
                    supports_parallel_requests=True,
                    supports_frequency_penalty=True,
                    supports_presence_penalty=True,
                    supports_stop_sequences=True,
                    supports_message_role=True,
                    supports_direct_pydantic_parse=False,
                    typical_speed=250.0,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.012,
                    output_cost_per_1k_tokens=0.036,
                ),
                metadata=LLMMetadata(
                    release_date=datetime(2024, 12, 17),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2024-12-17",
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=100000, temperature=0.7, max_context_tokens=200000, stream=True),
        )

    @staticmethod
    def o3_mini() -> LLMState:
        """Create LLMState for O3-mini model.

        Returns:
            LLMState: Configured state for O3-mini model
        """
        return LLMState(
            profile=LLMProfile(
                name="o3-mini-2025-01-31",
                version="2025-01",
                description="Efficient O3 mini model with extended capabilities",
                capabilities=LLMCapabilities(
                    max_context_window=200000,
                    max_output_tokens=100000,
                    supports_streaming=True,
                    supports_function_calling=True,
                    supports_vision=False,
                    supports_embeddings=False,
                    supports_json_mode=True,
                    supports_system_prompt=True,
                    supports_tools=True,
                    supports_parallel_requests=True,
                    supports_frequency_penalty=True,
                    supports_presence_penalty=True,
                    supports_stop_sequences=True,
                    supports_message_role=True,
                    supports_direct_pydantic_parse=False,
                    typical_speed=280.0,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.009,
                    output_cost_per_1k_tokens=0.027,
                ),
                metadata=LLMMetadata(
                    release_date=datetime(2025, 1, 31),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2025-01-31",
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=100000, temperature=0.7, max_context_tokens=200000, stream=True),
        )

    # Dictionary mapping model names to their factory methods
    MODELS: Dict[str, Callable[[], LLMState]] = {
        "chatgpt-4o-latest": chatgpt_4o_latest,
        "gpt-4o-mini-2024-07-18": gpt4o_mini,
        "o1-2024-12-17": o1,
        "o3-mini-2025-01-31": o3_mini,
    }

    @classmethod
    def get_model(cls, model_name: str) -> LLMState:
        """Get a model state by name.

        Args:
            model_name: Name of the model to retrieve

        Returns:
            LLMState: Configured state for the requested model

        Raises:
            KeyError: If model_name is not found
        """
        if model_name not in cls.MODELS:
            raise KeyError(f"Model {model_name} not found in OpenAI models")
        return cls.MODELS[model_name]()
