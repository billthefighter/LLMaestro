"""OpenAI-specific model implementations."""
from datetime import datetime
from typing import Dict, Callable

from llmaestro.llm.capabilities import LLMCapabilities, VisionCapabilities
from llmaestro.llm.models import LLMState, LLMProfile, LLMMetadata, LLMRuntimeConfig
from llmaestro.default_library.defined_profiders.openai.provider import OPENAI_PROVIDER


class OpenAIModels:
    """Interface for accessing OpenAI model configurations."""

    @staticmethod
    def gpt4_vision() -> LLMState:
        """Create LLMState for GPT-4 Vision model.

        Returns:
            LLMState: Configured state for GPT-4 Vision model
        """
        return LLMState(
            profile=LLMProfile(
                name="gpt-4-vision-preview",
                version="2024-03",
                description="Most capable GPT-4 model with vision capabilities",
                capabilities=LLMCapabilities(
                    max_context_window=128000,
                    max_output_tokens=4096,
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
                    typical_speed=150.0,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.01,
                    output_cost_per_1k_tokens=0.03,
                ),
                metadata=LLMMetadata(
                    release_date=datetime(2024, 3, 1),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2024-02-29",
                ),
                vision_capabilities=VisionCapabilities(
                    max_images_per_request=1,
                    supported_formats=["png", "jpeg"],
                    max_image_size_mb=20,
                    max_image_resolution=2048,
                    supports_image_annotations=False,
                    supports_image_analysis=True,
                    supports_image_generation=False,
                    cost_per_image=0.01,
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=4096, temperature=0.7, max_context_tokens=128000, stream=True),
        )

    @staticmethod
    def gpt4_turbo() -> LLMState:
        """Create LLMState for GPT-4 Turbo model.

        Returns:
            LLMState: Configured state for GPT-4 Turbo model
        """
        return LLMState(
            profile=LLMProfile(
                name="gpt-4-turbo-preview",
                version="2024-03",
                description="Latest GPT-4 model optimized for performance and cost",
                capabilities=LLMCapabilities(
                    max_context_window=128000,
                    max_output_tokens=4096,
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
                    typical_speed=180.0,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.01,
                    output_cost_per_1k_tokens=0.03,
                ),
                metadata=LLMMetadata(
                    release_date=datetime(2024, 3, 1),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2024-02-29",
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=4096, temperature=0.7, max_context_tokens=128000, stream=True),
        )

    @staticmethod
    def gpt35_turbo() -> LLMState:
        """Create LLMState for GPT-3.5 Turbo model.

        Returns:
            LLMState: Configured state for GPT-3.5 Turbo model
        """
        return LLMState(
            profile=LLMProfile(
                name="gpt-3.5-turbo",
                version="2024-03",
                description="Fast and cost-effective GPT-3.5 model for most tasks",
                capabilities=LLMCapabilities(
                    max_context_window=16385,
                    max_output_tokens=4096,
                    supports_streaming=True,
                    supports_function_calling=True,
                    supports_vision=False,
                    supports_embeddings=False,
                    supports_json_mode=True,
                    supports_system_prompt=True,
                    supports_tools=True,
                    supports_parallel_requests=True,
                    supports_message_role=True,
                    typical_speed=200.0,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.0005,
                    output_cost_per_1k_tokens=0.0015,
                ),
                metadata=LLMMetadata(
                    release_date=datetime(2024, 3, 1),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2024-02-29",
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=4096, temperature=0.7, max_context_tokens=16385, stream=True),
        )

    # Dictionary mapping model names to their factory methods
    MODELS: Dict[str, Callable[[], LLMState]] = {
        "gpt-4-vision-preview": gpt4_vision,
        "gpt-4-turbo-preview": gpt4_turbo,
        "gpt-3.5-turbo": gpt35_turbo,
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
