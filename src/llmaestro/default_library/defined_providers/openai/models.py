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

    @staticmethod
    def gpt_4o_2024_11_20() -> LLMState:
        """Create LLMState for gpt-4o-2024-11-20 model.

        Returns:
            LLMState: Configured state for gpt-4o-2024-11-20 model
        """
        return LLMState(
            profile=LLMProfile(
                name="gpt-4o-2024-11-20",
                version="2025-03",
                description="gpt-4o-2024-11-20 model",
                capabilities=LLMCapabilities(
                    max_context_window=8192,
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
                    supports_direct_pydantic_parse=False,
                    typical_speed=None,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                ),
                metadata=LLMMetadata(
                    release_date=datetime.fromtimestamp(1741045356),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2023-05-15",
                ),
                vision_capabilities=VisionCapabilities(
                    max_images_per_request=2,
                    supported_formats=["jpeg", "png", "webp"],
                    max_image_size_mb=20,
                    max_image_resolution=2048,
                    supports_image_annotations=False,
                    supports_image_analysis=True,
                    supports_image_generation=False,
                    cost_per_image=0.002,
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=1024, temperature=0.7, max_context_tokens=8192, stream=True),
        )

    @staticmethod
    def o1_mini_2024_09_12() -> LLMState:
        """Create LLMState for o1-mini-2024-09-12 model.

        Returns:
            LLMState: Configured state for o1-mini-2024-09-12 model
        """
        return LLMState(
            profile=LLMProfile(
                name="o1-mini-2024-09-12",
                version="2025-03",
                description="o1-mini-2024-09-12 model",
                capabilities=LLMCapabilities(
                    max_context_window=4096,
                    max_output_tokens=4096,
                    supports_streaming=True,
                    supports_function_calling=False,
                    supports_vision=False,
                    supports_embeddings=False,
                    supports_json_mode=False,
                    supports_system_prompt=False,
                    supports_tools=False,
                    supports_parallel_requests=True,
                    supports_frequency_penalty=False,
                    supports_presence_penalty=False,
                    supports_stop_sequences=True,
                    supports_message_role=True,
                    supports_direct_pydantic_parse=False,
                    typical_speed=None,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                ),
                metadata=LLMMetadata(
                    release_date=datetime.fromtimestamp(1741045356),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2023-05-15",
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=1024, temperature=0.7, max_context_tokens=4096, stream=True),
        )

    @staticmethod
    def o1_mini() -> LLMState:
        """Create LLMState for o1-mini model.

        Returns:
            LLMState: Configured state for o1-mini model
        """
        return LLMState(
            profile=LLMProfile(
                name="o1-mini",
                version="2025-03",
                description="o1-mini model",
                capabilities=LLMCapabilities(
                    max_context_window=4096,
                    max_output_tokens=4096,
                    supports_streaming=True,
                    supports_function_calling=False,
                    supports_vision=False,
                    supports_embeddings=False,
                    supports_json_mode=False,
                    supports_system_prompt=False,
                    supports_tools=False,
                    supports_parallel_requests=True,
                    supports_frequency_penalty=False,
                    supports_presence_penalty=False,
                    supports_stop_sequences=True,
                    supports_message_role=True,
                    supports_direct_pydantic_parse=False,
                    typical_speed=None,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                ),
                metadata=LLMMetadata(
                    release_date=datetime.fromtimestamp(1741045356),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2023-05-15",
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=1024, temperature=0.7, max_context_tokens=4096, stream=True),
        )

    @staticmethod
    def gpt_4_turbo() -> LLMState:
        """Create LLMState for gpt-4-turbo model.

        Returns:
            LLMState: Configured state for gpt-4-turbo model
        """
        return LLMState(
            profile=LLMProfile(
                name="gpt-4-turbo",
                version="2025-03",
                description="gpt-4-turbo model",
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
                    supports_direct_pydantic_parse=False,
                    typical_speed=None,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                ),
                metadata=LLMMetadata(
                    release_date=datetime.fromtimestamp(1741045356),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2023-05-15",
                ),
                vision_capabilities=VisionCapabilities(
                    max_images_per_request=2,
                    supported_formats=["jpeg", "png", "webp"],
                    max_image_size_mb=20,
                    max_image_resolution=2048,
                    supports_image_annotations=False,
                    supports_image_analysis=True,
                    supports_image_generation=False,
                    cost_per_image=0.002,
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=1024, temperature=0.7, max_context_tokens=128000, stream=True),
        )

    @staticmethod
    def gpt_4() -> LLMState:
        """Create LLMState for gpt-4 model.

        Returns:
            LLMState: Configured state for gpt-4 model
        """
        return LLMState(
            profile=LLMProfile(
                name="gpt-4",
                version="2025-03",
                description="gpt-4 model",
                capabilities=LLMCapabilities(
                    max_context_window=8192,
                    max_output_tokens=4096,
                    supports_streaming=True,
                    supports_function_calling=True,
                    supports_vision=False,
                    supports_embeddings=False,
                    supports_json_mode=False,
                    supports_system_prompt=True,
                    supports_tools=True,
                    supports_parallel_requests=True,
                    supports_frequency_penalty=True,
                    supports_presence_penalty=True,
                    supports_stop_sequences=True,
                    supports_message_role=True,
                    supports_direct_pydantic_parse=False,
                    typical_speed=None,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                ),
                metadata=LLMMetadata(
                    release_date=datetime.fromtimestamp(1741045356),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2023-05-15",
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=1024, temperature=0.7, max_context_tokens=8192, stream=True),
        )

    @staticmethod
    def o1_2024_12_17() -> LLMState:
        """Create LLMState for o1-2024-12-17 model.

        Returns:
            LLMState: Configured state for o1-2024-12-17 model
        """
        return LLMState(
            profile=LLMProfile(
                name="o1-2024-12-17",
                version="2025-03",
                description="o1-2024-12-17 model",
                capabilities=LLMCapabilities(
                    max_context_window=4096,
                    max_output_tokens=4096,
                    supports_streaming=True,
                    supports_function_calling=False,
                    supports_vision=True,
                    supports_embeddings=False,
                    supports_json_mode=False,
                    supports_system_prompt=True,
                    supports_tools=True,
                    supports_parallel_requests=True,
                    supports_frequency_penalty=False,
                    supports_presence_penalty=False,
                    supports_stop_sequences=True,
                    supports_message_role=True,
                    supports_direct_pydantic_parse=False,
                    typical_speed=None,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                ),
                metadata=LLMMetadata(
                    release_date=datetime.fromtimestamp(1741045356),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2023-05-15",
                ),
                vision_capabilities=VisionCapabilities(
                    max_images_per_request=2,
                    supported_formats=["jpeg", "png", "webp"],
                    max_image_size_mb=20,
                    max_image_resolution=2048,
                    supports_image_annotations=False,
                    supports_image_analysis=True,
                    supports_image_generation=False,
                    cost_per_image=0.002,
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=1024, temperature=0.7, max_context_tokens=4096, stream=True),
        )

    @staticmethod
    def gpt_4o_mini() -> LLMState:
        """Create LLMState for gpt-4o-mini model.

        Returns:
            LLMState: Configured state for gpt-4o-mini model
        """
        return LLMState(
            profile=LLMProfile(
                name="gpt-4o-mini",
                version="2025-03",
                description="gpt-4o-mini model",
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
                    supports_direct_pydantic_parse=False,
                    typical_speed=None,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                ),
                metadata=LLMMetadata(
                    release_date=datetime.fromtimestamp(1741045356),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2023-05-15",
                ),
                vision_capabilities=VisionCapabilities(
                    max_images_per_request=2,
                    supported_formats=["jpeg", "png", "webp"],
                    max_image_size_mb=20,
                    max_image_resolution=2048,
                    supports_image_annotations=False,
                    supports_image_analysis=True,
                    supports_image_generation=False,
                    cost_per_image=0.002,
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=1024, temperature=0.7, max_context_tokens=128000, stream=True),
        )

    @staticmethod
    def gpt_4o() -> LLMState:
        """Create LLMState for gpt-4o model.

        Returns:
            LLMState: Configured state for gpt-4o model
        """
        return LLMState(
            profile=LLMProfile(
                name="gpt-4o",
                version="2025-03",
                description="gpt-4o model",
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
                    supports_direct_pydantic_parse=False,
                    typical_speed=None,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                ),
                metadata=LLMMetadata(
                    release_date=datetime.fromtimestamp(1741045356),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2023-05-15",
                ),
                vision_capabilities=VisionCapabilities(
                    max_images_per_request=2,
                    supported_formats=["jpeg", "png", "webp"],
                    max_image_size_mb=20,
                    max_image_resolution=2048,
                    supports_image_annotations=False,
                    supports_image_analysis=True,
                    supports_image_generation=False,
                    cost_per_image=0.002,
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=1024, temperature=0.7, max_context_tokens=128000, stream=True),
        )

    @staticmethod
    def gpt_3_5_turbo_instruct() -> LLMState:
        """Create LLMState for gpt-3.5-turbo-instruct model.

        Returns:
            LLMState: Configured state for gpt-3.5-turbo-instruct model
        """
        return LLMState(
            profile=LLMProfile(
                name="gpt-3.5-turbo-instruct",
                version="2025-03",
                description="gpt-3.5-turbo-instruct model",
                capabilities=LLMCapabilities(
                    max_context_window=4096,
                    max_output_tokens=4096,
                    supports_streaming=False,
                    supports_function_calling=False,
                    supports_vision=False,
                    supports_embeddings=False,
                    supports_json_mode=False,
                    supports_system_prompt=False,
                    supports_tools=False,
                    supports_parallel_requests=True,
                    supports_frequency_penalty=False,
                    supports_presence_penalty=False,
                    supports_stop_sequences=False,
                    supports_message_role=True,
                    supports_direct_pydantic_parse=False,
                    typical_speed=None,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                ),
                metadata=LLMMetadata(
                    release_date=datetime.fromtimestamp(1741045356),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2023-05-15",
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=1024, temperature=0.7, max_context_tokens=4096, stream=True),
        )

    @staticmethod
    def gpt_3_5_turbo_instruct_0914() -> LLMState:
        """Create LLMState for gpt-3.5-turbo-instruct-0914 model.

        Returns:
            LLMState: Configured state for gpt-3.5-turbo-instruct-0914 model
        """
        return LLMState(
            profile=LLMProfile(
                name="gpt-3.5-turbo-instruct-0914",
                version="2025-03",
                description="gpt-3.5-turbo-instruct-0914 model",
                capabilities=LLMCapabilities(
                    max_context_window=4096,
                    max_output_tokens=4096,
                    supports_streaming=False,
                    supports_function_calling=False,
                    supports_vision=False,
                    supports_embeddings=False,
                    supports_json_mode=False,
                    supports_system_prompt=False,
                    supports_tools=False,
                    supports_parallel_requests=True,
                    supports_frequency_penalty=False,
                    supports_presence_penalty=False,
                    supports_stop_sequences=False,
                    supports_message_role=True,
                    supports_direct_pydantic_parse=False,
                    typical_speed=None,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                ),
                metadata=LLMMetadata(
                    release_date=datetime.fromtimestamp(1741045356),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2023-05-15",
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=1024, temperature=0.7, max_context_tokens=4096, stream=True),
        )

    @staticmethod
    def gpt_3_5_turbo_0125() -> LLMState:
        """Create LLMState for gpt-3.5-turbo-0125 model.

        Returns:
            LLMState: Configured state for gpt-3.5-turbo-0125 model
        """
        return LLMState(
            profile=LLMProfile(
                name="gpt-3.5-turbo-0125",
                version="2025-03",
                description="gpt-3.5-turbo-0125 model",
                capabilities=LLMCapabilities(
                    max_context_window=4096,
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
                    supports_direct_pydantic_parse=False,
                    typical_speed=None,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                ),
                metadata=LLMMetadata(
                    release_date=datetime.fromtimestamp(1741045356),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2023-05-15",
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=1024, temperature=0.7, max_context_tokens=4096, stream=True),
        )

    @staticmethod
    def gpt_3_5_turbo() -> LLMState:
        """Create LLMState for gpt-3.5-turbo model.

        Returns:
            LLMState: Configured state for gpt-3.5-turbo model
        """
        return LLMState(
            profile=LLMProfile(
                name="gpt-3.5-turbo",
                version="2025-03",
                description="gpt-3.5-turbo model",
                capabilities=LLMCapabilities(
                    max_context_window=4096,
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
                    supports_direct_pydantic_parse=False,
                    typical_speed=None,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                ),
                metadata=LLMMetadata(
                    release_date=datetime.fromtimestamp(1741045356),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2023-05-15",
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=1024, temperature=0.7, max_context_tokens=4096, stream=True),
        )

    @staticmethod
    def o3_mini_2025_01_31() -> LLMState:
        """Create LLMState for o3-mini-2025-01-31 model.

        Returns:
            LLMState: Configured state for o3-mini-2025-01-31 model
        """
        return LLMState(
            profile=LLMProfile(
                name="o3-mini-2025-01-31",
                version="2025-03",
                description="o3-mini-2025-01-31 model",
                capabilities=LLMCapabilities(
                    max_context_window=4096,
                    max_output_tokens=4096,
                    supports_streaming=True,
                    supports_function_calling=False,
                    supports_vision=False,
                    supports_embeddings=False,
                    supports_json_mode=False,
                    supports_system_prompt=True,
                    supports_tools=False,
                    supports_parallel_requests=True,
                    supports_frequency_penalty=False,
                    supports_presence_penalty=False,
                    supports_stop_sequences=True,
                    supports_message_role=True,
                    supports_direct_pydantic_parse=False,
                    typical_speed=None,
                    supported_languages={"en"},
                    input_cost_per_1k_tokens=0.0,
                    output_cost_per_1k_tokens=0.0,
                ),
                metadata=LLMMetadata(
                    release_date=datetime.fromtimestamp(1741045356),
                    is_preview=False,
                    is_deprecated=False,
                    min_api_version="2023-05-15",
                ),
            ),
            provider=OPENAI_PROVIDER,
            runtime_config=LLMRuntimeConfig(max_tokens=1024, temperature=0.7, max_context_tokens=4096, stream=True),
        )

    MODELS: Dict[str, Callable[[], LLMState]] = {
        "gpt-4o-2024-11-20": gpt_4o_2024_11_20,
        "o1-mini-2024-09-12": o1_mini_2024_09_12,
        "o1-mini": o1_mini,
        "gpt-4-turbo": gpt_4_turbo,
        "gpt-4": gpt_4,
        "o1-2024-12-17": o1_2024_12_17,
        "gpt-4o-mini": gpt_4o_mini,
        "gpt-4o": gpt_4o,
        "gpt-3.5-turbo-instruct": gpt_3_5_turbo_instruct,
        "gpt-3.5-turbo-instruct-0914": gpt_3_5_turbo_instruct_0914,
        "gpt-3.5-turbo-0125": gpt_3_5_turbo_0125,
        "gpt-3.5-turbo": gpt_3_5_turbo,
        "o3-mini": o3_mini,
        "o3-mini-2025-01-31": o3_mini_2025_01_31,
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
