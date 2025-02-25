from pydantic import BaseModel, Field
from pathlib import Path
from typing import Dict, Type, List
import yaml

from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.llm.models import Provider, LLMState, LLMProfile, LLMCapabilities, LLMMetadata, LLMRuntimeConfig
from llmaestro.llm.enums import ModelFamily
from llmaestro.llm.credentials import APIKey

import logging

logger = logging.getLogger(__name__)


class LLMDefaultFactory(BaseModel):
    """Default factory for LLM states."""

    model_library_path: Path = Field(
        default=Path(__file__).parent / "model_library", description="Path to model library directory"
    )
    interface_library_path: Path = Field(
        default=Path(__file__).parent / "interfaces", description="Path to interface implementations"
    )
    api_keys: List[APIKey] = Field(default_factory=list, description="Available API keys")

    async def DefaultLLMRegistryFactory(self) -> LLMRegistry:
        """Default factory for LLM registry.

        Creates a registry with:
        1. LLM instances from model library
        2. Associated credentials from API keys
        3. Appropriate interfaces for each instance

        Returns:
            Populated LLMRegistry
        """

        # Create new registry
        registry = LLMRegistry()

        try:
            # Get all instances from model library
            instances = self.LLMInstanceFactory(self.model_library_path)

            # Load and cache API keys by model family
            credential_map: Dict[ModelFamily, APIKey] = {}
            for key in self.api_keys:
                credential_map[key.model_family] = key
                logger.info(f"Found credentials for model family {key.model_family}")

            # Associate instances with credentials and interfaces
            for model_name, llm_state in instances.items():
                try:
                    # Get credentials for this instance's model family
                    model_family = llm_state.model_family
                    credentials = credential_map.get(model_family)

                    if not credentials:
                        logger.warning(
                            f"No credentials found for model {model_name} "
                            f"(family: {model_family}). Model will not be registered."
                        )
                        continue

                    # Get interface implementation
                    try:
                        interface_class = self.InterfaceFactory(model_family)
                    except ValueError as e:
                        logger.warning(
                            f"No interface implementation found for model {model_name} "
                            f"(family: {model_family}). Model will not be registered. Error: {str(e)}"
                        )
                        continue

                    # Register model with credentials and interface
                    instance = await registry.register_model(
                        state=llm_state, credentials=credentials, interface_class=interface_class
                    )
                    logger.info(
                        f"Successfully registered model {model_name} " f"with {interface_class.__name__} interface"
                    )

                except Exception as e:
                    logger.error(f"Failed to register model {model_name}: {str(e)}. " "Continuing with next model.")
                    continue

            # Log summary
            logger.info(
                f"Registry initialization complete. "
                f"Registered {len(registry.models)} models: "
                f"{list(registry.models.keys())}"
            )

            return registry

        except Exception as e:
            logger.error(f"Failed to initialize registry: {str(e)}")
            raise

    def InterfaceFactory(self, model_family: ModelFamily) -> Type["BaseLLMInterface"]:
        """Get the appropriate interface implementation for a model family.

        Args:
            model_family: The model family to get the interface for

        Returns:
            The interface class for the model family

        Raises:
            ValueError: If no interface implementation is found for the model family
        """
        # Lazy load and cache interface implementations
        if not hasattr(self, "_interface_cache"):
            self._interface_cache = self._load_interface_implementations()

        # Return cached implementation or raise error
        if model_family not in self._interface_cache:
            raise ValueError(
                f"No interface implementation found for model family {model_family}. "
                f"Available implementations: {list(self._interface_cache.keys())}"
            )

        return self._interface_cache[model_family]

    def _load_interface_implementations(self) -> Dict[ModelFamily, Type["BaseLLMInterface"]]:
        """Load and cache interface implementations from the interface library.

        Returns:
            Dictionary mapping model families to their interface implementations
        """
        import importlib.util
        import inspect

        implementations: Dict[ModelFamily, Type["BaseLLMInterface"]] = {}
        interface_path = self.interface_library_path or Path(__file__).parent / "interfaces"

        if not interface_path.exists():
            raise FileNotFoundError(f"Interface library directory not found at {interface_path}")

        # Import BaseLLMInterface for isinstance checks
        from llmaestro.llm.interfaces.base import BaseLLMInterface

        # Scan all Python files in interface directory
        for file_path in interface_path.rglob("*.py"):
            if file_path.name.startswith("_"):
                continue

            try:
                # Import module
                module_name = file_path.stem
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if not spec or not spec.loader:
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find interface classes in module
                for name, obj in inspect.getmembers(module):
                    if not inspect.isclass(obj):
                        continue

                    # Check if it's a concrete implementation of BaseLLMInterface
                    if issubclass(obj, BaseLLMInterface) and obj != BaseLLMInterface and not inspect.isabstract(obj):
                        try:
                            # Create temporary instance to get model family
                            # This is safe because Pydantic models can be instantiated with no args
                            # and will use default values or raise validation errors
                            instance = obj()
                            model_family = instance.model_family

                            if model_family in implementations:
                                logging.warning(
                                    f"Multiple implementations found for model family {model_family}. "
                                    f"Using {obj.__name__} instead of {implementations[model_family].__name__}"
                                )

                            implementations[model_family] = obj

                        except Exception as e:
                            logging.warning(
                                f"Failed to load interface implementation {obj.__name__} " f"from {file_path}: {str(e)}"
                            )

            except Exception as e:
                logging.error(f"Failed to load interfaces from {file_path}: {str(e)}")
                continue

        return implementations

    def LLMInstanceFactory(self, model_library_path: Path) -> Dict[str, LLMState]:
        """Create LLM instances from model library.

        Args:
            model_library_path: Path to model library directory

        Returns:
            Dictionary mapping model names to LLM states
        """
        instances = {}

        try:
            # Load configurations from model library
            if not model_library_path.exists():
                raise FileNotFoundError(f"Model library directory not found at {model_library_path}")

            for config_file in model_library_path.glob("*.yaml"):
                try:
                    with open(config_file) as f:
                        data = yaml.safe_load(f)

                    # Create provider instance
                    provider_data = data["provider"]
                    provider = Provider(**provider_data)

                    # Get default runtime config
                    default_runtime = LLMRuntimeConfig(**data.get("default_runtime_config", {}))

                    # Process each model
                    for model_data in data["models"]:
                        try:
                            profile_data = model_data["profile"]

                            # Create profile with capabilities and metadata
                            profile = LLMProfile(
                                name=profile_data["name"],
                                version=profile_data.get("version"),
                                description=profile_data.get("description"),
                                capabilities=LLMCapabilities(**profile_data["capabilities"]),
                                metadata=LLMMetadata(**profile_data["metadata"]),
                            )

                            # Create runtime config (merge default with model-specific)
                            runtime_config = LLMRuntimeConfig(
                                **{**default_runtime.model_dump(), **profile_data.get("runtime_config", {})}
                            )

                            # Create LLM state
                            state = LLMState(profile=profile, provider=provider, runtime_config=runtime_config)
                            instances[profile.name] = state
                            logger.info(f"Created instance for model {profile.name}")

                        except Exception as e:
                            logger.error(f"Failed to create instance from {config_file}: {str(e)}")
                            continue

                except Exception as e:
                    logger.error(f"Failed to load config from {config_file}: {str(e)}")
                    continue

            return instances

        except Exception as e:
            logger.error(f"Failed to load model library: {str(e)}")
            raise


if __name__ == "__main__":
    import asyncio

    factory = LLMDefaultFactory()
    # Create and run the async event loop
    registry = asyncio.run(factory.DefaultLLMRegistryFactory())
    print(registry)
