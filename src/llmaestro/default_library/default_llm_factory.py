from pydantic import BaseModel, Field
from pathlib import Path
from typing import Dict, Type
import logging
from enum import Enum

from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.llm.interfaces.base import BaseLLMInterface
from llmaestro.llm.models import Provider, LLMState
from llmaestro.llm.credentials import APIKey

logger = logging.getLogger(__name__)


class ProviderRequiredFiles(str, Enum):
    """Required files for a provider definition."""

    INTERFACE = "interface.py"
    MODELS = "models.py"
    PROVIDER = "provider.py"
    TOKENIZER = "tokenizer.py"


class LLMDefaultFactory(BaseModel):
    """Default factory for LLM states."""

    defined_providers_path: Path = Field(
        default=Path(__file__).parent / "defined_providers", description="Path to defined providers directory"
    )
    credentials: Dict[str, APIKey] = Field(
        default_factory=dict, description="Dictionary mapping provider names to their API keys"
    )

    async def DefaultLLMRegistryFactory(self) -> LLMRegistry:
        """Default factory for LLM registry.

        Creates a registry with:
        1. LLM instances from defined providers
        2. Associated credentials from provided API keys
        3. Appropriate interfaces for each instance

        Returns:
            Populated LLMRegistry

        Raises:
            RuntimeError: If no providers are registered, including a summary of all warnings
        """
        # Create new registry at the start
        registry = LLMRegistry()
        warnings = []

        try:
            # Get all provider directories
            provider_dirs = [
                d for d in self.defined_providers_path.iterdir() if d.is_dir() and not d.name.startswith("_")
            ]
            if not provider_dirs:
                raise RuntimeError(f"No provider directories found in {self.defined_providers_path}")

            # Track registered providers and models
            registered_providers = set()
            registered_models = []

            # Process each provider directory
            for provider_dir in provider_dirs:
                provider_name = provider_dir.name
                logger.info(f"Processing provider directory: {provider_name}")

                try:
                    # Check for required files
                    missing_files = []
                    for required_file in ProviderRequiredFiles:
                        if not (provider_dir / required_file.value).exists():
                            missing_files.append(required_file.value)

                    if missing_files:
                        warning = (
                            f"Provider {provider_name} is missing required files: {', '.join(missing_files)}. "
                            "Provider will be skipped."
                        )
                        warnings.append(warning)
                        logger.warning(warning)
                        continue

                    # Load provider configuration
                    provider = self._load_provider(provider_dir)
                    if not provider:
                        continue

                    # Get credentials for this provider
                    credentials = self.credentials.get(provider_name) or self.credentials.get(provider.family)
                    if not credentials:
                        warning = (
                            f"No credentials found for provider {provider_name} "
                            f"(family: {provider.family}). Provider will be skipped."
                        )
                        warnings.append(warning)
                        logger.warning(warning)
                        continue

                    # Load interface implementation
                    interface_class = self._load_interface(provider_dir)
                    if not interface_class:
                        continue

                    # Load and register models
                    models = self._load_models(provider_dir, provider)
                    if not models:
                        warning = f"No models found for provider {provider_name}"
                        warnings.append(warning)
                        logger.warning(warning)
                        continue

                    # Register each model
                    for model_name, llm_state in models.items():
                        try:
                            # Pass the interface class directly without instantiating
                            await registry.register_model(
                                state=llm_state, credentials=credentials, interface_class=interface_class
                            )
                            # Only add to registered models after successful registration
                            registered_models.append(model_name)
                            logger.info(
                                f"Successfully registered model {model_name} "
                                f"with {interface_class.__name__} interface"
                            )

                        except Exception as e:
                            warning = f"Failed to register model {model_name}: {str(e)}"
                            warnings.append(warning)
                            logger.error(warning)
                            continue

                    # Only add provider if at least one model was registered successfully
                    if any(model in registered_models for model in models.keys()):
                        registered_providers.add(provider_name)

                except Exception as e:
                    warning = f"Failed to process provider {provider_name}: {str(e)}"
                    warnings.append(warning)
                    logger.error(warning)
                    continue

            # Validate registration results and raise error with warnings if no providers
            if not registered_providers:
                warning_summary = "\n".join(warnings)
                raise RuntimeError(f"No providers were successfully registered. Warning summary:\n{warning_summary}")

            if not registered_models:
                warning_summary = "\n".join(warnings)
                raise RuntimeError(f"No models were successfully registered. Warning summary:\n{warning_summary}")

            if not self.credentials:
                warning_summary = "\n".join(warnings)
                raise RuntimeError(f"No credentials were provided. Warning summary:\n{warning_summary}")

            # Log summary
            logger.info(
                "Registry initialization complete. Registered %d providers: %s. Registered %d models: %s",
                len(registered_providers),
                list(registered_providers),
                len(registered_models),
                registered_models,
            )

            if warnings:
                logger.warning("Initialization completed with warnings:\n%s", "\n".join(warnings))

            return registry

        except Exception as e:
            # If we have warnings, include them in the error message
            if warnings:
                warning_summary = "\n".join(warnings)
                raise RuntimeError(f"Failed to initialize registry: {str(e)}\nWarning summary:\n{warning_summary}")
            else:
                raise RuntimeError(f"Failed to initialize registry: {str(e)}")

    def _load_provider(self, provider_dir: Path) -> Provider:
        """Load provider configuration from a provider directory.

        Args:
            provider_dir: Path to provider directory

        Returns:
            Provider configuration

        Raises:
            RuntimeError: If no provider instance is found or if there's an error loading the provider
        """
        provider_file = provider_dir / ProviderRequiredFiles.PROVIDER.value
        provider_module = self._import_module_from_file(provider_file)

        # Look for Provider instance in module
        provider = next((obj for name, obj in vars(provider_module).items() if isinstance(obj, Provider)), None)

        if not provider:
            raise RuntimeError(f"No Provider instance found in {provider_file}")

        return provider

    def _load_interface(self, provider_dir: Path) -> Type[BaseLLMInterface]:
        """Load interface implementation from a provider directory.

        Args:
            provider_dir: Path to provider directory

        Returns:
            Interface class

        Raises:
            RuntimeError: If no interface implementation is found or if there's an error loading the interface
        """
        interface_file = provider_dir / ProviderRequiredFiles.INTERFACE.value
        interface_module = self._import_module_from_file(interface_file)

        # Find interface class in module
        interface_class = next(
            (
                obj
                for name, obj in vars(interface_module).items()
                if isinstance(obj, type) and issubclass(obj, BaseLLMInterface) and obj != BaseLLMInterface
            ),
            None,
        )

        if not interface_class:
            raise RuntimeError(f"No interface implementation found in {interface_file}")

        return interface_class

    def _load_models(self, provider_dir: Path, provider: Provider) -> Dict[str, LLMState]:
        """Load model definitions from a provider directory.

        Args:
            provider_dir: Path to provider directory
            provider: Provider configuration

        Returns:
            Dictionary mapping model names to their LLM states

        Raises:
            RuntimeError: If no models are found or if there's an error loading the models
        """
        models_file = provider_dir / ProviderRequiredFiles.MODELS.value
        models_module = self._import_module_from_file(models_file)

        # Look for dictionary of model states or factory methods
        models: Dict[str, LLMState] = {}

        # Try different ways to find model definitions
        for name, obj in vars(models_module).items():
            # Skip non-model related classes
            if name in (
                "datetime",
                "Dict",
                "Callable",
                "LLMState",
                "LLMProfile",
                "LLMMetadata",
                "LLMCapabilities",
                "VisionCapabilities",
            ):
                continue

            # Case 1: Direct dictionary of LLMStates
            if isinstance(obj, dict) and all(isinstance(v, LLMState) for v in obj.values()):
                # Validate each state before adding
                for model_name, state in obj.items():
                    try:
                        # Ensure the state is valid
                        if not state.profile or not state.provider or not state.runtime_config:
                            logger.warning(f"Invalid LLMState for model {model_name}: missing required fields")
                            continue
                        models[model_name] = state
                    except Exception as e:
                        logger.warning(f"Failed to validate model {model_name}: {str(e)}")
                        continue

            # Case 2: Direct LLMState instance
            elif isinstance(obj, LLMState):
                try:
                    # Ensure the state is valid
                    if not obj.profile or not obj.provider or not obj.runtime_config:
                        logger.warning(f"Invalid LLMState instance {name}: missing required fields")
                        continue
                    models[obj.profile.name] = obj
                except Exception as e:
                    logger.warning(f"Failed to validate LLMState instance {name}: {str(e)}")
                    continue

            # Case 3: Class with MODELS dictionary or factory methods
            elif isinstance(obj, type) and name not in ("BaseModel", "datetime"):
                # Try to instantiate the class and get models
                try:
                    instance = obj()
                    # If class has a MODELS dictionary of factory methods
                    if hasattr(instance, "MODELS"):
                        class_models = instance.MODELS
                        if isinstance(class_models, dict):
                            # Call each factory method to get LLMState instances
                            for model_name, factory_method in class_models.items():
                                try:
                                    if callable(factory_method):
                                        state = factory_method()
                                        if isinstance(state, LLMState):
                                            # Ensure the state is valid
                                            if not state.profile or not state.provider or not state.runtime_config:
                                                logger.warning(
                                                    f"Invalid LLMState for model {model_name}: missing required fields"
                                                )
                                                continue
                                            models[model_name] = state
                                except Exception as e:
                                    logger.warning(f"Failed to create model {model_name}: {str(e)}")
                                    continue

                except Exception as e:
                    logger.warning(f"Failed to process class {name}: {str(e)}")
                    continue

        if not models:
            raise RuntimeError(f"No model definitions found in {models_file}")

        return models

    def _import_module_from_file(self, file_path: Path):
        """Import a Python module from a file path.

        Args:
            file_path: Path to Python file

        Returns:
            Imported module
        """
        import importlib.util

        try:
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if not spec or not spec.loader:
                raise ImportError(f"Failed to create module spec for {file_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        except Exception as e:
            raise ImportError(f"Failed to import module from {file_path}: {str(e)}")


if __name__ == "__main__":
    import asyncio

    credential = {"openai": APIKey(key="sk-proj-1234567890")}
    factory = LLMDefaultFactory(credentials=credential)
    # Create and run the async event loop

    registry = asyncio.run(factory.DefaultLLMRegistryFactory())
    print(registry)
