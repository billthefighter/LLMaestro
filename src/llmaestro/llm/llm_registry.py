"""Unified registry for managing LLM models."""
import logging
from typing import Dict, List, Type
from threading import Lock

from pydantic import BaseModel, ConfigDict, Field

from .models import LLMInstance, LLMState
from .credentials import APIKey
from .interfaces.base import BaseLLMInterface

logger = logging.getLogger(__name__)


class LLMRegistry(BaseModel):
    """Registry for managing LLM models and their interfaces.

    This registry serves as the single source of truth for:
    1. Model configurations and capabilities
    2. Model-provider relationships
    3. Model lifecycle management
    4. Interface instances
    """

    model_states: Dict[str, LLMState] = Field(default_factory=dict, description="Registered model states")
    interface_classes: Dict[str, Type[BaseLLMInterface]] = Field(
        default_factory=dict, description="Interface classes for each model"
    )
    credentials: Dict[str, APIKey] = Field(default_factory=dict, description="Credentials for each model")
    lock: Lock = Field(default_factory=Lock)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def register_model(
        self, state: LLMState, interface_class: Type[BaseLLMInterface], credentials: APIKey
    ) -> None:
        """Register a model configuration.

        Args:
            state: Model state to register
            credentials: Credentials for the model
            interface_class: Interface implementation class

        Raises:
            ValueError: If credentials are missing or state is invalid
        """
        logger.debug(f"Attempting to register model {state.model_name}")

        if not credentials:
            logger.error(f"No credentials provided for model {state.model_name}")
            raise ValueError(f"No credentials provided for model {state.model_name}")

        # Validate state before registration
        if not state.profile or not state.provider or not state.runtime_config:
            logger.error(f"Invalid LLMState for model {state.model_name}: missing required fields")
            raise ValueError(f"Invalid LLMState for model {state.model_name}: missing required fields")

        try:
            # Validate that the state is properly configured
            if not state.profile.name:
                logger.error("Model state missing name in profile")
                raise ValueError("Model state missing name in profile")
            if not state.profile.capabilities:
                logger.error("Model state missing capabilities in profile")
                raise ValueError("Model state missing capabilities in profile")
            if not state.provider.family:
                logger.error("Model state missing provider family")
                raise ValueError("Model state missing provider family")
        except Exception as e:
            logger.error(f"Invalid model state for {state.model_name}: {str(e)}")
            raise ValueError(f"Invalid model state for {state.model_name}: {str(e)}")

        logger.debug(f"Model {state.model_name} passed validation, registering...")

        with self.lock:
            self.model_states[state.model_name] = state
            self.interface_classes[state.model_name] = interface_class
            self.credentials[state.model_name] = credentials

        logger.debug(f"Successfully registered model {state.model_name}")

    async def create_instance(self, model_name: str) -> LLMInstance:
        """Create a new instance of a registered model.

        This method always returns a fresh instance with its own interface.

        Args:
            model_name: Name of the registered model

        Returns:
            A new LLMInstance configured for the requested model

        Raises:
            ValueError: If the model is not registered
        """
        with self.lock:
            if model_name not in self.model_states:
                logger.error(f"Model {model_name} not found in registry")
                raise ValueError(f"Model {model_name} is not registered")

            state = self.model_states[model_name]
            interface_class = self.interface_classes[model_name]
            credentials = self.credentials[model_name]

            logger.debug(f"Creating instance for model {model_name}")
            logger.debug(f"Using interface class: {interface_class.__name__}")
            logger.debug(f"State type: {type(state)}")
            logger.debug(f"Credentials type: {type(credentials)}")

        try:
            # Create a fresh interface instance
            logger.debug("Creating interface instance...")
            interface = interface_class(state=state, credentials=credentials)
            logger.debug(f"Interface created successfully: {type(interface)}")

            # Create a fresh LLM instance
            logger.debug("Creating LLM instance...")
            instance = LLMInstance(state=state, credentials=credentials, interface=interface)
            logger.debug("LLM instance created successfully")

            return instance

        except Exception as e:
            logger.error(f"Failed to create instance for model {model_name}: {str(e)}", exc_info=True)
            raise

    def get_registered_models(self) -> List[str]:
        """Get list of registered model names."""
        with self.lock:
            return list(self.model_states.keys())
