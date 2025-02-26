"""Unified registry for managing LLM models."""
import logging
from typing import Dict, List, Type
from threading import Lock

from pydantic import BaseModel, ConfigDict, Field

from .models import LLMInstance, LLMState
from .credentials import APIKey
from .interfaces.base import BaseLLMInterface


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
        """
        if not credentials:
            raise ValueError(f"No credentials provided for model {state.model_name}")

        with self.lock:
            self.model_states[state.model_name] = state
            self.interface_classes[state.model_name] = interface_class
            self.credentials[state.model_name] = credentials

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
                raise ValueError(f"Model {model_name} is not registered")

            state = self.model_states[model_name]
            interface_class = self.interface_classes[model_name]
            credentials = self.credentials[model_name]

        try:
            # Create a fresh interface instance
            interface = interface_class(state=state, credentials=credentials)

            # Create a fresh LLM instance
            return LLMInstance(state=state, credentials=credentials, interface=interface)

        except Exception as e:
            logging.error(f"Failed to create instance for model {model_name}: {str(e)}")
            raise

    def get_registered_models(self) -> List[str]:
        """Get list of registered model names."""
        with self.lock:
            return list(self.model_states.keys())


# Rebuild the model to resolve circular dependencies
LLMRegistry.model_rebuild()
