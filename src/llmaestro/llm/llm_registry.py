"""Unified registry for managing LLM models."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING, Type

import yaml
from pydantic import BaseModel, ConfigDict, Field

from llmaestro.config.base import RateLimitConfig

from .models import LLMCapabilities, LLMMetadata, LLMProfile, Provider, ModelFamily, LLMInstance, LLMState
from .credentials import APIKey

if TYPE_CHECKING:
    from .interfaces.base import BaseLLMInterface
    






class LLMRegistry(BaseModel):
    """Registry for managing LLM models and their interfaces.
    
    This registry serves as the single source of truth for:
    1. Model configurations and capabilities
    2. Model-provider relationships
    3. Model lifecycle management
    4. Interface instances
    """
    
    models: Dict[str, LLMInstance] = Field(
        default_factory=dict,
        description="Models in the registry"
    )
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    async def register_model(
        self, 
        state: LLMState,
        credentials: Optional[APIKey] = None,
        interface_class: Optional[Type["BaseLLMInterface"]] = None
    ) -> LLMInstance:
        """Register a model and create its instance.
        
        Args:
            state: Model state to register
            credentials: Optional explicit credentials override
            interface_class: Optional custom interface implementation
        """
        model_credentials = credentials
        if not model_credentials:
            raise ValueError(f"No credentials provided for model {state.model_name} and no credential manager configured")
        # Create instance
        instance = LLMInstance(
            state=state,
            credentials=model_credentials
        )
        
        try:
            # Initialize with interface
            await instance.initialize(interface_class)
            self.models[instance.model_name] = instance
            return instance
            
        except Exception as e:
            logging.error(
                f"Failed to initialize instance for model {state.model_name}: {str(e)}. "
                "Model will not be registered."
            )
            raise


