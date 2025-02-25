"""Unified registry for managing LLM models and their providers."""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .capabilities import RangeConfig
from .capability_factory import ModelCapabilityDetectorFactory
from .models import LLMCapabilities, LLMMetadata, LLMProfile, Provider
from .enums import ModelFamily


class LLMRuntime(BaseModel):
    """State information for a model and its provider."""
    registered_at: datetime = Field(
        default_factory=datetime.now,
        description="When the model was registered"
    )
    provider: Provider = Field(
        description="Provider configuration"
    )
    profile: LLMProfile = Field(
        description="Model profile containing capabilities and metadata"
    )
    is_deprecated: bool = Field(
        default=False,
        description="Whether the model is deprecated"
    )
    last_capability_update: Optional[datetime] = Field(
        default=None,
        description="When the model's capabilities were last updated"
    )
    recommended_replacement: Optional[str] = Field(
        default=None,
        description="Name of recommended replacement model if deprecated"
    )
    initialized_at: Optional[datetime] = Field(
        default=None,
        description="When the model's provider was initialized"
    )

    def mark_initialized(self) -> None:
        """Mark the model's provider as initialized."""
        self.initialized_at = datetime.now()

    @property
    def is_initialized(self) -> bool:
        """Check if the model's provider is initialized."""
        return self.initialized_at is not None


class LLMRegistry(BaseModel):
    """Unified registry for managing LLM models and their providers.
    
    This registry serves as the single source of truth for:
    1. Model configurations and capabilities
    2. Provider configurations and credentials
    3. Model-provider relationships
    4. Credential management
    5. Model lifecycle management
    """
    
    credential_manager: Optional[Any] = None
    strict_capability_detection: bool = False
    auto_update_capabilities: bool = True
    _models: Dict[str, LLMRuntime] = {}
    _provider_credentials: Dict[str, str] = {}
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def register_model(
        self, 
        model_name: str, 
        provider: Provider,
        profile: LLMProfile,
    ) -> None:
        """Register a model with its provider configuration."""
        self._models[model_name] = LLMRuntime(
            registered_at=datetime.now(),
            provider=provider,
            profile=profile,
            is_deprecated=profile.metadata.is_deprecated,
            recommended_replacement=profile.metadata.recommended_replacement
        )
        
    def initialize_provider(self, provider_name: str, api_key: str) -> None:
        """Initialize a provider with credentials.
        
        Args:
            provider_name: Name of the provider to initialize
            api_key: API key for authentication
            
        Raises:
            ValueError: If no models found for provider or credential manager not set
        """
        if not self.credential_manager:
            raise ValueError("Credential manager not set")
            
        # Find all models for this provider
        provider_models = [
            (name, state) for name, state in self._models.items()
            if state.provider.name == provider_name
        ]
        
        if not provider_models:
            raise ValueError(f"No models registered for provider {provider_name}")
            
        # Store credential
        self.credential_manager.add_credential(provider_models[0][1].provider, api_key)
        self._provider_credentials[provider_name] = api_key
        
        # Mark all models for this provider as initialized
        for _, state in provider_models:
            state.mark_initialized()
        
    def get_model(self, model_name: str) -> Optional[LLMProfile]:
        """Get a model's configuration.

        Args:
            model_name: Name of the model to retrieve

        Returns:
            LLMProfile if found and provider initialized, None otherwise
        """
        state = self._models.get(model_name)
        if not state:
            return None
            
        # Check if provider is initialized
        if not self.is_provider_initialized(state.provider.name):
            return None
            
        return state.profile
        
    def get_model_by_family(self, family: ModelFamily, model_name: str) -> Optional[LLMProfile]:
        """Get a model by its family and name.

        Args:
            family: Model family to search in
            model_name: Name of the model
            
        Returns:
            LLMProfile if found and provider initialized, None otherwise
        """
        state = self._models.get(model_name)
        if not state or state.profile.family != family:
            return None
            
        if not self.is_provider_initialized(state.provider.name):
            return None
            
        return state.profile
        
    def is_provider_initialized(self, provider_name: str) -> bool:
        """Check if a provider has valid credentials."""
        return provider_name in self._provider_credentials
        
    def get_provider_api_config(self, model_name: str) -> dict:
        """Get complete API configuration for a model.

        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing complete API configuration
            
        Raises:
            ValueError: If model not found or provider not initialized
        """
        state = self._models.get(model_name)
        if not state:
            raise ValueError(f"Unknown model: {model_name}")
            
        if not self.is_provider_initialized(state.provider.name):
            raise ValueError(f"Provider {state.provider.name} not initialized")
            
        return {
            "provider": state.provider.name,
            "name": model_name,
            "api_base": state.provider.api_base,
            "api_key": self._provider_credentials[state.provider.name],
            "capabilities": state.profile.capabilities.model_dump(),
            "rate_limits": state.provider.rate_limits,
        }
        
    def list_models(self) -> List[str]:
        """Get a list of all registered model names."""
        return list(self._models.keys())
        
    def list_models_by_provider(self, provider_name: str) -> List[str]:
        """Get a list of all models for a specific provider."""
        return [
            name for name, state in self._models.items()
            if state.provider.name == provider_name
        ]
        
    def list_initialized_models(self) -> List[str]:
        """Get a list of models with initialized providers."""
        return [
            name for name, state in self._models.items()
            if self.is_provider_initialized(state.provider.name)
        ]
        
    def validate_model(self, model_name: str) -> tuple[bool, Optional[str]]:
        """Validate if a model exists and is usable.

        Args:
            model_name: Name of the model to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        state = self._models.get(model_name)
        if not state:
            return False, f"Unknown model: {model_name}"
            
        if not self.is_provider_initialized(state.provider.name):
            return False, f"Provider {state.provider.name} not initialized"
            
        if state.is_deprecated:
            msg = f"Model {model_name} is deprecated"
            if state.recommended_replacement:
                msg += f". Consider using {state.recommended_replacement} instead"
            return False, msg
            
        return True, None

    async def detect_and_register_model(
        self,
        provider_name: str,
        model_name: str,
        api_key: str
    ) -> LLMProfile:
        """Detects capabilities of a model and registers it.

        Args:
            provider_name: Name of the provider
            model_name: Name of the model
            api_key: API key for capability detection

        Returns:
            LLMProfile for the registered model

        Raises:
            ValueError: If capability detection fails and strict mode is enabled
        """
        logger = logging.getLogger(__name__)

        try:
            capabilities = await ModelCapabilityDetectorFactory.detect_capabilities(
                provider_name,
                model_name,
                api_key
            )
            
            # Ensure required fields are set
            capabilities.name = model_name
            capabilities.family = ModelFamily.from_provider(provider_name)
            if not capabilities.version:
                capabilities.version = "latest"
            if not capabilities.description:
                capabilities.description = f"Auto-detected capabilities for {model_name}"
                
        except Exception as e:
            if self.strict_capability_detection:
                raise ValueError(f"Failed to detect capabilities for {model_name}: {str(e)}") from e
                
            logger.warning(
                f"Capability detection failed for {model_name}, using default capabilities. Error: {str(e)}"
            )
            capabilities = LLMCapabilities(
                name=model_name,
                family=ModelFamily.from_provider(provider_name),
                version="latest",
                description=f"Default capabilities for {model_name}",
                max_context_window=4096,
                typical_speed=50.0,
                input_cost_per_1k_tokens=0.01,
                output_cost_per_1k_tokens=0.02,
                supports_streaming=True,
                temperature=RangeConfig(min_value=0.0, max_value=2.0, default_value=1.0),
                top_p=RangeConfig(min_value=0.0, max_value=1.0, default_value=1.0),
            )

        # Create profile and provider
        profile = LLMProfile(
            capabilities=capabilities,
            metadata=LLMMetadata(
                release_date=datetime.now(),
                min_api_version="2024-02-29",
                is_deprecated=False
            ),
        )
        
        provider = Provider(
            name=provider_name,
            api_base=f"https://api.{provider_name}.com/v1",  # Default API base
            rate_limits={"requests_per_minute": 60},  # Default rate limit
            features=set()
        )
        
        # Register model
        self.register_model(model_name, provider, profile)
        return profile
        
    @classmethod
    def create_default(
        cls,
        credential_manager=None,
        strict_capability_detection: bool = False,
        auto_update_capabilities: bool = True,
    ) -> "LLMRegistry":
        """Create registry with default configurations.
        
        Args:
            credential_manager: Optional credential manager instance
            strict_capability_detection: Whether to enforce strict capability detection
            auto_update_capabilities: Whether to automatically update capabilities
            
        Returns:
            LLMRegistry instance with default configurations loaded
        """
        registry = cls(
            credential_manager=credential_manager,
            strict_capability_detection=strict_capability_detection,
            auto_update_capabilities=auto_update_capabilities
        )
        
        # Load configurations from model library
        library_path = Path(__file__).parent / "model_library"
        if not library_path.exists():
            return registry
            
        for config_file in library_path.glob("*.yaml"):
            try:
                with open(config_file) as f:
                    data = yaml.safe_load(f)
                    
                provider = Provider(**data["provider"])
                
                # Register each model with its provider
                for model_name, model_data in data["models"].items():
                    capabilities = LLMCapabilities(**model_data["capabilities"])
                    metadata = LLMMetadata(**model_data.get("metadata", {}))
                    profile = LLMProfile(
                        capabilities=capabilities,
                        metadata=metadata
                    )
                    registry.register_model(model_name, provider, profile)
                    
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to load config from {config_file}: {str(e)}"
                )
                
        return registry
