from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json

from litellm import completion
from pydantic import BaseModel

from ..core.models import AgentConfig

class LLMResponse(BaseModel):
    """Standardized response from LLM processing."""
    content: str
    metadata: Dict[str, Any] = {}

class BaseLLMInterface(ABC):
    """Base interface for LLM interactions."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        
    @abstractmethod
    async def process(self, input_data: Any, system_prompt: Optional[str] = None) -> LLMResponse:
        """Process input data and return a response."""
        pass

class OpenAIInterface(BaseLLMInterface):
    """OpenAI-specific implementation of the LLM interface."""
    
    async def process(self, input_data: Any, system_prompt: Optional[str] = None) -> LLMResponse:
        """Process input using OpenAI's API via LiteLLM."""
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # Format input data
        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, dict):
            messages.append({"role": "user", "content": json.dumps(input_data, indent=2)})
        else:
            messages.append({"role": "user", "content": str(input_data)})
            
        try:
            response = await completion(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                api_key=self.config.api_key
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                metadata={
                    "model": response.model,
                    "usage": response.usage._asdict() if response.usage else {}
                }
            )
        except Exception as e:
            raise RuntimeError(f"LLM processing failed: {str(e)}")

def create_llm_interface(config: AgentConfig) -> BaseLLMInterface:
    """Factory function to create appropriate LLM interface."""
    if "gpt" in config.model_name.lower() or "openai" in config.model_name.lower():
        return OpenAIInterface(config)
    else:
        raise ValueError(f"Unsupported model: {config.model_name}") 