from typing import Any, Optional
from litellm import acompletion

from .base import BaseLLMInterface, LLMResponse

class OpenAIInterface(BaseLLMInterface):
    """OpenAI-specific implementation of the LLM interface."""
    
    async def process(self, input_data: Any, system_prompt: Optional[str] = None) -> LLMResponse:
        """Process input using OpenAI's API via LiteLLM."""
        try:
            messages = self._format_messages(input_data, system_prompt)
            
            response = await acompletion(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stream=True
            )
            
            return await self._handle_response(response, messages)
            
        except Exception as e:
            return self._handle_error(e) 