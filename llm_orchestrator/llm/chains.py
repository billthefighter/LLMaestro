from typing import Any, Dict, Optional
from .base import BaseLLMInterface, LLMResponse
from ..prompts.loader import PromptLoader

class TaskProcessor:
    """Processes tasks using an LLM interface and prompt templates."""
    
    def __init__(self, llm: BaseLLMInterface, prompt_loader: Optional[PromptLoader] = None):
        self.llm = llm
        self.prompt_loader = prompt_loader or PromptLoader()
    
    async def process_task(self, task_type: str, **kwargs) -> Dict[str, Any]:
        """Process any task type using its prompt template."""
        system_prompt, user_prompt = self.prompt_loader.format_prompt(
            task_type,
            **kwargs
        )
        if not system_prompt or not user_prompt:
            raise ValueError(f"Failed to load prompt for task type: {task_type}")
            
        response = await self.llm.process(user_prompt, system_prompt=system_prompt)
        
        # Get the prompt to check expected response format
        prompt = self.prompt_loader.get_prompt(task_type)
        if not prompt:
            raise ValueError(f"No prompt template found for task type: {task_type}")
            
        if prompt.metadata.expected_response.format == "json":
            return self._parse_json_response(response)
        return {"content": response.content, "metadata": response.metadata}
    
    def _parse_json_response(self, response: LLMResponse) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        try:
            if isinstance(response.content, str):
                return eval(response.content)  # Safe since we control the LLM prompt format
            return response.content
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}") 