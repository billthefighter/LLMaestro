from typing import Any, Dict, Optional
from .base import BaseLLMInterface, LLMResponse
from ..prompts.loader import PromptLoader

# Task-specific system prompts
PDF_ANALYSIS_PROMPT = """You are an expert document analyzer. Your task is to analyze PDF content and categorize it.
Follow these steps:
1. Identify the main topics and themes
2. Determine the document type and purpose
3. Extract key information and insights

Respond in JSON format with this structure:
{
    "category": "main category",
    "subcategories": ["list", "of", "subcategories"],
    "key_points": ["list", "of", "key", "points"],
    "summary": "brief summary"
}"""

CODE_REFACTOR_PROMPT = """You are an expert code refactoring assistant. Your task is to analyze and improve code.
Follow these steps:
1. Identify code smells and potential improvements
2. Suggest refactoring changes while maintaining functionality
3. Consider performance, readability, and maintainability

Respond in JSON format with this structure:
{
    "analysis": {
        "code_smells": ["list", "of", "issues"],
        "improvement_areas": ["list", "of", "areas"]
    },
    "refactoring": {
        "suggested_changes": ["list", "of", "changes"],
        "code_snippets": {
            "original": "original code",
            "refactored": "refactored code"
        }
    }
}"""

LINT_FIX_PROMPT = """You are an expert code linting assistant. Your task is to fix code style issues.
Follow these steps:
1. Analyze each linting error
2. Propose fixes that comply with the style guide
3. Maintain code functionality while fixing style issues

Respond in JSON format with this structure:
{
    "fixes": [
        {
            "error": "error description",
            "line": line_number,
            "original": "original code",
            "fixed": "fixed code",
            "explanation": "why this fix was chosen"
        }
    ]
}"""

class TaskProcessor:
    """Processes specific types of tasks using an LLM interface."""
    
    def __init__(self, llm: BaseLLMInterface, prompt_loader: Optional[PromptLoader] = None):
        self.llm = llm
        self.prompt_loader = prompt_loader or PromptLoader()
    
    async def process_pdf_analysis(self, content: str) -> Dict[str, Any]:
        """Process a PDF analysis task."""
        system_prompt, user_prompt = self.prompt_loader.format_prompt(
            "pdf_analysis",
            content=content
        )
        if not system_prompt or not user_prompt:
            raise ValueError("Failed to load PDF analysis prompt")
            
        response = await self.llm.process(user_prompt, system_prompt=system_prompt)
        return self._parse_json_response(response)
    
    async def process_code_refactor(self, code: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Process a code refactoring task."""
        system_prompt, user_prompt = self.prompt_loader.format_prompt(
            "code_refactor",
            code=code,
            context=context or "No additional context provided."
        )
        if not system_prompt or not user_prompt:
            raise ValueError("Failed to load code refactoring prompt")
            
        response = await self.llm.process(user_prompt, system_prompt=system_prompt)
        return self._parse_json_response(response)
    
    async def process_lint_fix(self, code: str, errors: list[str]) -> Dict[str, Any]:
        """Process a lint fixing task."""
        system_prompt, user_prompt = self.prompt_loader.format_prompt(
            "lint_fix",
            code=code,
            errors=errors
        )
        if not system_prompt or not user_prompt:
            raise ValueError("Failed to load lint fixing prompt")
            
        response = await self.llm.process(user_prompt, system_prompt=system_prompt)
        return self._parse_json_response(response)
    
    def _parse_json_response(self, response: LLMResponse) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        try:
            if isinstance(response.content, str):
                return eval(response.content)  # Safe since we control the LLM prompt format
            return response.content
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}") 