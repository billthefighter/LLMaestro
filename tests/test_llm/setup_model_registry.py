"""Setup script for registering models before running connectivity tests."""
import os
from llmaestro.llm.models import register_all_models

def setup_model_registry():
    """Register all models with their respective API keys."""
    register_all_models(
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )

if __name__ == "__main__":
    setup_model_registry()
