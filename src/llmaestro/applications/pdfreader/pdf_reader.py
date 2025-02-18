"""PDF reader application for extracting structured data from PDFs using vision capabilities."""

import base64
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import yaml
from pdf2image import convert_from_path
from pydantic import BaseModel, Field

from src.core.models import AgentConfig, BaseResponse
from src.llm.interfaces.base import ImageInput, MediaType
from src.llm.models import ModelRegistry
from src.prompts.base import BasePrompt


class PDFReaderResponse(BaseResponse):
    """Standard response format for PDF reader."""

    extracted_data: Dict[str, Any] = Field(..., description="The structured data extracted from the PDF")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score for the extraction")
    warnings: Optional[List[str]] = Field(default=None, description="Any warnings or issues encountered")


class PDFReaderPrompt(BasePrompt):
    """Prompt for PDF reading and data extraction."""

    output_model: Type[BaseModel]

    def __init__(self, output_model: Type[BaseModel], **data):
        # Load prompt template from YAML
        prompt_path = Path(__file__).parent / "prompt.yaml"
        with open(prompt_path) as f:
            prompt_data = yaml.safe_load(f)

        # Update schema in the prompt data
        schema_str = output_model.model_json_schema()
        prompt_data["metadata"]["expected_response"]["schema"] = json.dumps(schema_str)

        # Add output model
        data.update(prompt_data)
        data["output_model"] = output_model

        super().__init__(**data)

    async def save(self) -> bool:
        """Save method not implemented for this prompt."""
        return True

    @classmethod
    async def load(cls, identifier: str) -> Optional["PDFReaderPrompt"]:
        """Load method not implemented for this prompt."""
        return None


class PDFReader:
    """PDF reader application for extracting structured data."""

    def __init__(
        self,
        output_model: Type[BaseModel],
        api_key: Optional[str] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize the PDF reader.

        Args:
            output_model: The Pydantic model defining the expected output structure
            api_key: Optional API key for the LLM service
            config_path: Optional path to config file (default: config/config.yaml)
        """
        self.output_model = output_model
        self.api_key = api_key
        self.config_path = Path(config_path) if config_path else Path("config/config.yaml")

        # Initialize components
        self._init_llm()
        self.prompt = PDFReaderPrompt(output_model=output_model)

    def _init_llm(self) -> None:
        """Initialize the LLM interface."""
        # Load config
        if not self.api_key:
            with open(self.config_path) as f:
                config_data = yaml.safe_load(f)
            self.api_key = config_data["llm"]["api_key"]

        # Initialize model registry
        self.model_registry = ModelRegistry()
        self.model_registry = ModelRegistry.from_yaml(Path("src/llm/models/claude.yaml"))

        # Create LLM config
        model_name = "claude-3-5-sonnet-latest"
        assert self.model_registry.get_model(model_name) is not None, f"Model {model_name} not found in registry"

        self.llm_config = AgentConfig(
            provider="anthropic",
            model_name=model_name,
            api_key=self.api_key,
            max_tokens=1024,
            temperature=0.7,
        )

        # Create LLM interface
        provider_class = self.model_registry.get_provider_class(self.llm_config.provider)
        self.llm = provider_class(config=self.llm_config, model_registry=self.model_registry)

    async def process_pdf(self, pdf_path: Union[str, Path]) -> PDFReaderResponse:
        """Process a PDF file and extract structured data.

        Args:
            pdf_path: Path to the PDF file to process

        Returns:
            PDFReaderResponse containing the extracted data and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Convert PDF to images
        images = convert_from_path(pdf_path)
        if not images:
            raise ValueError(f"No images extracted from PDF: {pdf_path}")

        # Process each page
        results = []
        for i, image in enumerate(images):
            # Convert to JPEG
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_data = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

            # Create image input
            image_input = ImageInput(content=img_data, media_type=MediaType.JPEG, file_name=f"page_{i+1}.jpg")

            # Prepare prompt variables
            schema_str = self.output_model.model_json_schema()
            variables = {"output_schema": json.dumps(schema_str, indent=2)}

            # Render prompt
            system_prompt, user_prompt, attachments = self.prompt.render(**variables)

            # Format messages with attachments
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "attachments": attachments},
            ]

            # Process with LLM
            response = await self.llm.process(input_data=messages, images=[image_input])

            # Parse response
            if response.content:
                result = json.loads(response.content)
                results.append(result)

        # Combine results from all pages
        if len(results) == 1:
            final_result = results[0]
        else:
            # Combine data from multiple pages (implement your own logic here)
            final_result = self._combine_results(results)

        # Convert to response model
        return PDFReaderResponse(**final_result)

    def _combine_results(self, results: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple pages.

        Override this method to implement custom logic for combining results.
        """
        # Default implementation: take highest confidence result
        return max(results, key=lambda x: x.get("confidence", 0))


async def main():
    """Example usage of the PDF reader."""

    # Define your output model
    class SampleData(BaseModel):
        title: str
        content: Dict[str, Any]
        date: datetime

    # Create PDF reader
    reader = PDFReader(output_model=SampleData)

    # Process a PDF
    result = await reader.process_pdf("path/to/your/document.pdf")
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
