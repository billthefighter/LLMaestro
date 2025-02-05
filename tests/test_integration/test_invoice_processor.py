import json
import pytest
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel
import yaml

from src.prompts.base import BasePrompt
from src.prompts.types import VersionInfo, PromptMetadata, ResponseFormat
from src.llm.interfaces.anthropic import AnthropicLLM
from src.core.models import AgentConfig
from src.llm.models import ModelRegistry
from src.llm.interfaces.base import ImageInput, MediaType
from PIL import Image, ImageDraw, ImageFont
import base64
import io
from pdf2image import convert_from_path

# Test Model for Invoice Data
class InvoiceData(BaseModel):
    invoice_number: str
    line_items: Dict[str, float]  # item name -> price
    total_value: float
    invoice_date: datetime

# Test Prompt for Invoice Processing
class InvoicePrompt(BasePrompt):
    """Prompt for processing invoice images."""

    output_model: type[BaseModel]

    def __init__(self, output_model: type[BaseModel], **data):
        data.update({
            "name": "invoice-processor",
            "description": "Process invoice images and extract structured data",
            "system_prompt": "You are an expert at processing invoices and extracting structured data. Extract the requested information in the exact format specified.",
            "user_prompt": "Please analyze this invoice image and extract the following information in JSON format matching this schema: {output_schema}",
            "metadata": PromptMetadata(
                type="invoice_processing",
                expected_response=ResponseFormat(
                    format="json",
                    schema=json.dumps(output_model.model_json_schema())
                ),
                tags=["test", "invoice", "vision"]
            ),
            "current_version": VersionInfo(
                number="1.0.0",
                timestamp=datetime.now(),
                author="Test Author",
                description="Initial version",
                change_type="initial"
            ),
            "output_model": output_model
        })
        super().__init__(**data)

    async def save(self) -> bool:
        """Mock save method."""
        return True

    @classmethod
    async def load(cls, identifier: str) -> Optional["InvoicePrompt"]:
        """Mock load method."""
        return None

@pytest.fixture
def model_registry():
    """Load the model registry."""
    registry = ModelRegistry()
    registry = ModelRegistry.from_yaml(Path("src/llm/models/claude.yaml"))
    print("Available models in registry:", list(registry._models.keys()))  # Debug print
    return registry

@pytest.fixture
def llm_config(model_registry):
    """Create test LLM configuration."""
    model_name = "claude-3-5-sonnet-latest"  # Use the latest model from registry
    assert model_registry.get_model(model_name) is not None, f"Model {model_name} not found in registry"

    # Load API key from config
    with open("config/config.yaml") as f:
        config_data = yaml.safe_load(f)
    api_key = config_data["llm"]["api_key"]

    return AgentConfig(
        provider="anthropic",
        model_name=model_name,
        api_key=api_key,
        max_tokens=1024,
        temperature=0.7
    )

@pytest.fixture
def anthropic_llm(llm_config, model_registry):
    """Create Anthropic LLM instance."""
    return AnthropicLLM(config=llm_config, model_registry=model_registry)

@pytest.fixture
def invoice_prompt():
    """Create test invoice prompt."""
    return InvoicePrompt(output_model=InvoiceData)

@pytest.fixture
def sample_invoice_image(tmp_path):
    """Create a sample invoice image for testing."""
    # Create a new image with a white background
    img = Image.new('RGB', (800, 1000), color='white')
    draw = ImageDraw.Draw(img)

    # Add invoice content
    draw.text((50, 50), "INVOICE", fill='black', font=None, font_size=24)
    draw.text((50, 100), "Invoice #: INV-2024-001", fill='black', font=None)
    draw.text((50, 130), "Date: 2024-02-04", fill='black', font=None)

    # Add line items
    draw.text((50, 200), "Line Items:", fill='black', font=None)
    draw.text((70, 230), "1. Widget A - $100.00", fill='black', font=None)
    draw.text((70, 260), "2. Widget B - $150.00", fill='black', font=None)
    draw.text((70, 290), "3. Widget C - $200.00", fill='black', font=None)

    # Add total
    draw.text((50, 350), "Total: $450.00", fill='black', font=None)

    # Save the image as PDF
    img_path = tmp_path / "sample-invoice.pdf"
    img.save(img_path, format='PDF')
    return img_path

@pytest.mark.integration
@pytest.mark.asyncio
async def test_invoice_processing(invoice_prompt, anthropic_llm, sample_invoice_image, tmp_path):
    """
    Test invoice processing with vision capabilities.

    This test:
    1. Loads a sample invoice image
    2. Converts it to JPG
    3. Processes it with Claude to extract structured data
    4. Validates the response against our schema
    """
    # Convert PDF to JPG
    pdf_path = sample_invoice_image
    jpg_path = tmp_path / "sample-invoice.jpg"

    # Convert PDF to JPG using pdf2image
    images = convert_from_path(pdf_path)
    images[0].save(jpg_path, 'JPEG')

    # Read and encode the image
    with open(jpg_path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode('utf-8')

    # Create ImageInput object
    image_input = ImageInput(
        content=img_data,
        media_type=MediaType.JPEG,
        file_name="sample-invoice.jpg"
    )

    # Prepare the prompt with schema
    schema_str = InvoiceData.model_json_schema()
    variables = {"output_schema": json.dumps(schema_str, indent=2)}

    # Render the prompt
    system_prompt, user_prompt = invoice_prompt.render(**variables)

    # Format messages for LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Process with LLM
    response = await anthropic_llm.process(
        input_data=messages,
        images=[image_input]
    )

    # Parse and validate response
    assert response is not None, "Response should not be None"
    assert response.content is not None, "Response content should not be None"

    try:
        result_dict = json.loads(response.content)
        result = InvoiceData(**result_dict)
    except Exception as e:
        pytest.fail(f"Failed to parse response as InvoiceData: {e}")

    # Validate the extracted data
    assert isinstance(result.invoice_number, str), "Invoice number should be a string"
    assert isinstance(result.line_items, dict), "Line items should be a dictionary"
    assert isinstance(result.total_value, float), "Total value should be a float"
    assert isinstance(result.invoice_date, datetime), "Invoice date should be a datetime"

    # Validate line items
    for item_name, price in result.line_items.items():
        assert isinstance(item_name, str), "Item name should be a string"
        assert isinstance(price, float), "Item price should be a float"

    # Validate total matches sum of line items
    calculated_total = sum(result.line_items.values())
    assert abs(calculated_total - result.total_value) < 0.01, "Total should match sum of line items"
