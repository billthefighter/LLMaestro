"""Test invoice data extraction using LLM."""

import pytest
from pathlib import Path
import json
from typing import cast, Dict, Any
from llmaestro.core.attachments import FileAttachment, BaseAttachment
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.prompts.types import ResponseFormat, PromptMetadata
from llmaestro.llm.llm_registry import LLMRegistry
from llmaestro.core.models import LLMResponse
from llmaestro.llm.responses import ResponseFormatType
from llmaestro.llm.enums import MediaType
import base64

@pytest.fixture
def sample_invoice_path() -> Path:
    """Get the path to the sample invoice PDF."""
    path = Path(__file__).parent.parent / "test_data" / "sample-invoice.pdf"
    assert path.exists(), f"Sample invoice not found at {path}"
    return path

@pytest.fixture
def invoice_attachment(sample_invoice_path: Path) -> FileAttachment:
    """Create a FileAttachment from the sample invoice."""
    with open(sample_invoice_path, "rb") as f:
        content = base64.b64encode(f.read()).decode()
    media_type = MediaType.from_file_extension(str(sample_invoice_path))
    return FileAttachment(
        content=content,
        media_type=media_type,
        file_name=sample_invoice_path.name
    )

@pytest.fixture
def invoice_response_format() -> ResponseFormat:
    """Create a response format for invoice data extraction."""
    schema = {
        "type": "object",
        "properties": {
            "total": {
                "type": "string",
                "description": "The total amount before VAT in euro"
            },
            "vat_percentage": {
                "type": "string",
                "description": "The VAT percentage"
            },
            "vat_total": {
                "type": "string",
                "description": "The total VAT amount in euro"
            },
            "gross_total": {
                "type": "string",
                "description": "The gross amount including VAT in euro"
            }
        },
        "required": ["total", "vat_percentage", "vat_total", "gross_total"]
    }

    return ResponseFormat(
        format=ResponseFormatType.JSON,
        schema=json.dumps(schema)
    )

@pytest.fixture
def invoice_prompt(invoice_attachment: FileAttachment, invoice_response_format: ResponseFormat) -> MemoryPrompt:
    """Create a MemoryPrompt for invoice data extraction."""
    return MemoryPrompt(
        name="invoice_extractor",
        description="Extract financial data from invoice",
        system_prompt="""You are an expert at extracting financial data from invoices.
Extract the requested information accurately, maintaining the exact numerical values and currency format as shown in the invoice.
Return the data in the specified JSON format.""",
        user_prompt="Please extract the parameters requested in the expected response format.",
        metadata=PromptMetadata(
            type="invoice_extraction",
            expected_response=invoice_response_format,
            tags=["finance", "invoice", "data_extraction"]
        ),
        attachments=[invoice_attachment]
    )

def test_sample_invoice_path_exists(sample_invoice_path: Path):
    """Test that the sample invoice file exists."""
    assert sample_invoice_path.exists()
    assert sample_invoice_path.is_file()
    assert sample_invoice_path.suffix == '.pdf'

def test_invoice_attachment_creation(invoice_attachment: FileAttachment):
    """Test that the invoice attachment is created correctly."""
    assert invoice_attachment is not None
    assert isinstance(invoice_attachment, FileAttachment)
    assert invoice_attachment.media_type.value == 'application/pdf'
    assert invoice_attachment.content is not None
    assert invoice_attachment.get_size() > 0

def test_invoice_response_format_creation(invoice_response_format: ResponseFormat):
    """Test that the response format is created correctly."""
    assert invoice_response_format is not None
    assert invoice_response_format.format == ResponseFormatType.JSON
    assert invoice_response_format.schema is not None

    # Verify schema can be parsed
    schema_dict: Dict[str, Any] = json.loads(invoice_response_format.schema)
    assert schema_dict["type"] == "object"
    assert "total" in schema_dict["properties"]
    assert "vat_percentage" in schema_dict["properties"]
    assert "vat_total" in schema_dict["properties"]
    assert "gross_total" in schema_dict["properties"]

def test_invoice_prompt_creation(invoice_prompt: MemoryPrompt):
    """Test that the invoice prompt is created correctly."""
    assert invoice_prompt is not None
    assert invoice_prompt.name == "invoice_extractor"
    assert len(invoice_prompt.attachments) == 1
    assert isinstance(invoice_prompt.attachments[0], FileAttachment)
    assert invoice_prompt.metadata is not None
    assert invoice_prompt.metadata.expected_response is not None

@pytest.mark.asyncio
@pytest.mark.integration
async def test_invoice_data_extraction(test_settings, llm_registry: LLMRegistry, invoice_prompt: MemoryPrompt):
    """Test extracting data from a sample invoice using real LLM."""
    if not test_settings.use_real_tokens:
        pytest.skip("Skipping test that requires LLM API tokens")
    # Arrange
    model_name = llm_registry.get_registered_models()[0]
    llm_instance = await llm_registry.create_instance(model_name)

    # Act
    response = await llm_instance.interface.process(invoice_prompt)

    # Assert
    assert isinstance(response, LLMResponse)
    assert response.success is True
    assert isinstance(response.content, str)

    # Parse the JSON response
    data = json.loads(response.content)

    # Verify the extracted values
    assert data["total"] == "381.12 euro"
    assert data["vat_percentage"] == "19%"
    assert data["vat_total"] == "72.41 euro"
    assert data["gross_total"] == "453.52 euro"
