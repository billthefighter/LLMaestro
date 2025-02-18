"""Tests for the PDFReader application."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from llmaestro.applications.pdfreader.pdf_reader import PDFReader, PDFReaderPrompt, PDFReaderResponse
from llmaestro.llm.interfaces.base import ImageInput, LLMResponse


# Test Models
class TestInvoiceData(BaseModel):
    """Test model for invoice data."""
    invoice_number: str
    date: datetime
    total_amount: float
    vendor_name: str
    items: Optional[Dict[str, float]] = None


# Mock Data
MOCK_RESPONSE_DATA = {
    "extracted_data": {
        "invoice_number": "INV-001",
        "date": "2024-01-01T00:00:00",
        "total_amount": 100.50,
        "vendor_name": "Test Vendor",
        "items": {"item1": 50.25, "item2": 50.25}
    },
    "confidence": 0.95
}


def create_test_pdf(path: Path, num_pages: int = 1):
    """Create a test PDF with the specified number of pages."""
    c = canvas.Canvas(str(path), pagesize=letter)
    for i in range(num_pages):
        c.drawString(100, 750, f"Test Invoice - Page {i+1}")
        c.drawString(100, 700, "Invoice Number: INV-001")
        c.drawString(100, 650, "Date: January 1, 2024")
        c.drawString(100, 600, "Total Amount: $100.50")
        c.drawString(100, 550, "Vendor: Test Vendor")
        if i < num_pages - 1:
            c.showPage()
    c.save()


# Fixtures
@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = AsyncMock()
    mock.process.return_value = MagicMock(content=json.dumps(MOCK_RESPONSE_DATA))
    return mock


@pytest.fixture
def mock_pdf_image():
    """Mock PDF image for testing."""
    mock = MagicMock()
    mock.save.side_effect = lambda buf, format: buf.write(b"test image data")
    return mock


@pytest.fixture
def mock_convert_from_path():
    """Mock convert_from_path function."""
    with patch("src.llmaestro.applications.pdfreader.pdf_reader.convert_from_path") as mock:
        yield mock


@pytest.fixture
def test_config_path(tmp_path):
    """Create a test config file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
llm:
  model: claude-3-sonnet-20240229
  api_key: test-key
prompt:
  system: "You are a helpful assistant that extracts data from invoices."
  human: "Please extract the following information from this invoice image: {fields}"
    """)
    return config_path


@pytest.fixture
def pdf_reader(mock_llm):
    """Create a PDFReader instance for testing."""
    with patch("llmaestro.applications.pdfreader.pdf_reader.AnthropicLLM") as mock_llm_class:
        mock_llm_class.return_value = mock_llm
        reader = PDFReader(output_model=TestInvoiceData)
        return reader


# Unit Tests
def test_pdf_reader_init(pdf_reader):
    """Test PDFReader initialization."""
    assert pdf_reader.output_model == TestInvoiceData
    assert isinstance(pdf_reader.prompt, PDFReaderPrompt)
    assert pdf_reader.llm is not None


def test_pdf_reader_prompt_init():
    """Test PDFReaderPrompt initialization."""
    prompt = PDFReaderPrompt(output_model=TestInvoiceData)
    assert prompt.output_model == TestInvoiceData
    assert "pdf-reader" in prompt.name
    assert prompt.metadata.type == "pdf_analysis"


def test_pdf_reader_response_validation():
    """Test PDFReaderResponse validation."""
    # Valid data
    response = PDFReaderResponse(**MOCK_RESPONSE_DATA)
    assert response.confidence == 0.95

    # Convert extracted_data to TestInvoiceData
    data = TestInvoiceData(**response.extracted_data)
    assert data.total_amount == 100.50
    assert data.invoice_number == "INV-001"
    assert data.vendor_name == "Test Vendor"
    assert data.items == {"item1": 50.25, "item2": 50.25}
    assert data.date == datetime.fromisoformat("2024-01-01T00:00:00")


@pytest.mark.asyncio
async def test_process_pdf_success(pdf_reader, mock_convert_from_path, mock_pdf_image, tmp_path):
    """Test successful PDF processing."""
    pdf_path = tmp_path / "test.pdf"
    create_test_pdf(pdf_path)

    mock_convert_from_path.return_value = [mock_pdf_image]
    result = await pdf_reader.process_pdf(pdf_path)

    assert isinstance(result, PDFReaderResponse)
    data = TestInvoiceData(**result.extracted_data)
    assert isinstance(data, TestInvoiceData)
    assert data.invoice_number == "INV-001"
    assert result.confidence == 0.95
    assert data.total_amount == 100.50


@pytest.mark.asyncio
async def test_process_pdf_file_not_found(pdf_reader):
    """Test handling of non-existent PDF."""
    with pytest.raises(FileNotFoundError):
        await pdf_reader.process_pdf("nonexistent.pdf")


@pytest.mark.asyncio
async def test_process_pdf_no_pages(pdf_reader, mock_convert_from_path, tmp_path):
    """Test handling of PDF with no pages."""
    pdf_path = tmp_path / "empty.pdf"
    create_test_pdf(pdf_path, num_pages=0)

    mock_convert_from_path.return_value = []
    with pytest.raises(ValueError, match="No images extracted from PDF"):
        await pdf_reader.process_pdf(pdf_path)


@pytest.mark.asyncio
async def test_process_pdf_multiple_pages(pdf_reader, mock_convert_from_path, mock_pdf_image, tmp_path):
    """Test processing of multi-page PDF."""
    pdf_path = tmp_path / "multipage.pdf"
    create_test_pdf(pdf_path, num_pages=2)

    mock_convert_from_path.return_value = [mock_pdf_image, mock_pdf_image]
    result = await pdf_reader.process_pdf(pdf_path)

    assert isinstance(result, PDFReaderResponse)
    data = TestInvoiceData(**result.extracted_data)
    assert isinstance(data, TestInvoiceData)
    assert data.invoice_number == "INV-001"
    assert result.confidence == 0.95
    assert data.total_amount == 100.50


@pytest.mark.asyncio
async def test_custom_result_combination(mock_llm, test_config_path, tmp_path, mock_pdf_image):
    """Test custom result combination logic."""
    class CustomPDFReader(PDFReader):
        def _combine_results(self, results: list[Dict[str, Any]]) -> Dict[str, Any]:
            """Custom combination logic that sums confidences."""
            combined_data = {}
            total_confidence = 0.0

            for result in results:
                combined_data.update(result["extracted_data"])
                total_confidence += result["confidence"]

            return {
                "extracted_data": combined_data,
                "confidence": total_confidence / len(results)
            }

    # Create reader with custom combination logic
    with patch("llmaestro.applications.pdfreader.pdf_reader.AnthropicLLM") as mock_llm_class:
        mock_llm_class.return_value = mock_llm
        reader = CustomPDFReader(
            output_model=TestInvoiceData,
            config_path=test_config_path
        )

        # Process a multi-page PDF
        pdf_path = tmp_path / "multipage.pdf"
        create_test_pdf(pdf_path, num_pages=2)

        with patch("llmaestro.applications.pdfreader.pdf_reader.convert_from_path") as mock_convert:
            mock_convert.return_value = [mock_pdf_image, mock_pdf_image]
            result = await reader.process_pdf(pdf_path)

            assert isinstance(result, PDFReaderResponse)
            data = TestInvoiceData(**result.extracted_data)
            assert isinstance(data, TestInvoiceData)
            assert result.confidence == 0.95
            assert data.total_amount == 100.50


# Integration Tests
@pytest.mark.integration
@pytest.mark.asyncio
async def test_pdf_reader_integration(tmp_path):
    """Integration test with real PDF and LLM."""
    # Create a test invoice PDF
    pdf_path = tmp_path / "test_invoice.pdf"
    create_test_pdf(pdf_path)

    # Initialize reader with real components
    reader = PDFReader(output_model=TestInvoiceData)

    # Mock the LLM response for integration test
    mock_response = MagicMock()
    mock_response.content = json.dumps(MOCK_RESPONSE_DATA)

    with patch.object(reader.llm, 'process', return_value=mock_response):
        # Process the PDF
        result = await reader.process_pdf(pdf_path)
        assert isinstance(result, PDFReaderResponse)
        data = TestInvoiceData(**result.extracted_data)
        assert isinstance(data, TestInvoiceData)
        assert data.invoice_number == "INV-001"
        assert result.confidence == 0.95
        assert data.total_amount == 100.50


@pytest.mark.asyncio
async def test_pdf_reader_with_real_config(test_config_path):
    """Test PDFReader with a real config file."""
    reader = PDFReader(
        output_model=TestInvoiceData,
        config_path=test_config_path
    )
    assert reader.prompt is not None
