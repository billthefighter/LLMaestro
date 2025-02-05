"""Tests for the PDFReader application."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from src.applications.pdfreader.pdf_reader import PDFReader, PDFReaderPrompt, PDFReaderResponse
from src.llm.interfaces.base import ImageInput, LLMResponse


# Test Models
class TestInvoiceData(BaseModel):
    """Test model for invoice data."""
    invoice_number: str
    date: datetime
    total: float
    items: Dict[str, float]


# Mock Data
MOCK_RESPONSE_DATA = {
    "extracted_data": {
        "invoice_number": "INV-2024-001",
        "date": "2024-02-04T00:00:00Z",
        "total": 450.0,
        "items": {
            "Widget A": 100.0,
            "Widget B": 150.0,
            "Widget C": 200.0
        }
    },
    "confidence": 0.95,
    "warnings": ["Item C price might be ambiguous"]
}


# Fixtures
@pytest.fixture
def mock_llm():
    """Create a mock LLM interface."""
    mock = AsyncMock()
    mock.process.return_value = LLMResponse(
        content=json.dumps(MOCK_RESPONSE_DATA),
        metadata={"id": "test-id"},
    )
    return mock


@pytest.fixture
def mock_pdf_image():
    """Create a mock PDF image."""
    mock = MagicMock()
    mock.save = MagicMock()
    return mock


@pytest.fixture
def mock_convert_from_path(mock_pdf_image):
    """Mock the convert_from_path function."""
    with patch("src.applications.pdfreader.pdf_reader.convert_from_path") as mock_convert:
        mock_convert.return_value = [mock_pdf_image]
        yield mock_convert


@pytest.fixture
def test_config_path(tmp_path):
    """Create a test config file."""
    config_path = tmp_path / "config.yaml"
    config_data = {
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-latest",
            "api_key": "test-key"
        }
    }
    config_path.write_text(json.dumps(config_data))
    return config_path


@pytest.fixture
def pdf_reader(mock_llm, test_config_path):
    """Create a PDFReader instance with mocked components."""
    with patch("src.applications.pdfreader.pdf_reader.AnthropicLLM") as mock_llm_class:
        mock_llm_class.return_value = mock_llm
        reader = PDFReader(
            output_model=TestInvoiceData,
            config_path=test_config_path
        )
        yield reader


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
    assert response.warnings == ["Item C price might be ambiguous"]
    assert response.extracted_data["total"] == 450.0

    # Invalid confidence
    with pytest.raises(ValueError):
        PDFReaderResponse(
            extracted_data=MOCK_RESPONSE_DATA["extracted_data"],
            confidence=1.5  # Should be between 0 and 1
        )


@pytest.mark.asyncio
async def test_process_pdf_success(pdf_reader, mock_convert_from_path, tmp_path):
    """Test successful PDF processing."""
    # Create a test PDF
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()

    # Process the PDF
    result = await pdf_reader.process_pdf(pdf_path)

    # Verify the result
    assert isinstance(result, PDFReaderResponse)
    assert result.confidence == 0.95
    assert "Widget A" in result.extracted_data["items"]
    assert result.extracted_data["total"] == 450.0


@pytest.mark.asyncio
async def test_process_pdf_file_not_found(pdf_reader):
    """Test handling of non-existent PDF file."""
    with pytest.raises(FileNotFoundError):
        await pdf_reader.process_pdf("nonexistent.pdf")


@pytest.mark.asyncio
async def test_process_pdf_no_pages(pdf_reader, mock_convert_from_path):
    """Test handling of PDF with no pages."""
    mock_convert_from_path.return_value = []

    with pytest.raises(ValueError, match="No images extracted from PDF"):
        await pdf_reader.process_pdf("empty.pdf")


@pytest.mark.asyncio
async def test_process_pdf_multiple_pages(pdf_reader, mock_convert_from_path, mock_pdf_image):
    """Test processing of multi-page PDF."""
    # Mock multiple pages
    mock_convert_from_path.return_value = [mock_pdf_image, mock_pdf_image]

    # Process the PDF
    result = await pdf_reader.process_pdf("multipage.pdf")

    # Verify multiple pages were processed
    assert pdf_reader.llm.process.call_count == 2
    assert isinstance(result, PDFReaderResponse)


@pytest.mark.asyncio
async def test_custom_result_combination(mock_llm, test_config_path):
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
    with patch("src.applications.pdfreader.pdf_reader.AnthropicLLM") as mock_llm_class:
        mock_llm_class.return_value = mock_llm
        reader = CustomPDFReader(
            output_model=TestInvoiceData,
            config_path=test_config_path
        )

        # Process a multi-page PDF
        mock_convert_from_path.return_value = [mock_pdf_image, mock_pdf_image]
        result = await reader.process_pdf("multipage.pdf")

        # Verify custom combination was used
        assert result.confidence == 0.95  # Average of two 0.95 confidences


# Integration Tests
@pytest.mark.integration
@pytest.mark.asyncio
async def test_pdf_reader_integration(tmp_path):
    """Integration test with real PDF and LLM."""
    # Create a test invoice PDF
    pdf_path = tmp_path / "test_invoice.pdf"

    # TODO: Create a real PDF with invoice data
    # For now, we'll just create an empty file
    pdf_path.touch()

    # Initialize reader with real components
    reader = PDFReader(output_model=TestInvoiceData)

    # Process the PDF
    result = await reader.process_pdf(pdf_path)

    # Verify the result
    assert isinstance(result, PDFReaderResponse)
    assert 0 <= result.confidence <= 1
    assert isinstance(result.extracted_data["invoice_number"], str)
    assert isinstance(result.extracted_data["date"], datetime)
    assert isinstance(result.extracted_data["total"], float)
    assert isinstance(result.extracted_data["items"], dict)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pdf_reader_with_real_config():
    """Test PDFReader with real config file."""
    reader = PDFReader(output_model=TestInvoiceData)
    assert reader.api_key is not None
    assert reader.llm is not None
