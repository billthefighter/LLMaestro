import asyncio
import csv
import os
import base64
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import tempfile

from pydantic import BaseModel, Field, field_validator
from pdf2image import convert_from_path

from llmaestro.core.attachments import FileAttachment
from llmaestro.llm.responses import ResponseFormat, ResponseFormatType
from llmaestro.llm.enums import MediaType
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.prompts.types import PromptMetadata
from llmaestro.llm.llm_registry import LLMRegistry, LLMInstance
from llmaestro.core.models import LLMResponse
from llmaestro.llm.schema_utils import schema_to_json


class TaxableItem(BaseModel):
    """Represents a single taxable item from a receipt."""

    date: datetime = Field(
        description="The date of the purchase/transaction in any standard format (e.g. YYYY-MM-DD, MM/DD/YYYY)"
    )
    category: str = Field(
        description="The expense category of the item (e.g. Office Supplies, Travel, Equipment)"
    )
    cost: float = Field(
        description="The total cost of the item including tax in decimal format"
    )
    supplier: str = Field(
        description="The name of the vendor or supplier who provided the item"
    )
    source_file: str = Field(
        description="The filename of the receipt/invoice this item was extracted from"
    )
    description: str = Field(
        description="A clear description of what was purchased"
    )
    tax_code: Optional[str] = Field(
        default="",
        description="The applicable tax code for this item if available"
    )
    notes: Optional[str] = Field(
        default="",
        description="Any additional notes or clarifications about this item"
    )

    @field_validator("date", mode="before")
    def parse_date(cls, v):
        """Parse date from string if needed."""
        if isinstance(v, str):
            # Try common date formats
            formats = [
                "%Y-%m-%d",
                "%m/%d/%Y",
                "%d/%m/%Y",
                "%B %d, %Y",
                "%d-%m-%Y",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse date: {v}")
        return v

    def to_csv_dict(self) -> Dict[str, str]:
        """Convert to CSV-friendly dictionary."""
        return {
            "date": self.date.strftime("%Y-%m-%d"),
            "category": self.category,
            "cost": str(self.cost),
            "supplier": self.supplier,
            "source_file": self.source_file,
            "description": self.description,
            "tax_code": self.tax_code or "",
            "notes": self.notes or "",
        }

class TaxableItems(BaseModel):
    """Represents a list of taxable items."""

    items: List[TaxableItem]

class PDFTaxProcessor:
    """Process PDF files containing tax-related purchases and generate a CSV report."""

    def __init__(self, input_dir: str, output_file: str, llm: LLMInstance):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.llm = llm
        self.temp_dir = Path(input_dir) / "temp_images"

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def _convert_pdf_to_png(self, pdf_path: Path) -> List[Path]:
        """Convert PDF pages to PNG images.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of paths to generated PNG files
        """
        try:
            # Create temp directory if it doesn't exist
            self.temp_dir.mkdir(exist_ok=True)

            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=300,  # High DPI for better OCR
                fmt="png"
            )

            # Save images and collect paths
            png_files = []
            for i, image in enumerate(images, 1):
                png_path = self.temp_dir / f"{pdf_path.stem}_page_{i}.png"
                image.save(str(png_path))
                png_files.append(png_path)

            self.logger.info(f"Converted {pdf_path.name} to {len(png_files)} PNG files")
            return png_files

        except Exception as e:
            self.logger.error(f"Error converting {pdf_path.name} to PNG: {str(e)}")
            raise

    def _create_image_attachment(self, file_path: Path) -> FileAttachment:
        """Create a FileAttachment from an image file."""
        with open(file_path, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        media_type = MediaType.from_file_extension(str(file_path))
        return FileAttachment(
            content=content,
            media_type=media_type,
            file_name=file_path.name
        )

    def _create_response_format(self) -> ResponseFormat:
        """Create a response format for tax item extraction."""
        return ResponseFormat.from_pydantic_model(
            model=TaxableItems,
            convert_to_json_schema=False,
            format_type=ResponseFormatType.PYDANTIC
        )

    def _create_pdf_prompt(self, pdf_attachment: FileAttachment, response_format: ResponseFormat) -> MemoryPrompt:
        """Create a MemoryPrompt for PDF tax data extraction."""
        return MemoryPrompt(
            name="tax_extractor",
            description="Extract tax-related purchase data from PDF",
            system_prompt="""You are an expert at extracting financial and tax-related data from receipts and invoices.
Extract all purchase items from the document, including dates, amounts, and categories.
Group related items together and provide clear descriptions.
If uncertain about any value, add a note explaining the uncertainty.
Return the data in the specified JSON format.
IMPORTANT: Return ONLY the raw JSON without any markdown formatting or code blocks.""",
            user_prompt="Please extract all purchase items from this document with their details.",
            metadata=PromptMetadata(
                type="tax_extraction",
                tags=["finance", "tax", "receipts", "data_extraction"]
            ),
            attachments=[pdf_attachment],
            expected_response=response_format
        )

    def _write_to_csv(self, items: List[TaxableItem]) -> None:
        """Write extracted items to a CSV file."""
        if not items:
            self.logger.warning("No items to write to CSV")
            return

        fieldnames = ["date", "category", "cost", "supplier", "source_file", "description", "tax_code", "notes"]
        with open(self.output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in items:
                writer.writerow(item.to_csv_dict())
        self.logger.info(f"Successfully wrote {len(items)} items to {self.output_file}")

    async def process_pdfs(self) -> None:
        """Process all PDF files in the input directory."""
        # Get list of PDF files
        pdf_files = [f for f in self.input_dir.glob("*.pdf")]
        if not pdf_files:
            self.logger.warning("No PDF files found in the input directory")
            return

        response_format = self._create_response_format()
        all_items: List[TaxableItem] = []

        # Process each PDF
        for pdf_path in pdf_files:
            self.logger.info(f"Processing {pdf_path.name}...")

            try:
                # Convert PDF to PNG
                png_files = self._convert_pdf_to_png(pdf_path)

                # Process each PNG
                for png_path in png_files:
                    # Create attachment and prompt
                    image_attachment = self._create_image_attachment(png_path)
                    prompt = self._create_pdf_prompt(image_attachment, response_format)

                    # Process with LLM
                    response = await self.llm.interface.process(prompt)

                    if not response.success:
                        self.logger.error(f"Failed to process {png_path.name}: {response.error}")
                        continue

                    # Parse items
                    try:
                        items_data = json.loads(response.content)
                        items = TaxableItems.model_validate(items_data).items
                        all_items.extend(items)
                        self.logger.info(f"Extracted {len(items)} items from {png_path.name}")
                    except Exception as e:
                        self.logger.error(f"Error parsing items from {png_path.name}: {e}")
                        continue

                    # Clean up PNG file
                    png_path.unlink()

            except Exception as e:
                self.logger.error(f"Error processing {pdf_path.name}: {e}")
                continue

        # Clean up temp directory
        if self.temp_dir.exists():
            self.temp_dir.rmdir()

        # Write results
        self._write_to_csv(all_items)
        self.logger.info(f"Processing complete. Results written to {self.output_file}")


async def main():
    """Example usage of the PDF tax processor.

    Before running this script, ensure you have the required dependencies:
    pip install pdf2image  # For PDF to PNG conversion

    Also ensure you have poppler installed:
    - On macOS: brew install poppler
    - On Ubuntu: apt-get install poppler-utils
    - On Windows: Download from: http://blog.alivate.com.au/poppler-windows/
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Set up directories
    input_dir = "tax_receipts"
    output_file = "tax_items.csv"

    # Initialize LLM
    llm_registry = LLMRegistry()  # Add your configuration here
    model_name = "gpt-4o-mini-2024-07-18"
    llm = await llm_registry.create_instance(model_name)

    # Create processor and run
    try:
        processor = PDFTaxProcessor(input_dir, output_file, llm)
        await processor.process_pdfs()
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        logger.error("Please install required packages: pip install pdf2image")
    except Exception as e:
        logger.error(f"Error processing PDFs: {e}")


if __name__ == "__main__":
    asyncio.run(main())
