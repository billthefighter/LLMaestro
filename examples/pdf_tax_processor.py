import asyncio
import csv
import os
import base64
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import tempfile
from dateutil import parser

from pydantic import BaseModel, Field, field_validator
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError

from llmaestro.core.attachments import FileAttachment
from llmaestro.llm.responses import ResponseFormat, ResponseFormatType
from llmaestro.llm.enums import MediaType
from llmaestro.prompts.memory import MemoryPrompt
from llmaestro.prompts.types import PromptMetadata
from llmaestro.llm.llm_registry import LLMRegistry, LLMInstance
from llmaestro.core.models import LLMResponse
from llmaestro.llm.schema_utils import schema_to_json
from llmaestro.agents.agent_pool import AgentPool
from llmaestro.core.orchestrator import Orchestrator
from llmaestro.core.conversations import get_detailed_conversation_dump, ConversationGraph, ConversationContext
from llmaestro.prompts.base import BasePrompt
from llmaestro.core.logging_config import configure_logging

# Configure root logger first
logger = configure_logging(level=logging.DEBUG, module_name="pdf_tax_processor")

# Configure OpenAI interface logger specifically
openai_logger = logging.getLogger("interface")
openai_logger.setLevel(logging.DEBUG)

# Configure httpx logger for API calls
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.DEBUG)

class TaxableItem(BaseModel):
    """Represents a single taxable item from a receipt."""

    model_config = {"arbitrary_types_allowed": True}

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
        """Parse date string into datetime object."""
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                # First try parsing as ISO format with time
                return datetime.fromisoformat(v).date()
            except ValueError:
                try:
                    # Then try parsing with dateutil for other formats
                    return parser.parse(v).date()
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Unable to parse date: {v}") from e
        raise ValueError(f"Unable to parse date: {v}")

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

    model_config = {"arbitrary_types_allowed": True}
    items: List[TaxableItem]

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, 'model_dump'):
            # Check if it's an instance or class
            if not isinstance(obj, type):
                return obj.model_dump()
            return str(obj)
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

class PDFTaxProcessor:
    """Process PDF files containing tax-related purchases and generate a CSV report."""

    def __init__(self, input_dir: str, output_file: str, llm_registry: LLMRegistry, model_name: str):
        """Initialize the PDF tax processor.

        Args:
            input_dir: Directory containing PDF files to process
            output_file: Path to output CSV file
            llm_registry: LLM registry for managing model instances
            model_name: Name of the model to use for processing
        """
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.temp_dir = Path(input_dir) / "temp_images"
        self.conversation_dir = Path(input_dir) / "conversation_dumps"

        # Initialize LLM instance and orchestrator
        self.llm_registry = llm_registry
        self.model_name = model_name
        self.llm = None
        self.agent_pool = AgentPool(llm_registry=llm_registry, default_model_name=model_name)
        self.orchestrator = Orchestrator(agent_pool=self.agent_pool)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Create output directories
        self.conversation_dir.mkdir(exist_ok=True)

    async def initialize(self) -> None:
        """Initialize the processor components."""
        # Create LLM instance
        self.llm = await self.llm_registry.create_instance(self.model_name)

        # Check if the model supports vision capabilities
        if not self.llm.state.profile.capabilities.supports_vision:
            self.logger.warning(
                f"Model '{self.model_name}' does not support vision capabilities. "
                f"PDF processing may not work correctly as image attachments will be ignored."
            )

    def _convert_pdf_to_png(self, pdf_path: Path) -> List[Path]:
        """Convert a PDF file to PNG images.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of paths to the generated PNG files

        Raises:
            ValueError: If the PDF is empty or corrupted
            FileNotFoundError: If generated PNG files are not found
        """
        # Validate PDF file
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if pdf_path.stat().st_size == 0:
            raise ValueError(f"PDF file is empty: {pdf_path}")

        # Create a temporary directory in the instance temp_dir
        temp_dir_path = Path(tempfile.mkdtemp(dir=self.temp_dir))
        self.logger.info(f"Converting {pdf_path.name} to PNG in {temp_dir_path}")

        try:
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=200,
                output_folder=str(temp_dir_path),
                fmt="png",
                paths_only=False  # Get PIL images instead of paths
            )

            if not images:
                raise ValueError(f"No pages found in PDF: {pdf_path}")

            # Save images with controlled naming
            png_files = []
            for i, image in enumerate(images, 1):
                png_path = temp_dir_path / f"page_{i:03d}.png"
                image.save(str(png_path))
                if not png_path.exists():
                    raise FileNotFoundError(f"Failed to save PNG file: {png_path}")
                png_files.append(png_path)

            self.logger.info(f"Generated {len(png_files)} PNG files for {pdf_path.name}")
            return png_files

        except PDFPageCountError as e:
            self.logger.error(f"Error getting page count for {pdf_path.name}: {str(e)}")
            raise ValueError(f"Unable to read PDF file (possibly corrupted): {pdf_path}") from e
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

    def _save_conversation_dump(self, conversation_id: str, pdf_name: str, conversation: 'ConversationGraph') -> Path:
        """Save a detailed dump of the conversation to a JSON file.

        Args:
            conversation_id: ID of the conversation
            pdf_name: Name of the PDF being processed
            conversation: The conversation graph to dump

        Returns:
            Path to the saved dump file
        """
        dump = get_detailed_conversation_dump(conversation)

        # Add some additional metadata
        dump["pdf_file"] = pdf_name
        dump["processing_timestamp"] = datetime.now()

        # Create filename with timestamp to avoid collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_file = self.conversation_dir / f"{pdf_name}_{timestamp}_conversation.json"

        with open(dump_file, 'w', encoding='utf-8') as f:
            json.dump(dump, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)

        self.logger.info(f"Saved conversation dump to {dump_file}")
        return dump_file

    async def _process_single_pdf(self, pdf_path: Path) -> Tuple[List[TaxableItem], Path]:
        """Process a single PDF file using the orchestrator for conversation management."""
        all_items: List[TaxableItem] = []

        # Create initial prompt
        initial_prompt = MemoryPrompt(
            name="tax_extraction_init",
            description="Initialize tax document processing",
            system_prompt="You are processing a multi-page tax document. Maintain context across pages.",
            user_prompt="Beginning tax document processing.",
            metadata=PromptMetadata(type="tax_extraction_init")
        )

        # Create conversation through orchestrator
        conversation = await self.orchestrator.create_conversation(
            name=f"tax_processing_{pdf_path.stem}",
            initial_prompt=initial_prompt,
            metadata={"source_file": str(pdf_path)}
        )

        try:
            # Convert PDF to images
            png_files = self._convert_pdf_to_png(pdf_path)

            # Process each page
            for i, png_path in enumerate(png_files, 1):
                image_attachment = self._create_image_attachment(png_path)

                # Create page-specific prompt
                prompt = self._create_pdf_prompt(
                    image_attachment,
                    self._create_response_format(),
                    page_number=i,
                    total_pages=len(png_files)
                )

                # Execute prompt through orchestrator
                response_id = await self.orchestrator.execute_prompt(
                    conversation=conversation,
                    prompt=prompt
                )

                # Get response and process items
                response_node = conversation.nodes[response_id]
                if isinstance(response_node.content, LLMResponse) and response_node.content.success:
                    try:
                        self.logger.debug(f"Raw response content: {response_node.content.content}")
                        # When using OpenAI's parse endpoint, content will already be JSON
                        items_data = json.loads(response_node.content.content)
                        items = TaxableItems.model_validate(items_data).items
                        all_items.extend(items)
                        self.logger.info(f"Extracted {len(items)} items from page {i}")
                    except Exception as e:
                        self.logger.error(f"Error parsing items from page {i}: {e}")

                # Clean up PNG file
                png_path.unlink()

            # Save conversation dump
            dump_file = self._save_conversation_dump(
                conversation.id,
                pdf_path.stem,
                conversation
            )

            return all_items, dump_file

        except Exception as e:
            self.logger.error(f"Error processing {pdf_path.name}: {str(e)}")
            raise

    def _create_pdf_prompt(
        self,
        pdf_attachment: FileAttachment,
        response_format: ResponseFormat,
        page_number: int,
        total_pages: int
    ) -> MemoryPrompt:
        """Create a MemoryPrompt for PDF tax data extraction with conversation context."""
        return MemoryPrompt(
            name="tax_extractor",
            description="Extract tax-related purchase data from PDF",
            system_prompt=f"""You are an expert at extracting financial and tax-related data from receipts and invoices.
Processing page {page_number} of {total_pages}.
Extract all purchase items from this page, including dates, amounts, and categories.
If this is not the first page, ensure consistency with previously extracted items.
Group related items together and provide clear descriptions.
If uncertain about any value, add a note explaining the uncertainty.
It is possible that the page is not a receipt or invoice, and in that case, you should return None.
Return the data in the specified JSON format.
IMPORTANT: Return ONLY the raw JSON without any markdown formatting or code blocks.""",
            user_prompt="Please extract all purchase items from this page with their details.",
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
        """Process all PDF files in the input directory using conversation-based approach."""
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Get list of PDF files
        pdf_files = [f for f in self.input_dir.glob("*.pdf")]
        if not pdf_files:
            self.logger.warning("No PDF files found in the input directory")
            return

        all_items: List[TaxableItem] = []
        conversation_dumps: List[Path] = []

        # Process each PDF
        for pdf_path in pdf_files:
            self.logger.info(f"Processing {pdf_path.name}...")
            items, dump_file = await self._process_single_pdf(pdf_path)
            all_items.extend(items)
            conversation_dumps.append(dump_file)

        # Clean up temp directory
        if self.temp_dir.exists():
            # Remove all files and directories in the temp directory
            for temp_file in self.temp_dir.rglob('*'):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir():
                        temp_file.rmdir()
                except Exception as e:
                    self.logger.error(f"Failed to remove {temp_file}: {e}")

            # Now remove the temp directory itself
            try:
                self.temp_dir.rmdir()
            except Exception as e:
                self.logger.error(f"Failed to remove temp directory {self.temp_dir}: {e}")

        # Write results
        self._write_to_csv(all_items)

        # Create a summary of all conversations
        summary_file = self.conversation_dir / f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary = {
            "processing_timestamp": datetime.now().isoformat(),
            "total_pdfs_processed": len(pdf_files),
            "total_items_extracted": len(all_items),
            "conversation_dumps": [str(dump) for dump in conversation_dumps],
            "output_csv": str(self.output_file)
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Processing complete. Results written to {self.output_file}")
        self.logger.info(f"Conversation dumps saved in {self.conversation_dir}")
        self.logger.info(f"Processing summary saved to {summary_file}")


async def main():
    """Example usage of the PDF tax processor."""
    # Set up directories
    input_dir = "tax_receipts"
    output_file = "tax_items.csv"

    # Initialize LLM registry with credentials
    llm_registry = LLMRegistry()
    model_name = "gpt-4o-mini-2024-07-18"  # Use the model we know works with beta parsing

    # Here you would register your model and credentials
    # await llm_registry.register_model(state=your_state, interface_class=your_interface, credentials=your_credentials)

    # Create processor and run
    try:
        processor = PDFTaxProcessor(input_dir, output_file, llm_registry, model_name)
        await processor.initialize()  # Make sure to call initialize
        await processor.process_pdfs()
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        logger.error("Please install required packages: pip install pdf2image")
    except Exception as e:
        logger.error(f"Error processing PDFs: {e}")


if __name__ == "__main__":
    asyncio.run(main())
