import asyncio
import csv
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from src.agents.agent_pool import Agent, AgentPool
from src.core.models import AgentConfig, SubTask, Task
from src.core.task_manager import FileStrategy, TaskManager
from src.llm.chains import ChainStep, SequentialChain
from src.llm.interfaces import BaseLLMInterface, ImageInput, LLMResponse, MediaType
from src.prompts.loader import PromptLoader


class TaxableItem(BaseModel):
    """Represents a single taxable item from a receipt."""

    date: datetime
    category: str
    cost: float
    supplier: str
    source_file: str
    description: str
    tax_code: Optional[str] = Field(default="")
    notes: Optional[str] = Field(default="")

    @validator("date", pre=True)
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


class PDFTaxProcessor:
    """Process PDF files containing tax-related purchases and generate a CSV report."""

    def __init__(self, input_dir: str, output_file: str):
        self.input_dir = input_dir
        self.output_file = output_file
        self.agent_pool = AgentPool(max_agents=5)
        self.task_manager = TaskManager()
        self.prompt_loader = PromptLoader()

    def _create_pdf_processing_chain(self, llm: BaseLLMInterface) -> SequentialChain:
        """Create a chain for processing PDF content."""
        return SequentialChain(
            steps=[
                ChainStep(
                    task_type="extract_tax_items",
                    output_transform=self._parse_tax_items,
                ),
            ],
            llm=llm,
            prompt_loader=self.prompt_loader,
        )

    def _parse_tax_items(self, response: LLMResponse) -> List[TaxableItem]:
        """Parse the LLM response into TaxableItem objects."""
        try:
            items_data = response.json()
            return [TaxableItem.model_validate(item) for item in items_data["items"]]
        except Exception as e:
            print(f"Error parsing tax items: {e}")
            return []

    def _write_to_csv(self, items: List[TaxableItem]) -> None:
        """Write extracted items to a CSV file."""
        if not items:
            print("No items to write to CSV")
            return

        fieldnames = ["date", "category", "cost", "supplier", "source_file", "description", "tax_code", "notes"]

        with open(self.output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in items:
                writer.writerow(item.to_csv_dict())

    async def process_pdfs(self) -> None:
        """Process all PDF files in the input directory."""
        # Get list of PDF files
        pdf_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print("No PDF files found in the input directory")
            return

        # Create a task for processing PDFs
        pdf_data = {}
        for pdf_file in pdf_files:
            file_path = os.path.join(self.input_dir, pdf_file)
            # Create image input for the PDF
            with open(file_path, "rb") as f:
                pdf_data[pdf_file] = ImageInput(
                    content=f.read(),
                    media_type=MediaType.PDF,
                    file_name=pdf_file,
                )

        # Create and execute task
        task = self.task_manager.create_task(
            task_type="extract_tax_items",
            input_data=pdf_data,
            config={"strategy": "file"},
        )

        # Process the task
        results = self.task_manager.execute(task)

        # Flatten results and write to CSV
        all_items = []
        for result in results.values():
            if isinstance(result, list):
                all_items.extend(result)

        self._write_to_csv(all_items)
        print(f"Processing complete. Results written to {self.output_file}")


async def main():
    """Example usage of the PDF tax processor."""
    # Set up directories
    input_dir = "tax_receipts"
    output_file = "tax_items.csv"

    # Create processor and run
    processor = PDFTaxProcessor(input_dir, output_file)
    await processor.process_pdfs()


if __name__ == "__main__":
    asyncio.run(main())
